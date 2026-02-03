import os
import math
import time
import argparse
import multiprocessing as mp
import inspect
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import models  # <- needed for VGG perceptual loss

# ---- project deps ----
import input_args
import dataloader
from model.SPNADF import SPNADF_Net


# ---- cudnn ----
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ====================== helpers: seed/shape/utils ======================
def seed_worker(worker_id):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] set to {seed}")


def ensure_ch1_5d(x: torch.Tensor) -> torch.Tensor:
    """Normalize to (B,T,1,H,W) for any of 4D/5D input."""
    if x.ndim == 5:
        if x.size(2) != 1:
            x = x[:, :, :1]
        return x
    if x.ndim == 4:
        # (B,1,H,W) -> (B,1,1,H,W) ; (B,T,H,W)->(B,T,1,H,W)
        if x.size(1) == 1:
            return x.unsqueeze(1)
        return x.unsqueeze(2)
    raise RuntimeError(f"Expect 4D/5D, got {tuple(x.shape)}")


def align_seq(a: torch.Tensor, b: torch.Tensor):
    """Align time dimension to the min T, both return (B,T,1,H,W)."""
    a = ensure_ch1_5d(a)
    b = ensure_ch1_5d(b)
    T = min(a.size(1), b.size(1))
    return a[:, :T], b[:, :T]


def to_device_float(t, device):
    return t.to(device, dtype=torch.float32)


# ====================== metrics ======================
def psnr_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction='mean')
    eps = 1e-12
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction='mean')
    return torch.sqrt(mse + 1e-12)


# ====================== spatial grads ======================
def _spatial_grads_forward_diff(x: torch.Tensor):
    """
    Forward diff gradients, keep same spatial size:
      x: (B,T,1,H,W)
      ∇x: diff along W (pad right 1)
      ∇y: diff along H (pad bottom 1)
    """
    gx = x[..., :, 1:] - x[..., :, :-1]
    gx = F.pad(gx, (0, 1, 0, 0, 0, 0))
    gy = x[..., 1:, :] - x[..., :-1, :]
    gy = F.pad(gy, (0, 0, 0, 1, 0, 0))
    return gx, gy


# ====================== unified loss: L1 + grad L1 ======================
def spatial_l1_grad_loss(pred_seq: torch.Tensor, gt_seq: torch.Tensor) -> torch.Tensor:
    """
    mean_t [ |A-B|_1 + |∇xA-∇xB|_1 + |∇yA-∇yB|_1 ]
    时间维使用“升余弦 + 归一化”权重，并做轻微幂次压缩。
    """
    A, B = align_seq(pred_seq, gt_seq)  # (B,T,1,H,W)

    l1_t = torch.abs(A - B).mean(dim=(0, 2, 3, 4))  # (T,)

    Agx, Agy = _spatial_grads_forward_diff(A)
    Bgx, Bgy = _spatial_grads_forward_diff(B)
    l1_gx_t = torch.abs(Agx - Bgx).mean(dim=(0, 2, 3, 4))  # (T,)
    l1_gy_t = torch.abs(Agy - Bgy).mean(dim=(0, 2, 3, 4))  # (T,)

    per_t = l1_t + l1_gx_t + l1_gy_t

    T = per_t.shape[0]
    if T == 1:
        return per_t.mean()

    device, dtype = per_t.device, per_t.dtype
    u = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
    w = 0.5 * (1.0 - torch.cos(math.pi * u))

    alpha = 0.1
    w = w ** alpha
    w = w / (w.sum() + 1e-12)

    return (per_t * w).sum()


def ms_weighted_loss(scales, gt_seq, weights):
    """
    scales: list[ (B,T,1,H,W) ]，从“最深”到“最浅”，已经是 full-res
    gt_seq: (B,T,1,H,W)
    weights: list[float] 或 None，对应各尺度权重（从最深到最浅）

    每个尺度的 GT 做低通（下采样再上采样），避免粗尺度拟合细节。
    """
    gt_seq = ensure_ch1_5d(gt_seq)
    device = gt_seq.device
    dtype = gt_seq.dtype

    if not scales:
        return torch.zeros((), device=device, dtype=dtype)

    S = len(scales)

    if weights is None:
        ws = [1.0] * S
    else:
        ws = list(weights)
        if len(ws) < S:
            ws = ws + [ws[-1]] * (S - len(ws))
        elif len(ws) > S:
            ws = ws[:S]

    total_loss = None
    total_w = 0.0

    B, T, C, H, W = gt_seq.shape

    for s in range(S):
        w = float(ws[s])
        if w <= 0:
            continue

        factor = 2 ** (S - 1 - s)  # 最深尺度最大，最浅尺度=1
        if factor > 1:
            gt_bt = gt_seq.view(B * T, C, H, W)
            gt_down = F.avg_pool2d(gt_bt, kernel_size=factor, stride=factor)
            gt_up = F.interpolate(gt_down, size=(H, W), mode='bilinear', align_corners=False).view(B, T, C, H, W)
        else:
            gt_up = gt_seq

        loss_s = spatial_l1_grad_loss(scales[s], gt_up)

        total_loss = w * loss_s if total_loss is None else (total_loss + w * loss_s)
        total_w += w

    if total_loss is None or total_w == 0:
        return torch.zeros((), device=device, dtype=dtype)

    return total_loss / total_w


# ====================== Perceptual loss (VGG, gray→RGB) ======================
class VGGPerceptualLossGray(nn.Module):
    _CUTS = {"relu1_2": 4, "relu2_2": 9, "relu3_3": 16, "relu4_3": 23, "relu5_3": 30}

    def __init__(self, device, value_range: float = 1.0, layer: str = "relu3_3"):
        super().__init__()
        try:
            vgg_feat = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg_feat = models.vgg16(pretrained=True).features

        cut = self._CUTS.get(layer, 16)
        self.vgg = vgg_feat[:cut].to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1),
        )
        self.value_range = float(value_range)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_ch1_5d(x)  # (B,T,1,H,W)
        x = torch.clamp(x / max(self.value_range, 1e-6), 0.0, 1.0)
        B, T, _, H, W = x.shape
        x3 = x.expand(B, T, 3, H, W).contiguous().view(B * T, 3, H, W)
        x3 = (x3 - self.mean) / self.std
        return x3

    def forward(self, pred_seq: torch.Tensor, gt_seq: torch.Tensor) -> torch.Tensor:
        A, B = align_seq(pred_seq, gt_seq)
        fA = self.vgg(self._prep(A))
        fB = self.vgg(self._prep(B))
        return F.l1_loss(fA, fB)


# ====================== reflectivity scaling (safe) ======================
def get_ref_ms_true(out: dict):
    """
    out["ref_muti"] or out["ref_multi"]: list of (B,T,1,H,W)
    return: list of (B,T,1,H,W)
    """
    ref = out.get("ref_muti", None)
    if ref is None:
        ref = out.get("ref_multi", None)
    if ref is None:
        return []
    return [ensure_ch1_5d(r) for r in ref]


# ====================== validation: per-noise + avg ======================
@torch.no_grad()
def validate_one_noise(args, v_loader, model: nn.Module, iteration: int, writer: SummaryWriter, noise_scale: float):
    model.eval()

    data_range = float(getattr(args, 'depth_range', 1.0))
    refl_range = float(getattr(args, 'refl_range', 1.0))

    psnr_sum = 0.0
    rmse_sum = 0.0
    ref_psnr_sum = 0.0
    ref_loss_sum = 0.0
    w_loss_sum = 0.0
    count = 0

    for data in v_loader:
        bin_seq, depth_stamp_seq, depth_stamp_seq_norm, gtr_seq, gtd_seq, sigprob_seq, B_divP, _ = data

        bin_seq = to_device_float(bin_seq, args.device)
        depth_stamp_seq = to_device_float(depth_stamp_seq, args.device)
        depth_stamp_seq_norm = to_device_float(depth_stamp_seq_norm, args.device)
        gtr_seq = to_device_float(gtr_seq, args.device)
        gtd_seq = to_device_float(gtd_seq, args.device)
        sigprob_seq = to_device_float(sigprob_seq, args.device)
        B_divP = to_device_float(B_divP, args.device)

        out = model(
            x=depth_stamp_seq_norm,
            depth_stamp_seq=depth_stamp_seq,
            bin_seq=bin_seq,
            B_divP=B_divP,
            args=args,
            reset_state=True,
        )

        y_final = out['y_final']
        depth_ms = out.get('depth_multi', [])

        pred_5d, gt_5d = align_seq(depth_ms[-1] if depth_ms else y_final, gtd_seq)

        TT = pred_5d.shape[1]
        last_pred = pred_5d[:, TT - 1:TT]
        last_gt = gt_5d[:, TT - 1:TT]

        psnr_sum += psnr_torch(last_pred, last_gt, data_range=data_range).item()
        rmse_sum += rmse_torch(last_pred, last_gt).item()

        prob_ms = out.get('prob_multi', [])
        w_loss = ms_weighted_loss(prob_ms, sigprob_seq, args.ms_prob_weights)
        w_loss_sum += w_loss.item()

        ref_ms_true = get_ref_ms_true(out)
        if ref_ms_true:
            ref_main = ref_ms_true[-1]
            ref_pred_5d, ref_gt_5d = align_seq(ref_main, gtr_seq)
            ref_loss = spatial_l1_grad_loss(ref_pred_5d, ref_gt_5d)
            ref_loss_sum += ref_loss.item()

            TT_ref = ref_pred_5d.shape[1]
            last_ref_pred = ref_pred_5d[:, TT_ref - 1:TT_ref]
            last_ref_gt = ref_gt_5d[:, TT_ref - 1:TT_ref]
            ref_psnr_sum += psnr_torch(last_ref_pred, last_ref_gt, data_range=refl_range).item()

        count += 1

    mean_psnr = psnr_sum / max(count, 1)
    mean_rmse = rmse_sum / max(count, 1)
    mean_wloss = w_loss_sum / max(count, 1)
    mean_refpsnr = ref_psnr_sum / max(count, 1)
    mean_refloss = ref_loss_sum / max(count, 1)

    tag = f"tm_bkgx{noise_scale:g}"
    writer.add_scalar(f"val/{tag}_psnr_last", mean_psnr, iteration)
    writer.add_scalar(f"val/{tag}_rmse_last", mean_rmse, iteration)
    writer.add_scalar(f"val/{tag}_w_loss", mean_wloss, iteration)
    writer.add_scalar(f"val/{tag}_ref_psnr_last", mean_refpsnr, iteration)
    writer.add_scalar(f"val/{tag}_ref_loss", mean_refloss, iteration)

    print(
        f"[Validation][tm][bkg x{noise_scale:g}] "
        f"PSNR(depth,last)={mean_psnr:.3f} | RMSE(depth,last)={mean_rmse:.4e} | "
        f"w-loss(prob,ms)={mean_wloss:.4e} | "
        f"PSNR(ref,last)={mean_refpsnr:.3f} | ref-loss={mean_refloss:.4e}"
    )

    return mean_psnr, mean_refpsnr


@torch.no_grad()
def validate_multi_noise(args, model: nn.Module, iteration: int, writer: SummaryWriter):
    scales = [float(x) for x in getattr(args, "val_noise_scales", [1.0])]
    dep_list, ref_list = [], []

    for s in scales:
        setattr(args, "val_fixed_noise_scale", float(s))
        vset = dataloader.val_dataloader(args)
        v_loader = DataLoader(dataset=vset, num_workers=0, batch_size=1, shuffle=True)

        dpsnr, rpsnr = validate_one_noise(args, v_loader, model, iteration, writer, noise_scale=float(s))
        dep_list.append(dpsnr)
        ref_list.append(rpsnr)

    dep_avg = float(np.mean(dep_list)) if dep_list else 0.0
    ref_avg = float(np.mean(ref_list)) if ref_list else 0.0

    writer.add_scalar("val/tm_psnr_last_avg_over_noise", dep_avg, iteration)
    writer.add_scalar("val/tm_ref_psnr_last_avg_over_noise", ref_avg, iteration)

    print(f"[Validation][tm] AVG over noise scales: PSNR(depth,last)={dep_avg:.3f} | PSNR(ref,last)={ref_avg:.3f}")
    return dep_avg, ref_avg


# ====================== visualization (optional, per-noise subfolder) ======================
@torch.no_grad()
def save_tm_plots(args, model: nn.Module, v_loader, base_dir: str, epoch: int, noise_scale: float, max_T: int = 360):
    model.eval()
    os.makedirs(base_dir, exist_ok=True)

    sub_dir = os.path.join(base_dir, f"bkgx{noise_scale:g}", "all")
    os.makedirs(sub_dir, exist_ok=True)

    def _np(t: torch.Tensor):
        return t.detach().cpu().numpy()

    saved = 0
    max_save = int(getattr(args, "plot_max_save", 10))

    for idx, data in enumerate(v_loader):
        if idx % 5 != 0:
            continue

        bin_seq, depth_stamp_seq, depth_stamp_seq_norm, gtr_seq, gtd_seq, sigprob_seq, B_divP, _ = data

        bin_seq = to_device_float(bin_seq, args.device)
        depth_stamp_seq = to_device_float(depth_stamp_seq, args.device)
        depth_stamp_seq_norm = to_device_float(depth_stamp_seq_norm, args.device)
        gtr_seq = to_device_float(gtr_seq, args.device)
        gtd_seq = to_device_float(gtd_seq, args.device)
        sigprob_seq = to_device_float(sigprob_seq, args.device)
        B_divP = to_device_float(B_divP, args.device)

        out_all = model(
            x=depth_stamp_seq_norm,
            depth_stamp_seq=depth_stamp_seq,
            bin_seq=bin_seq,
            B_divP=B_divP,
            args=args,
            reset_state=True,
        )

        d_pred = out_all['y_final']
        depth_ms = out_all.get('depth_multi', [])
        prob_ms = out_all.get('prob_multi', [])
        uncertain = out_all.get('uncertain', None)
        wobs = out_all.get('wobs', None)

        ref_ms = get_ref_ms_true(out_all)

        d_hat_tm_main = depth_ms[-1] if depth_ms else d_pred
        prob_main = prob_ms[-1] if prob_ms else None
        ref_main = ref_ms[-1] if ref_ms else None

        t_in, _ = align_seq(depth_stamp_seq_norm, depth_stamp_seq_norm)

        refl_gt, _ = align_seq(gtr_seq, gtr_seq)
        dep_gt, _ = align_seq(gtd_seq, gtd_seq)
        pgt, _ = align_seq(sigprob_seq, sigprob_seq)

        d_pred, _ = align_seq(d_pred, t_in)
        d_hat_tm_main, _ = align_seq(d_hat_tm_main, t_in)

        if prob_main is not None:
            prob_main, _ = align_seq(prob_main, t_in)
        if ref_main is not None:
            ref_main, _ = align_seq(ref_main, t_in)
        if uncertain is not None:
            uncertain, _ = align_seq(uncertain, t_in)
        bin_seq_aligned, _ = align_seq(bin_seq, t_in)
        if wobs is not None:
            wobs, _ = align_seq(wobs, t_in)

        in_np = _np(t_in[0])
        refl_np = _np(refl_gt[0])
        gt_np = _np(dep_gt[0])
        pgt_np = _np(pgt[0])

        pred_np = _np(d_pred[0])
        tm_main_np = _np(d_hat_tm_main[0])

        ref_main_np = _np(ref_main[0]) if ref_main is not None else None
        prob_main_np = _np(prob_main[0]) if prob_main is not None else None

        uncert_np = _np(uncertain[0]) if uncertain is not None else None
        bin_np = _np(bin_seq_aligned[0])
        wobs_np = _np(wobs[0]) if wobs is not None else None

        def _norm01(a):
            a = a.astype(np.float32)
            mn, mx = float(a.min()), float(a.max())
            if mx <= mn:
                return np.zeros_like(a, dtype=np.float32)
            return (a - mn) / (mx - mn + 1e-12)

        err_pred = _norm01(np.abs(pred_np - gt_np).astype(np.float32))

        T_total = in_np.shape[0]
        Tshow = min(T_total, max_T)

        depth_block = [
            ("GT Depth", np.stack([_norm01(gt_np[t, 0]) for t in range(T_total)], 0)[:, None]),
            ("KF depth (y_final)", np.stack([_norm01(pred_np[t, 0]) for t in range(T_total)], 0)[:, None]),
            ("TM depth (main)", np.stack([_norm01(tm_main_np[t, 0]) for t in range(T_total)], 0)[:, None]),
            ("|KF depth − GT|", np.stack([err_pred[t, 0] for t in range(T_total)], 0)[:, None]),
        ]

        refl_block = []
        if ref_main_np is not None:
            refl_block = [
                ("Reflectivity GT", np.stack([_norm01(refl_np[t, 0]) for t in range(T_total)], 0)[:, None]),
                ("Pred refl (main)", np.stack([_norm01(ref_main_np[t, 0]) for t in range(T_total)], 0)[:, None]),
            ]

        prob_block = [("GT cond. prob", np.stack([_norm01(pgt_np[t, 0]) for t in range(T_total)], 0)[:, None])]
        if prob_main_np is not None:
            prob_block.append(
                ("Posterior w (main)", np.stack([_norm01(prob_main_np[t, 0]) for t in range(T_total)], 0)[:, None])
            )

        misc_block = [
            ("Input (depth_stamp_seq_norm)", np.stack([_norm01(in_np[t, 0]) for t in range(T_total)], 0)[:, None]),
            ("bin_seq", np.stack([_norm01(bin_np[t, 0]) for t in range(T_total)], 0)[:, None]),
        ]
        if wobs_np is not None:
            misc_block.append(("wobs", np.stack([_norm01(wobs_np[t, 0]) for t in range(T_total)], 0)[:, None]))
        if uncert_np is not None:
            misc_block.append(("Uncertainty", np.stack([_norm01(uncert_np[t, 0]) for t in range(T_total)], 0)[:, None]))

        rows_data = depth_block + refl_block + prob_block + misc_block
        R = len(rows_data)

        fig, axs = plt.subplots(R, Tshow, figsize=(2.4 * Tshow, R * 1.8), constrained_layout=False)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.06, wspace=0.03, hspace=0.08)

        for i in range(Tshow):
            axs[0, i].set_title(f"t={i}", fontsize=12)

        for r, (title, arr) in enumerate(rows_data):
            for i in range(Tshow):
                axs[r, i].imshow(arr[i, 0], cmap='gray', interpolation='nearest')
                axs[r, i].axis('off')
            axs[r, 0].set_ylabel(title, rotation=0, labelpad=14, fontsize=12, va='center')

        fig.suptitle(f'TM | Epoch {epoch + 1} | bkg x{noise_scale:g}', fontsize=16, y=0.98)

        out_path = os.path.join(sub_dir, f'epoch_{epoch + 1:04d}_idx{idx:04d}.png')
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f'[Plot][bkg x{noise_scale:g}] Saved {out_path}')

        if wobs is not None:
            wobs_save_path = os.path.join(sub_dir, f'epoch_{epoch + 1:04d}_idx{idx:04d}_wobs.pt')
            torch.save(wobs[0].detach().cpu().to(torch.float16), wobs_save_path)

        saved += 1
        if saved >= max_save:
            break


# ====================== checkpoint I/O ======================
def get_trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def save_checkpoint(args, model: nn.Module, optimizer, epoch, iteration, best_metric_dep, best_metric_ref, tag="latest"):
    os.makedirs(args.weights_dir, exist_ok=True)
    model_ref = model.module if hasattr(model, "module") else model

    state = {
        'epoch': epoch,
        'iter': iteration,
        'best_metric_dep': float(best_metric_dep),
        'best_metric_ref': float(best_metric_ref),
        # for compatibility: try to save submodules if exist
        'tm': model_ref.tm.state_dict() if hasattr(model_ref, "tm") else None,
        'lin': model_ref.lin.state_dict() if hasattr(model_ref, "lin") else None,
        'optimizer': optimizer.state_dict(),
    }
    path = os.path.join(args.weights_dir, f'ckpt_{tag}.pt')
    torch.save(state, path)
    print(f"[Checkpoint] Saved -> {path}")


def load_ckpt(model: nn.Module, path, optimizer=None, map_location="cpu", strict=True):
    sd = torch.load(path, map_location=map_location)
    model_ref = model.module if hasattr(model, "module") else model

    if isinstance(sd, dict) and any(k in sd for k in ["tm", "lin", "optimizer"]):
        if "tm" in sd and sd["tm"] is not None and hasattr(model_ref, "tm"):
            model_ref.tm.load_state_dict(sd["tm"], strict=strict)
        if "lin" in sd and sd["lin"] is not None and hasattr(model_ref, "lin"):
            model_ref.lin.load_state_dict(sd["lin"], strict=strict)
    elif isinstance(sd, dict) and "state_dict" in sd:
        model_ref.load_state_dict(sd["state_dict"], strict=strict)
    else:
        model_ref.load_state_dict(sd, strict=strict)

    epoch = sd.get("epoch") if isinstance(sd, dict) else None
    it = sd.get("iter") if isinstance(sd, dict) else None
    best_dep = sd.get("best_metric_dep") if isinstance(sd, dict) else None
    best_ref = sd.get("best_metric_ref") if isinstance(sd, dict) else None

    if optimizer is not None and isinstance(sd, dict) and "optimizer" in sd:
        try:
            optimizer.load_state_dict(sd["optimizer"])
            print("[Checkpoint] Optimizer loaded.")
        except Exception as e:
            print(f"[Checkpoint] Skip optimizer load: {e}")

    return epoch, it, best_dep, best_ref


# ====================== model builder (robust) ======================
def build_model_from_args(args) -> nn.Module:
    """
    兼容不同构造函数签名：尽可能把 args 里同名字段喂进去。
    """
    cls = SPNADF_Net
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name in ("self",):
            continue
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
    # 常见输入通道字段兜底
    if "in_ch" in sig.parameters and "in_ch" not in kwargs:
        kwargs["in_ch"] = 1
    if "args" in sig.parameters and "args" not in kwargs:
        kwargs["args"] = args

    try:
        model = cls(**kwargs)
    except TypeError:
        # 最保底：只传 args
        try:
            model = cls(args)
        except Exception:
            model = cls()
    return model.to(args.device)


# ====================== main train (TM only) ======================
def main(args):
    base_dir = os.path.join(os.getcwd(), 'rnn_result')
    plot_save_dir = os.path.join(base_dir, 'plot_save')
    weights_dir = os.path.join(base_dir, 'weight')
    os.makedirs(plot_save_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    args.plotdir = plot_save_dir
    args.weights_dir = weights_dir

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=os.path.join(args.log_dir, 'TM'))

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    # curriculum-safe global counter across dataloader workers
    shared_seen = mp.Value('l', 22650 * 6)
    shared_lock = mp.Lock()

    trainset = dataloader.train_dataloader(args, shared_seen=shared_seen, shared_lock=shared_lock)

    seed = int(getattr(args, "seed", 112))
    t_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=(16 > 0),
        pin_memory=True,
    )

    model = build_model_from_args(args).to(args.device)
    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_count)]).to(args.device)

    perc_depth = VGGPerceptualLossGray(args.device, value_range=args.depth_range, layer=args.perc_layer)
    perc_ref = VGGPerceptualLossGray(args.device, value_range=getattr(args, 'refl_range', 1.0), layer=args.perc_layer)

    optimizer = torch.optim.Adam(get_trainable_parameters(model), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 以 ref PSNR 为准越大越好（你原来就是 step(metric_ref_avg)）
        factor=0.5,
        patience=8,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=float(args.eta_min),
        eps=1e-8,
    )
    print(
        f"[LR] ReduceLROnPlateau: mode=max factor=0.5 patience=8 threshold=1e-4 "
        f"min_lr={float(args.eta_min)} base_lr={args.lr}"
    )

    # resume
    if args.resume_path is not None and os.path.isfile(args.resume_path):
        _, iteration, best_metric_dep, best_metric_ref = load_ckpt(model, args.resume_path, optimizer)
        if iteration is None:
            iteration = 0
    else:
        iteration, best_metric_dep, best_metric_ref = 0, None, None

    if best_metric_dep is None:
        best_metric_dep = -1e9
    if best_metric_ref is None:
        best_metric_ref = -1e9

    print(f"==> Training TM-only | TrainNoiseRange={getattr(args,'train_noise_scales',None)} "
          f"| ValNoiseScales={getattr(args,'val_noise_scales',[1.0])}")

    for epoch in range(args.start_epoch, args.total_epochs):
        print(f"\nEpoch {epoch + 1}/{args.total_epochs}\n---------------------------------")
        epoch_loss = 0.0
        start_time = time.time()

        model.train()

        it = iter(t_loader)
        pbar = tqdm(range(len(t_loader)), dynamic_ncols=True)

        for _ in pbar:
            # 你当前固定 T_batch=1（如果以后要随机帧长，可以恢复 dataset.num_frames_cfg 的写法）
            T_batch = 1

            data = next(it)
            bin_seq, depth_stamp_seq, depth_stamp_seq_norm, gtr_seq, gtd_seq, sigprob_seq, B_divP, scale = data

            bin_seq = to_device_float(bin_seq, args.device)
            depth_stamp_seq = to_device_float(depth_stamp_seq, args.device)
            depth_stamp_seq_norm = to_device_float(depth_stamp_seq_norm, args.device)
            gtr_seq = to_device_float(gtr_seq, args.device)
            gtd_seq = to_device_float(gtd_seq, args.device)
            sigprob_seq = to_device_float(sigprob_seq, args.device)
            B_divP = to_device_float(B_divP, args.device)

            optimizer.zero_grad(set_to_none=True)

            out = model(
                x=depth_stamp_seq_norm,
                depth_stamp_seq=depth_stamp_seq,
                bin_seq=bin_seq,
                B_divP=B_divP,
                args=args,
                reset_state=True,
            )

            y_final = out['y_final']
            depth_ms = out.get('depth_multi', [])
            prob_ms = out.get('prob_multi', [])
            ref_ms = get_ref_ms_true(out)

            # ---- TM-only losses ----
            L_main = spatial_l1_grad_loss(y_final, gtd_seq)
            L_tm = ms_weighted_loss(depth_ms, gtd_seq, args.ms_depth_weights)
            L_w = ms_weighted_loss(prob_ms, sigprob_seq, args.ms_prob_weights)
            L_ref = ms_weighted_loss(ref_ms, gtr_seq, args.ms_prob_weights) if ref_ms else torch.zeros((), device=args.device)

            L_perc_d = torch.zeros((), device=args.device)
            L_perc_r = torch.zeros((), device=args.device)

            if args.lambda_perc_depth > 0:
                depth_main = depth_ms[-1] if depth_ms else y_final
                d_ma, d_gt = align_seq(depth_main, gtd_seq)
                L_perc_d = perc_depth(d_ma, d_gt)

            if args.lambda_perc_ref > 0 and ref_ms:
                ref_main = ref_ms[-1]
                r_ma, r_gt = align_seq(ref_main, gtr_seq)
                L_perc_r = perc_ref(r_ma, r_gt)

            loss = (args.lambda_main * L_main
                    + args.lambda_tm * L_tm
                    + args.lambda_w * L_w
                    + args.lambda_ref * L_ref
                    + args.lambda_perc_depth * L_perc_d
                    + args.lambda_perc_ref * L_perc_r)

            if (iteration % args.print_every) == 0:
                print(f"[Iter {iteration}] "
                      f"L_main={args.lambda_main * L_main:.4e} | "
                      f"L_tm={args.lambda_tm * L_tm:.4e} | "
                      f"L_w={args.lambda_w * L_w:.4e} | "
                      f"L_ref={args.lambda_ref * L_ref:.4e} | "
                      f"L_perc_d={args.lambda_perc_depth * L_perc_d:.4e} | "
                      f"L_perc_r={args.lambda_perc_ref * L_perc_r:.4e} | "
                      f"LR={optimizer.param_groups[0]['lr']:.3e} | "
                      f"scale={float(scale.mean().item()):.3f} | "
                      f"T_batch={T_batch}")

            # ---- logs ----
            writer.add_scalar('train/L_main', float(L_main.item()), iteration)
            writer.add_scalar('train/L_tm', float(L_tm.item()), iteration)
            writer.add_scalar('train/L_w', float(L_w.item()), iteration)
            writer.add_scalar('train/L_ref', float(L_ref.item()), iteration)
            writer.add_scalar('train/L_perc_d', float(L_perc_d.item()), iteration)
            writer.add_scalar('train/L_perc_r', float(L_perc_r.item()), iteration)
            writer.add_scalar('train/noise_scale', float(scale.mean().item()), iteration)

            loss.backward()
            clip_grad_norm_(get_trainable_parameters(model), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            writer.add_scalar('train/tm_loss', float(loss.item()), iteration)
            writer.add_scalar('train/lr', float(optimizer.param_groups[0]['lr']), iteration)
            iteration += 1

        # epoch end
        epoch_time = time.time() - start_time
        mean_loss = epoch_loss / max(1, len(t_loader))
        writer.add_scalar('train/tm_epoch_loss', mean_loss, epoch)
        writer.add_scalar('train/epoch_time_sec', epoch_time, epoch)
        print(f"[Epoch {epoch + 1}] TM loss={mean_loss:.4f} time={epoch_time:.1f}s")

        # ---- multi-noise validation every val_every epochs (and last epoch) ----
        do_val = ((epoch + 1) % int(getattr(args, "val_every", 5)) == 0) or ((epoch + 1) == args.total_epochs)
        if do_val:
            metric_dep_avg, metric_ref_avg = validate_multi_noise(args, model, iteration, writer)

            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(metric_ref_avg)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[LR] ReduceLROnPlateau: {old_lr:.6e} -> {new_lr:.6e} (metric_ref_avg={metric_ref_avg:.4f})")
            else:
                print(f"[LR] ReduceLROnPlateau: keep {new_lr:.6e} (metric_ref_avg={metric_ref_avg:.4f})")
            writer.add_scalar('train/lr_after_plateau_step', float(new_lr), epoch)

            improved_dep = metric_dep_avg > best_metric_dep
            improved_ref = metric_ref_avg > best_metric_ref

            if improved_dep:
                best_metric_dep = metric_dep_avg
                save_checkpoint(
                    args, model, optimizer, epoch, iteration,
                    best_metric_dep, best_metric_ref,
                    tag='best_depth_tm_avg',
                )

            if improved_ref:
                best_metric_ref = metric_ref_avg
                save_checkpoint(
                    args, model, optimizer, epoch, iteration,
                    best_metric_dep, best_metric_ref,
                    tag='best_ref_tm_avg',
                )

            save_checkpoint(args, model, optimizer, epoch, iteration, best_metric_dep, best_metric_ref, tag='latest')

            if ((epoch + 1) % 50) == 0:
                save_checkpoint(args, model, optimizer, epoch, iteration, best_metric_dep, best_metric_ref,
                                tag=f'epoch_{epoch + 1:04d}')

            # plots per noise scale
            for s in [float(x) for x in getattr(args, "val_noise_scales", [1.0])]:
                if int(getattr(args, "plot_max_save", 0)) <= 0:
                    break
                setattr(args, "val_fixed_noise_scale", float(s))
                vset = dataloader.val_dataloader(args)
                v_loader = DataLoader(dataset=vset, num_workers=0, batch_size=1, shuffle=False)
                save_tm_plots(
                    args, model, v_loader, args.plotdir, epoch,
                    noise_scale=float(s),
                    max_T=min(360, getattr(args, 'max_T', 360)),
                )
        else:
            print(f"[Epoch {epoch + 1}] skip validation/plots (val_every={getattr(args,'val_every',5)})")


# ====================== arg parser ======================
if __name__ == '__main__':
    set_seed(112)

    parser = argparse.ArgumentParser(description='TM-only training + multi-noise val + ReduceLROnPlateau')

    # dataset/sensor/training from your project
    input_args.training_args(parser)
    input_args.sensor_args(parser)

    # model config (keep what you used before; constructor is handled robustly)
    parser.add_argument('--base', type=int, default=32, help='base channels')
    parser.add_argument('--k', type=int, default=4, help='block size k')
    parser.add_argument('--rank', type=int, default=2, help='low-rank r')
    parser.add_argument('--mode', type=str, default='4', choices=['4', '8'], help='neighborhood mode')

    # ranges
    parser.add_argument('--depth_range', type=float, default=66.6)
    parser.add_argument('--refl_range', type=float, default=1.0)

    # loss weights (TM-only)
    parser.add_argument('--lambda_w', type=float, default=0)
    parser.add_argument('--lambda_ref', type=float, default=50)
    parser.add_argument('--lambda_tm', type=float, default=2.0)
    parser.add_argument('--lambda_main', type=float, default=0.0)

    # perceptual
    parser.add_argument('--lambda_perc_depth', type=float, default=0)
    parser.add_argument('--lambda_perc_ref', type=float, default=1)
    parser.add_argument('--perc_layer', type=str, default='relu3_3',
                        choices=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])

    # multi-scale weights
    parser.add_argument('--ms_depth_weights', type=float, nargs='+', default=[0.125, 0.25, 0.5, 2.0])
    parser.add_argument('--ms_prob_weights', type=float, nargs='+', default=[0.125, 0.25, 0.5, 2.0])

    # resume/train
    parser.add_argument('--resume_path', type=str, default='rnn_result/weight/ckpt_best_ref_tm_avg.pt')
    parser.add_argument('--max_T', type=int, default=360)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eta_min', type=float, default=1e-10)
    parser.add_argument('--print_every', type=int, default=25)

    # validation cadence
    parser.add_argument('--val_every', type=int, default=3, help='run multi-noise validation every N epochs')

    # train noise curriculum
    parser.add_argument('--train_curriculum', type=int, default=1, help='1: enable curriculum Uniform expansion, 0: disable')
    parser.add_argument('--train_p_reach_max', type=float, default=0.25, help='progress p where curriculum reaches max range')
    parser.add_argument('--train_noise_scales', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                        help='(optional) endpoints list; will be overridden by train_noise_min/max if both provided')
    parser.add_argument('--train_noise_mode', type=str, default='random', choices=['random', 'fixed'],
                        help='only used when train_curriculum=0')

    # val noise sweep
    parser.add_argument('--val_noise_scales', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                        help='validate by sweeping these noise scales and averaging metrics')
    parser.add_argument('--val_noise_mode', type=str, default='fixed', choices=['fixed', 'random'])
    parser.add_argument('--val_fixed_noise_scale', type=float, default=None)

    # plot control
    parser.add_argument('--plot_max_save', type=int, default=10,
                        help='max number of plot images to save per noise scale (<=0 disables plotting)')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[64, 128],
                        help='patch size [H W] for random crop')

    args = parser.parse_args()
    main(args)
