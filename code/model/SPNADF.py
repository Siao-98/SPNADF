from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.UNetRecurrent import IntensityDepthRecurrent


# ====================== Pixel-wise mixture Kalman (drop blocks; keep logic) ======================
def kalman_update_pix_mixture(
    xm: torch.Tensor,     # (B,1,H,W)
    z: torch.Tensor,      # (B,1,H,W)
    Pm: torch.Tensor,     # (B,1,H,W)
    U: torch.Tensor,      # (B,1,H,W)
    bin: torch.Tensor,    # (B,1,H,W)
    w: torch.Tensor,      # (B,1,H,W)
    Dmax,
    eps: float = 1e-6,
    tau: float = 1.0,
    var_floor: float = 1e-6,
):
    # light clamp for stability (no PSD/jitter etc.)
    Pm = Pm.clamp_min(var_floor)
    U  = U.clamp_min(var_floor)

    mask = (bin > 0).to(dtype=xm.dtype)
    z_eff = torch.where(mask.bool(), z, xm)

    r = z_eff - xm
    S = (Pm + U).clamp_min(var_floor)

    K = (Pm / S).clamp(0.0, 1.0)
    mu_s  = xm + K * r
    var_s = (1.0 - K) * Pm

    mu_b  = xm
    var_b = Pm

    logN = -0.5 * (math.log(2.0 * math.pi) + torch.log(S) + (r * r) / S)

    w_pix = w.clamp(1e-12, 1.0 - 1e-12)

    if isinstance(Dmax, torch.Tensor):
        Dmax_f = Dmax
    else:
        Dmax_f = torch.as_tensor(Dmax, device=xm.device, dtype=xm.dtype)
    Dmax_f = (Dmax_f + 0.0 * xm).clamp_min(1e-30)
    log_pb = -torch.log(Dmax_f)

    logit = (torch.log(w_pix) + logN) - (torch.log1p(-w_pix) + log_pb)
    if tau != 1.0:
        logit = logit / tau

    rho = torch.sigmoid(logit) * mask

    mu = rho * mu_s + (1.0 - rho) * mu_b
    var = rho * (var_s + (mu_s - mu).pow(2)) + (1.0 - rho) * (var_b + (mu_b - mu).pow(2))
    var = var.clamp_min(var_floor)

    # --- add here ---
    no_echo = (bin <= 0)
    mu = torch.where(no_echo, xm, mu)
    var = torch.where(no_echo, Pm, var)
    rho = torch.where(no_echo, torch.zeros_like(rho), rho)
    # --- end ---

    create_new = rho * (K * z_eff)
    return mu, var, rho, create_new


# ====================== Helpers: recurrent states (detach + downsample + expand) ======================
def _detach_recurrent_states(states, is_lstm: bool):
    if states is None:
        return None
    out = []
    for s in states:
        if s is None:
            out.append(None)
        else:
            if is_lstm:
                h, c = s
                out.append((h.detach(), c.detach()))
            else:
                out.append(s.detach())
    return out


def _downsample_recurrent_states(states, scale: int, is_lstm: bool, mode: str = "bilinear"):
    """
    Downsample spatial dims by 'scale' for every level (approx).
    states: list(len=num_encoders) of None / (h,c) / h
    """
    if states is None or scale == 1:
        return states
    out = []
    for s in states:
        if s is None:
            out.append(None)
            continue
        if is_lstm:
            h, c = s
            h2 = F.interpolate(h, scale_factor=1.0 / scale, mode=mode, align_corners=False)
            c2 = F.interpolate(c, scale_factor=1.0 / scale, mode=mode, align_corners=False)
            out.append((h2, c2))
        else:
            h = s
            h2 = F.interpolate(h, scale_factor=1.0 / scale, mode=mode, align_corners=False)
            out.append(h2)
    return out


def _expand_recurrent_states(states_lr, K: int, B: int, is_lstm: bool):
    """
    Expand LR states from (B,...) to (K*B,...), matching the batched MC probes.
    """
    if states_lr is None:
        return None
    out = []
    for s in states_lr:
        if s is None:
            out.append(None)
            continue
        if is_lstm:
            h, c = s
            hK = h.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, *h.shape[1:])
            cK = c.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, *c.shape[1:])
            out.append((hK, cK))
        else:
            h = s
            hK = h.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, *h.shape[1:])
            out.append(hK)
    return out


def _save_step_counter(tm: nn.Module):
    """
    Save tm.backbone._step_counter if it exists.
    Returns (tensor_clone or None, counter_ref or None).
    """
    if hasattr(tm, "backbone") and hasattr(tm.backbone, "_step_counter"):
        ref = tm.backbone._step_counter
        return ref.detach().clone(), ref
    return None, None


def _restore_step_counter(saved, ref):
    if saved is not None and ref is not None:
        ref.copy_(saved)


# ====================== MC uncertainty propagation (pixel-wise; LR optional) ======================
@torch.no_grad()
def propagate_uncertainty_tm_mc_batch_states_lr(
    tm: nn.Module,
    D_in: torch.Tensor,                  # (B,1,H,W)
    photon_in_fixed: torch.Tensor,       # (B,4,H,W)
    D_obs_fixed: torch.Tensor,           # (B,1,H,W)
    Umap_fixed: torch.Tensor,            # (B,1,H,W)
    intensity_baseline_fixed=None,       # (B,1,H,W) or None
    P_prior_pix: Optional[torch.Tensor] = None,  # (B,1,H,W)
    prev_states_photon_fixed=None,       # list(len=num_encoders)
    prev_states_tof_fixed=None,          # list(len=num_encoders)
    k_probe: int = 8,
    eps: float = 1e-12,
    baseline_mode: Literal["fixed", "same_as_D"] = "fixed",
    downsample_scale: int = 2,
    ds_mode: str = "bilinear",
):
    B, _, H, W = D_in.shape
    K = int(k_probe)
    ds = int(downsample_scale)

    if P_prior_pix is None:
        P_prior_pix = torch.full_like(D_in, 1e-6)

    is_lstm = getattr(getattr(tm, "backbone", None), "is_lstm", False)

    # ---- downsample inputs ----
    if ds > 1:
        Hlr, Wlr = H // ds, W // ds
        D_lr    = F.interpolate(D_in,            size=(Hlr, Wlr), mode=ds_mode, align_corners=False)
        Dobs_lr = F.interpolate(D_obs_fixed,     size=(Hlr, Wlr), mode=ds_mode, align_corners=False)
        U_lr    = F.interpolate(Umap_fixed,      size=(Hlr, Wlr), mode=ds_mode, align_corners=False)
        ph_lr   = F.interpolate(photon_in_fixed, size=(Hlr, Wlr), mode=ds_mode, align_corners=False)
        P_lr    = F.interpolate(P_prior_pix,     size=(Hlr, Wlr), mode=ds_mode, align_corners=False)
        I_lr = None if intensity_baseline_fixed is None else F.interpolate(
            intensity_baseline_fixed, size=(Hlr, Wlr), mode=ds_mode, align_corners=False
        )

        # downsample states to match LR
        prev_p_lr = _downsample_recurrent_states(prev_states_photon_fixed, ds, is_lstm, mode=ds_mode)
        prev_t_lr = _downsample_recurrent_states(prev_states_tof_fixed,    ds, is_lstm, mode=ds_mode)
    else:
        Hlr, Wlr = H, W
        D_lr, Dobs_lr, U_lr, ph_lr, P_lr = D_in, D_obs_fixed, Umap_fixed, photon_in_fixed, P_prior_pix
        I_lr = intensity_baseline_fixed
        prev_p_lr = prev_states_photon_fixed
        prev_t_lr = prev_states_tof_fixed

    # treat states as constants
    prev_p_lr = _detach_recurrent_states(prev_p_lr, is_lstm)
    prev_t_lr = _detach_recurrent_states(prev_t_lr, is_lstm)

    # ---- sample D perturbations in LR ----
    sqrt_var = torch.sqrt(P_lr.clamp_min(eps))
    noise = torch.randn((K, B, 1, Hlr, Wlr), device=D_in.device, dtype=D_in.dtype) * sqrt_var.unsqueeze(0)
    D_pert = (D_lr.unsqueeze(0) + noise).reshape(K * B, 1, Hlr, Wlr)

    # ---- expand fixed inputs to (K*B) ----
    photon_in = ph_lr.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, *ph_lr.shape[1:])
    D_obs    = Dobs_lr.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, 1, Hlr, Wlr)
    Umap     = U_lr.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, 1, Hlr, Wlr)

    tof_in = torch.cat([D_pert, D_obs, Umap], dim=1)

    if baseline_mode == "same_as_D":
        depth_baseline = D_pert
    else:
        depth_baseline = D_lr.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, 1, Hlr, Wlr)

    intensity_baseline = None if I_lr is None else I_lr.unsqueeze(0).expand(K, -1, -1, -1, -1).reshape(K * B, 1, Hlr, Wlr)

    # ---- expand states to (K*B) ----
    prev_states_photon = _expand_recurrent_states(prev_p_lr, K, B, is_lstm)
    prev_states_tof    = _expand_recurrent_states(prev_t_lr, K, B, is_lstm)

    # ---- external step_counter protection (tm internal counter remains) ----
    saved_ctr, ctr_ref = _save_step_counter(tm)
    try:
        with torch.no_grad():
            # NOTE: tm now outputs prob_scales instead of ref_scales.
            _, depth_scales, _, _ = tm(
                photon_in, tof_in,
                prev_states_photon=prev_states_photon,
                prev_states_tof=prev_states_tof,
                depth_baseline=depth_baseline,
                intensity_baseline=intensity_baseline,
            )
            y = depth_scales[-1]  # (K*B,1,Hlr,Wlr)
    finally:
        _restore_step_counter(saved_ctr, ctr_ref)

    y = y.reshape(K, B, 1, Hlr, Wlr)
    P_lr_out = y.var(dim=0, unbiased=False)

    if (Hlr, Wlr) != (H, W):
        return F.interpolate(P_lr_out, size=(H, W), mode=ds_mode, align_corners=False)
    return P_lr_out


# ====================== DKFN (NO block ops; keep original IO/logic) ======================
class SPNADF_Net(nn.Module):
    def __init__(
        self,
        k=4,
        rank=2,
        align_layers=2,
        use_gn=True,
        in_ch: int = 1,
        base: int = 64,
        mode: Literal["4", "8"] = "4",
        tm: Optional[nn.Module] = None,
        args=None,
    ):
        super().__init__()
        self.mode = mode
        self.tm = tm if tm is not None else IntensityDepthRecurrent(
            photon_input_channels=4,
            tof_input_channels=3,
            Dmax=66.6,
            skip_type="concat",
            recurrent_block_type="convlstm",
            num_encoders=4,
            base_num_channels=32,
            num_residual_blocks=4,
            norm="group",
            use_upsample_conv=True,
            max_depth_residual=10,
            use_depth_residual=True,
            use_deep_supervision=True,
            use_ms_depth_residual=True,
            downsample_mode="avgpool",
        )

    def forward(self, x, depth_stamp_seq=None, bin_seq=None, B_divP=None, args=None, reset_state=True):

        B, T, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        Dmax = 0.5 * 3e8 * (1 / args.rep_rate)
        sigma_d = 0.5 * 3e8 * args.pulse_width

        # pixel KF states
        D_prior = torch.full((B, 1, H, W), 0.5 * Dmax, device=device, dtype=dtype)
        P       = torch.full((B, 1, H, W), (Dmax / 12.0) ** 2, device=device, dtype=dtype)  # variance
        U_meas  = torch.full((B, 1, H, W), (sigma_d ** 2), device=device, dtype=dtype)

        # keep same "w_piror_tm" semantics: initial 0.5
        w_prior_tm = torch.full((B, 1, H, W), 0.5, device=device, dtype=dtype)
        I_prior = None

        prev_states_p = None
        prev_states_t = None

        y_final_seq = []
        uncertain_seq = []
        wobs_seq = []
        Create_new_seq = []

        prob_multi_scales = None
        depth_multi_scales = None
        ref_multi_scales = None

        for t in range(T):
            D_obs = x[:, t]                 # (B,1,H,W)
            z     = depth_stamp_seq[:, t]   # (B,1,H,W)
            bmask = bin_seq[:, t]           # (B,1,H,W)

            # ======== (same as old t>0 pre-update) produce wobs + Umap ========
            if t > 0:
                _, P_tmp, w_obs_pix, _ = kalman_update_pix_mixture(
                    xm=D_prior,
                    z=z,
                    Pm=P,
                    U=U_meas,
                    bin=bmask,
                    w=w_prior_tm,
                    Dmax=Dmax,
                    eps=1e-6,
                    tau=1.0,
                    var_floor=1e-6,
                )
                Umap = torch.sqrt(P_tmp).detach() / (Dmax / 12.0 + 1e-3)
            else:
                w_obs_pix = bmask
                Umap = torch.sqrt(P).detach() / (Dmax / 12.0 + 1e-3)

            wobs_seq.append(w_obs_pix.detach())

            # ======== TM (input semantics unchanged) ========
            I_p = I_prior if t > 0 else torch.full((B, 1, H, W), 0.5, device=device, dtype=dtype)

            photon_in = torch.cat([w_obs_pix, I_p, bmask, Umap], dim=1)     # (B,4,H,W)
            tof_in    = torch.cat([D_prior, D_obs, Umap], dim=1)            # (B,3,H,W)

            # IMPORTANT: save "input-side" states for propagation
            prev_states_p_in = prev_states_p
            prev_states_t_in = prev_states_t

            # tm now outputs prob_scales (NOT ref_scales)
            prob_scales, depth_scales, prev_states_p, prev_states_t = self.tm(
                photon_in, tof_in,
                prev_states_photon=prev_states_p_in,
                prev_states_tof=prev_states_t_in,
                depth_baseline=D_prior,
                intensity_baseline=I_prior,
            )

            S = len(prob_scales)
            if prob_multi_scales is None:
                prob_multi_scales = [[] for _ in range(S)]
                depth_multi_scales = [[] for _ in range(S)]
                ref_multi_scales = [[] for _ in range(S)]

            eps0 = 1e-6
            B_t = B_divP[:, t, :, 0, 0].unsqueeze(-1).unsqueeze(-1)  # broadcastable

            for s in range(S):
                p_s = prob_scales[s].clamp(1e-3, 1 - 1e-3)  # probability from tm
                d_s = depth_scales[s]

                # inverse: r = p*B/(1-p)
                r_s = (p_s * B_t) / (1.0 - p_s + eps0)
                r_s = r_s + eps0

                if p_s.shape[-2:] != (H, W):
                    p_s = F.interpolate(p_s, size=(H, W), mode="bilinear", align_corners=False)
                    d_s = F.interpolate(d_s, size=(H, W), mode="bilinear", align_corners=False)
                    r_s = F.interpolate(r_s, size=(H, W), mode="bilinear", align_corners=False)

                prob_multi_scales[s].append(p_s)
                depth_multi_scales[s].append(d_s)
                ref_multi_scales[s].append(r_s)

            # TM priors for KF update
            w_prior_tm = prob_multi_scales[-1][-1]        # (B,1,H,W)

            D_tm = depth_scales[-1]
            if D_tm.shape[-2:] != (H, W):
                D_tm = F.interpolate(D_tm, size=(H, W), mode="bilinear", align_corners=False)

            # ======== process noise (pixel variance) ========
            Q = torch.full_like(P, (Dmax / 60.0) ** 2)

            # ======== neural variance propagation (pixel): MC with hidden states + LR downsample ========
            P_prop = propagate_uncertainty_tm_mc_batch_states_lr(
                tm=self.tm,
                D_in=D_tm,
                photon_in_fixed=photon_in,
                D_obs_fixed=D_obs,
                Umap_fixed=Umap,
                intensity_baseline_fixed=I_prior,
                P_prior_pix=P,  # previous KF variance
                prev_states_photon_fixed=prev_states_p_in,
                prev_states_tof_fixed=prev_states_t_in,
                k_probe=4,
                baseline_mode="fixed",
                downsample_scale=2,
                ds_mode="bilinear",
            )
            Pm = P_prop + Q

            # ======== KF update (pixel mixture) ========
            D_post, P_post, rho_pix, create_new = kalman_update_pix_mixture(
                xm=D_tm,
                z=z,
                Pm=Pm,
                U=U_meas,
                bin=bmask,
                w=w_prior_tm,
                Dmax=Dmax,
                eps=1e-6,
                tau=1.0,
                var_floor=1e-6,
            )

            y_final_seq.append(D_post)
            uncertain_seq.append(torch.sqrt(P_post).detach())
            Create_new_seq.append(create_new.detach())

            # roll
            D_prior = D_post
            P = P_post.detach()

            # intensity prior remains "ref-like": use reconstructed r (last scale, last time)
            I_prior = ref_multi_scales[-1][-1]

        # stack multi-scale (same output format)
        prob_multi_bt, depth_multi_bt, ref_multi_bt = [], [], []
        for s in range(len(prob_multi_scales)):
            prob_multi_bt.append(torch.stack(prob_multi_scales[s], dim=1))    # (B,T,1,H,W)
            depth_multi_bt.append(torch.stack(depth_multi_scales[s], dim=1))
            ref_multi_bt.append(torch.stack(ref_multi_scales[s], dim=1))

        y_final = torch.stack(y_final_seq, dim=1)
        return {
            "y_final": y_final,
            "x_lin":   y_final,
            "wobs":    torch.stack(wobs_seq, dim=1),
            "prob_multi": prob_multi_bt,
            "depth_multi": depth_multi_bt,
            "ref_muti": ref_multi_bt,
            "uncertain": torch.stack(uncertain_seq, dim=1),
            "Create_new": torch.stack(Create_new_seq, dim=1),
        }
