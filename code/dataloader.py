# dataloader.py

from torch.utils.data import Dataset, get_worker_info
import numpy as np
import random
import os
import glob
import random
from random import choices
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import scipy.io as scio

from utils import SPAD_data_generator, utils


def _get_noise_scales(args, name: str, default_list):
    v = getattr(args, name, None)
    if v is None:
        return list(default_list)
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    # allow a single float
    return [float(v)]





class train_dataloader(Dataset):

    def __init__(self, args, shared_seen=None, shared_lock=None):
        super().__init__()
        self.rvideo_paths = sorted(glob.glob(os.path.join(args.gtr_data_dir, '*.mp4')))
        self.dvideo_paths = sorted(glob.glob(os.path.join(args.gtd_data_dir, '*.mp4')))
        assert len(self.rvideo_paths) == len(self.dvideo_paths)

        self.args = args
        self.use_all_frames = getattr(args, 'train_use_all_frames', False)
        self.num_frames_cfg = getattr(args, 'train_num_frames', getattr(args, 'num_frames', 8))

        # ---- noise scale endpoints / discrete list ----
        self.train_noise_scales = _get_noise_scales(args, 'train_noise_scales', default_list=[1.0])

        # If curriculum disabled, we only support 'random' or 'fixed'
        mode = getattr(args, 'train_noise_mode', 'random')
        mode = str(mode).lower()
        if mode not in ('random', 'fixed'):
            mode = 'random'
        self.train_noise_mode = mode

        # ===========================
        # Curriculum sampling (Uniform expansion)
        # ===========================
        self.train_curriculum = bool(getattr(args, 'train_curriculum', True))

        # Use provided list ONLY for endpoints of continuous range
        self.scale_lo = float(min(self.train_noise_scales))
        self.scale_hi = float(max(self.train_noise_scales))

        # Total steps for progress (sample-level)
        self.total_epochs = int(getattr(args, 'total_epochs', 1000))
        self.iters_per_epoch = int(getattr(args, 'iters_per_epoch', 42))
        bs = int(getattr(args, 'batch_size', 1))
        self.total_steps = max(1, self.total_epochs * self.iters_per_epoch * bs)

        # Internal step counter (local fallback)
        self.shared_seen = shared_seen          # multiprocessing.Value('l', 0) or None
        self.shared_lock = shared_lock          # multiprocessing.Lock() or None
        self._seen_local = 0

        # Reach full range at p_reach_max (default 0.75)
        self.p_reach_max = float(getattr(args, 'train_p_reach_max', 0.75))
        if self.p_reach_max <= 1e-6:
            self.p_reach_max = 1.0

        # per-worker RNG (avoid identical streams across workers)
        self.base_seed = int(getattr(args, 'seed', 112))
        self._rng = None

    def __len__(self):
        return len(self.rvideo_paths)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            wi = get_worker_info()
            wid = 0 if wi is None else wi.id
            # large stride to reduce correlation
            self._rng = np.random.default_rng(self.base_seed + 100003 * wid)
        return self._rng

    def _get_global_seen(self) -> int:
        if self.shared_seen is None:
            return int(self._seen_local)
        if self.shared_lock is None:
            return int(self.shared_seen.value)
        with self.shared_lock:
            return int(self.shared_seen.value)

    def _inc_global_seen(self) -> int:
        if self.shared_seen is None:
            self._seen_local += 1
            return int(self._seen_local)
        if self.shared_lock is None:
            self.shared_seen.value += 1
            return int(self.shared_seen.value)
        with self.shared_lock:
            self.shared_seen.value += 1
            return int(self.shared_seen.value)

    def _progress(self) -> float:
        return min(1.0, max(0.0, self._get_global_seen() / float(self.total_steps)))

    def _sample_scale_curriculum(self, rng: np.random.Generator) -> float:

        p = self._progress()
        pr = p / self.p_reach_max
        if pr < 0.0:
            pr = 0.0
        elif pr > 1.0:
            pr = 1.0
        upper = self.scale_lo + pr * (self.scale_hi - self.scale_lo)
        if upper < self.scale_lo:
            upper = self.scale_lo
        return float(rng.uniform(self.scale_lo, upper))

    def _sample_scale_discrete(self, rng: np.random.Generator) -> float:
        """
        Discrete sampling from self.train_noise_scales.
        - mode='random': uniform choice from list
        - mode='fixed' : always pick the first element
        """
        if self.train_noise_mode == 'fixed':
            return float(self.train_noise_scales[0])
        return float(rng.choice(self.train_noise_scales))

    def __getitem__(self, idx):
        rvid_path = self.rvideo_paths[idx]
        dvid_path = self.dvideo_paths[idx]

        vid = cv2.VideoCapture(rvid_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()

        rng = self._get_rng()

        # --- frame policy ---
        if self.use_all_frames:
            num_frames = total_frames
            start_frame = 0
        else:
            num_frames = min(int(self.num_frames_cfg), total_frames)
            max_start = max(0, total_frames - num_frames)
            start_frame = int(rng.integers(0, max_start + 1))

        # ---- pick noise scale ----
        self._inc_global_seen()

        if self.train_curriculum:
            scale = float(self._sample_scale_curriculum(rng))  # continuous: Uniform(lo, upper(p))
        else:
            scale = float(self._sample_scale_discrete(rng))    # discrete: from list

        downsample_rate = int(rng.integers(1, 5))  # [1,4]

        # ---- read frames ----
        gtr_seq = utils.frames_extraction(
            rvid_path, num_frames, random_val=1, start_frame=start_frame,
            dwratio=downsample_rate, downsample=self.args.downsample
        )
        gtd_seq = utils.frames_extraction(
            dvid_path, num_frames, random_val=1, start_frame=start_frame,
            dwratio=downsample_rate, downsample=self.args.downsample,
        )
        assert gtr_seq.shape == gtd_seq.shape

        gtr_seq, gtd_seq = self.transforms(gtr_seq, gtd_seq)
        assert gtr_seq.shape == gtd_seq.shape

        # ----- build time_travel from gtd_seq -----
        time_travel = np.zeros_like(gtd_seq, dtype=np.float64)
        for i in range(time_travel.shape[0]):
            d = -gtd_seq[i, :, :].astype(np.float64) + 255.0
            time_travel[i, :, :] = SPAD_data_generator.timetravel(
                d,
                scale_min=self.args.scale_min, scale_max=self.args.scale_max,
                expand=self.args.expand, c_speed=self.args.c_speed
            )

        # prealloc
        bin_seq = np.zeros_like(gtd_seq, dtype=np.float64)
        timestamp_seq = np.zeros_like(gtd_seq, dtype=np.float64)
        sigprob_seq = np.zeros_like(gtd_seq, dtype=np.float64)
        B_divP = np.ones_like(gtd_seq, dtype=np.float64)

        # reflectivity normalize
        gtr_seq = utils.normalize(gtr_seq, max_value=255.0)

        # apply solar background scale (only this parameter changes SNR)
        base_bkg = float(self.args.solar_background_per_meter)
        solar_bkg_override = base_bkg * scale

        for i in range(bin_seq.shape[0]):
            b_i, ts_i, p_i, b_divp = SPAD_data_generator.timestamp(
                time_travel[i, :, :],
                gtr_seq[i, :, :],
                self.args,
                solar_background_per_meter_override=solar_bkg_override
            )
            bin_seq[i, :, :] = b_i
            timestamp_seq[i, :, :] = ts_i
            sigprob_seq[i, :, :] = p_i
            B_divP[i, :, :] = b_divp

        # derived
        bin_seq = utils.normalize(bin_seq, max_value=1.0)
        depth_stamp_seq = 0.5 * 3e8 * timestamp_seq
        depth_stamp_seq_norm = utils.normalize(
            depth_stamp_seq, max_value=0.5 * 3e8 * (1 / self.args.rep_rate)
        )
        gtd_seq_m = 0.5 * 3e8 * time_travel

        # add channel dim -> (T,1,H,W)
        bin_seq = bin_seq[:, None, :, :]
        depth_stamp_seq = depth_stamp_seq[:, None, :, :]
        depth_stamp_seq_norm = depth_stamp_seq_norm[:, None, :, :]
        gtr_seq = gtr_seq[:, None, :, :]
        gtd_seq_m = gtd_seq_m[:, None, :, :]
        sigprob_seq = sigprob_seq[:, None, :, :]
        B_divP = B_divP[:, None, :, :]  # IMPORTANT: keep (T,1,H,W)

        # to tensor
        return (
            torch.from_numpy(bin_seq).float(),
            torch.from_numpy(depth_stamp_seq).float(),
            torch.from_numpy(depth_stamp_seq_norm).float(),
            torch.from_numpy(gtr_seq).float(),
            torch.from_numpy(gtd_seq_m).float(),
            torch.from_numpy(sigprob_seq).float(),
            torch.from_numpy(B_divP).float(),
            torch.tensor(scale, dtype=torch.float32)  # optional: for logging
        )

    def transforms(self, gtr_seq, gtd_seq):
        if not self.args.transforms:
            return gtr_seq, gtd_seq

        patch_size = self.args.patch_size
        assert isinstance(patch_size, (list, tuple)) and len(patch_size) == 2
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])

        H, W = gtr_seq.shape[1], gtr_seq.shape[2]
        assert patch_h <= H and patch_w <= W

        left = random.randint(0, W - patch_w)
        right = left + patch_w
        top = random.randint(0, H - patch_h)
        bottom = top + patch_h

        gtr_seq = gtr_seq[:, top:bottom, left:right]
        gtd_seq = gtd_seq[:, top:bottom, left:right]

        do_nothing = lambda x: x
        flipud = lambda x: x[::-1, :]
        rot180 = lambda x: np.rot90(x, k=2, axes=(0, 1))
        rot180_flipud = lambda x: (np.rot90(x, k=2, axes=(0, 1)))[::-1, :]

        rot90 = lambda x: np.rot90(x, axes=(0, 1))
        rot90_flipud = lambda x: (np.rot90(x, axes=(0, 1)))[::-1, :]
        rot270 = lambda x: np.rot90(x, k=3, axes=(0, 1))
        rot270_flipud = lambda x: (np.rot90(x, axes=(0, 1)))[::-1, :]

        N = gtr_seq.shape[0]

        if patch_h == patch_w:
            aug_list = [do_nothing, flipud, rot90, rot90_flipud, rot180, rot180_flipud, rot270, rot270_flipud]
            w_aug = [7, 4, 4, 4, 4, 4, 4, 4]
        else:
            aug_list = [do_nothing, flipud, rot180, rot180_flipud]
            w_aug = [7, 4, 4, 4]

        op = choices(aug_list, w_aug)[0]
        for j in range(N):
            gtr_seq[j, ...] = op(gtr_seq[j, ...])
            gtd_seq[j, ...] = op(gtd_seq[j, ...])

        return gtr_seq, gtd_seq




class val_dataloader(Dataset):
    """
    Val: allow either:
      - fixed noise scale (args.val_fixed_noise_scale != None)
      - random per sample (val_noise_mode='random')
    Training script will usually create multiple val_dataloader, one per scale, to sweep.
    """
    def __init__(self, args):
        super().__init__()
        self.rvideo_paths = sorted(glob.glob(os.path.join(args.val_gtr_data_dir, '*.mp4')))
        self.dvideo_paths = sorted(glob.glob(os.path.join(args.val_gtd_data_dir, '*.mp4')))
        assert len(self.rvideo_paths) == len(self.dvideo_paths)

        self.args = args
        self.use_all_frames = getattr(args, 'val_use_all_frames', False)
        self.num_frames_cfg = getattr(args, 'val_num_frames', getattr(args, 'num_frames', 8))

        self.val_noise_scales = _get_noise_scales(args, 'val_noise_scales', default_list=[1.0])
        self.val_noise_mode = getattr(args, 'val_noise_mode', 'fixed')  # fixed|random
        self.val_fixed_noise_scale = getattr(args, 'val_fixed_noise_scale', None)

        seed = int(getattr(args, 'seed', 112)) + 999
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rvideo_paths)

    def __getitem__(self, idx):
        rvid_path = self.rvideo_paths[idx]
        dvid_path = self.dvideo_paths[idx]

        vid = cv2.VideoCapture(rvid_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()

        if self.use_all_frames:
            num_frames = total_frames
            start_frame = 0
        else:
            num_frames = min(int(self.num_frames_cfg), total_frames)
            max_start = max(0, total_frames - num_frames)
            start_frame = random.randint(0, max_start)

        # choose scale (DISCRETE for val)
        if self.val_fixed_noise_scale is not None:
            scale = float(self.val_fixed_noise_scale)


        gtr_seq = utils.frames_extraction(
            rvid_path, num_frames, random_val=1, start_frame=start_frame,
            dwratio=self.args.dwratio, downsample=self.args.downsample
        )
        gtd_seq_rgb = utils.frames_extraction(
            dvid_path, num_frames, random_val=1, start_frame=start_frame,
            dwratio=self.args.dwratio, downsample=self.args.downsample
        )
        assert gtr_seq.shape == gtd_seq_rgb.shape

        time_travel = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        for i in range(time_travel.shape[0]):
            d = -gtd_seq_rgb[i, :, :].astype(np.float64) + 255.0
            time_travel[i, :, :] = SPAD_data_generator.timetravel(
                d,
                scale_min=self.args.scale_min, scale_max=self.args.scale_max,
                expand=self.args.expand, c_speed=self.args.c_speed
            )

        bin_seq = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        timestamp_seq = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        sigprob_seq = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        B_divP = np.ones_like(gtd_seq_rgb, dtype=np.float64)

        gtr_seq = utils.normalize(gtr_seq, max_value=255.0)

        base_bkg = float(self.args.solar_background_per_meter)
        solar_bkg_override = base_bkg * scale

        for i in range(bin_seq.shape[0]):
            bin_i, ts_i, p_i, b_divp = SPAD_data_generator.timestamp(
                time_travel[i, :, :],
                gtr_seq[i, :, :],
                self.args,
                solar_background_per_meter_override=solar_bkg_override
            )
            bin_seq[i, :, :] = bin_i
            timestamp_seq[i, :, :] = ts_i
            sigprob_seq[i, :, :] = p_i
            B_divP[i, :, :] = b_divp

        bin_seq = utils.normalize(bin_seq, max_value=1.0)
        depth_stamp_seq = 0.5 * 3e8 * timestamp_seq
        depth_stamp_seq_norm = utils.normalize(depth_stamp_seq, max_value=0.5 * 3e8 * (1 / self.args.rep_rate))
        gtd_seq_m = 0.5 * 3e8 * time_travel

        bin_seq = torch.from_numpy(bin_seq[:, None, :, :]).float()
        depth_stamp_seq = torch.from_numpy(depth_stamp_seq[:, None, :, :]).float()
        depth_stamp_seq_norm = torch.from_numpy(depth_stamp_seq_norm[:, None, :, :]).float()
        gtr_seq = torch.from_numpy(gtr_seq[:, None, :, :]).float()
        gtd_seq_m = torch.from_numpy(gtd_seq_m[:, None, :, :]).float()
        sigprob_seq = torch.from_numpy(sigprob_seq[:, None, :, :]).float()
        B_divP = torch.from_numpy(B_divP[:, None, :, :]).float()

        return bin_seq, depth_stamp_seq, depth_stamp_seq_norm, gtr_seq, gtd_seq_m, sigprob_seq, B_divP, torch.tensor(scale, dtype=torch.float32)


class test_dataloader(Dataset):
    def __init__(self, args, rvideo_path, dvideo_path):
        super().__init__()
        self.rvideo_path = rvideo_path
        self.dvideo_path = dvideo_path
        self.args = args

        # fixed scale for test if provided
        self.test_noise_scale = float(getattr(args, 'test_noise_scale', 1.0))

    def __len__(self):
        vid = cv2.VideoCapture(self.rvideo_path)
        total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()
        return total - self.args.num_frames + 1

    def __getitem__(self, idx):
        vid = cv2.VideoCapture(self.rvideo_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()
        assert idx <= total_frames - self.args.num_frames

        gtr_seq = utils.frames_extraction(
            self.rvideo_path, self.args.num_frames, random_val=1, start_frame=idx,
            dwratio=self.args.dwratio, downsample=self.args.downsample
        )
        gtd_seq_rgb = utils.frames_extraction(
            self.dvideo_path, self.args.num_frames, random_val=1, start_frame=idx,
            dwratio=self.args.dwratio, downsample=self.args.downsample
        )
        assert gtr_seq.shape == gtd_seq_rgb.shape

        time_travel = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        for i in range(time_travel.shape[0]):
            d = -gtd_seq_rgb[i, :, :].astype(np.float64) + 255.0
            time_travel[i, :, :] = SPAD_data_generator.timetravel(
                d,
                scale_min=self.args.scale_min, scale_max=self.args.scale_max,
                expand=self.args.expand, c_speed=self.args.c_speed
            )

        bin_seq = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        timestamp_seq = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        sigprob_seq = np.zeros_like(gtd_seq_rgb, dtype=np.float64)
        B_divP = np.ones_like(gtd_seq_rgb, dtype=np.float64)

        gtr_seq = utils.normalize(gtr_seq, max_value=255.0)

        base_bkg = float(self.args.solar_background_per_meter)
        solar_bkg_override = base_bkg * self.test_noise_scale

        for i in range(bin_seq.shape[0]):
            bin_i, ts_i, p_i, b_divp = SPAD_data_generator.timestamp(
                time_travel[i, :, :],
                gtr_seq[i, :, :],
                self.args,
                solar_background_per_meter_override=solar_bkg_override
            )
            bin_seq[i, :, :] = bin_i
            timestamp_seq[i, :, :] = ts_i
            sigprob_seq[i, :, :] = p_i
            B_divP[i, :, :] = b_divp

        bin_seq = utils.normalize(bin_seq, max_value=1.0)
        depth_stamp_seq = 0.5 * 3e8 * timestamp_seq
        depth_stamp_seq_norm = utils.normalize(depth_stamp_seq, max_value=0.5 * 3e8 * (1 / self.args.rep_rate))
        gtd_seq_m = 0.5 * 3e8 * time_travel

        return (
            torch.from_numpy(bin_seq[:, None, :, :]).float(),
            torch.from_numpy(depth_stamp_seq[:, None, :, :]).float(),
            torch.from_numpy(depth_stamp_seq_norm[:, None, :, :]).float(),
            torch.from_numpy(gtr_seq[:, None, :, :]).float(),
            torch.from_numpy(gtd_seq_m[:, None, :, :]).float(),
            torch.from_numpy(sigprob_seq[:, None, :, :]).float(),
            torch.from_numpy(B_divP[:, None, :, :]).float(),
            torch.tensor(self.test_noise_scale, dtype=torch.float32)
        )

class fullvideo_dataloader(Dataset):
    """
    Read a full video (return all frames once). Keep outputs consistent with train/val:
      bin_seq, depth_stamp_seq, depth_stamp_seq_norm, gtr_seq, gtd_seq, sigprob_seq, B_divP, scale
    all shaped (T,1,H,W)
    """
    def __init__(self, args, rvideo_path: str, dvideo_path: str, noise_scale: float = 1.0):
        super().__init__()
        self.args = args
        self.rvideo_path = rvideo_path
        self.dvideo_path = dvideo_path
        self.noise_scale = float(noise_scale)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        vid = cv2.VideoCapture(self.rvideo_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()

        num_frames = total_frames
        start_frame = 0

        gtr_seq = utils.frames_extraction(
            self.rvideo_path, num_frames, random_val=1, start_frame=start_frame,
            dwratio=self.args.dwratio, downsample=self.args.downsample
        )
        gtd_seq = utils.frames_extraction(
            self.dvideo_path, num_frames, random_val=1, start_frame=start_frame,
            dwratio=self.args.dwratio, downsample=self.args.downsample
        )
        assert gtr_seq.shape == gtd_seq.shape

        time_travel = np.zeros_like(gtd_seq, dtype=np.float64)
        for i in range(time_travel.shape[0]):
            d = -gtd_seq[i, :, :].astype(np.float64) + 255.0
            time_travel[i, :, :] = SPAD_data_generator.timetravel(
                d,
                scale_min=self.args.scale_min, scale_max=self.args.scale_max,
                expand=self.args.expand, c_speed=self.args.c_speed
            )

        bin_seq = np.zeros_like(gtd_seq, dtype=np.float64)
        timestamp_seq = np.zeros_like(gtd_seq, dtype=np.float64)
        sigprob_seq = np.zeros_like(gtd_seq, dtype=np.float64)
        B_divP = np.ones_like(gtd_seq, dtype=np.float64)

        gtr_seq = utils.normalize(gtr_seq, max_value=255.0)

        base_bkg = float(self.args.solar_background_per_meter)
        solar_bkg_override = base_bkg * self.noise_scale

        for i in range(bin_seq.shape[0]):
            b_i, ts_i, p_i, b_divp = SPAD_data_generator.timestamp(
                time_travel[i, :, :],
                gtr_seq[i, :, :],
                self.args,
                solar_background_per_meter_override=solar_bkg_override
            )
            bin_seq[i, :, :] = b_i
            timestamp_seq[i, :, :] = ts_i
            sigprob_seq[i, :, :] = p_i
            B_divP[i, :, :] = b_divp

        bin_seq = utils.normalize(bin_seq, max_value=1.0)
        depth_stamp_seq = 0.5 * 3e8 * timestamp_seq
        depth_stamp_seq_norm = utils.normalize(depth_stamp_seq, max_value=0.5 * 3e8 * (1 / self.args.rep_rate))
        gtd_seq_m = 0.5 * 3e8 * time_travel

        return (
            torch.from_numpy(bin_seq[:, None, :, :]).float(),
            torch.from_numpy(depth_stamp_seq[:, None, :, :]).float(),
            torch.from_numpy(depth_stamp_seq_norm[:, None, :, :]).float(),
            torch.from_numpy(gtr_seq[:, None, :, :]).float(),
            torch.from_numpy(gtd_seq_m[:, None, :, :]).float(),
            torch.from_numpy(sigprob_seq[:, None, :, :]).float(),
            torch.from_numpy(B_divP[:, None, :, :]).float(),
            torch.tensor(self.noise_scale, dtype=torch.float32),
        )
