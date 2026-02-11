# SP-NADF — Single-Photon Neural Assumed-Density Filter

This repository provides the code for **SP-NADF (Single-Photon Neural Assumed-Density Filter)**: an method for
**joint depth + reflectivity** reconstruction from **single-photon LiDAR time-of-arrival (ToA) events**, with **pixel-wise
uncertainty** for risk-aware recursion.

**Repo layout note:** `README.md` and `code/` are at the same level. Result visualizations are under `Video/`.

---

## Contents
- [Method Summary](#method-summary)
- [Visual Results](#visual-results)
- [Code Structure](#code-structure)
- [Environment](#environment)
- [Data](#data)
- [Training](#training)
- [Outputs](#outputs)
- [Citation](#citation)
- [License](#license)

---

## Method Summary

SP-NADF targets dynamic-scene reconstruction under sparse photons, strong background noise, and fast motion. It:
- operates directly on **ToA events** (instead of long-exposure histograms),
- integrates **photon-statistics physics** with **spatio-temporal neural priors**,
- outputs **actionable uncertainty estimates** to assess reconstruction risk and guide robust inference.

---

## Visual Results

> GitHub Markdown doesn’t provide a native way to center images rendered via `![]()`.  
> Use an HTML wrapper instead.

### Online temporal evolution (depth + intensity)
<p align="center">
  <img src="Video/mosaic_2x3.webp" alt="SP-NADF Online1" width="900">
</p>

### Uncertainty + ESP effect
<p align="center">
  <img src="Video/mosaic_2x2.webp" alt="SP-NADF Online2" width="900">
</p>

### Comparison with SPLiDER (four scenes)

<p align="center">
  <img src="Video/mosaic_2x4_part1.webp" alt="SPLiDER comparison — scene 1" width="900">
</p>

<p align="center">
  <img src="Video/mosaic_2x4_part2.webp" alt="SPLiDER comparison — scene 2" width="900">
</p>

<p align="center">
  <img src="Video/mosaic_2x4_part3.webp" alt="SPLiDER comparison — scene 3" width="900">
</p>

<p align="center">
  <img src="Video/mosaic_2x4_part4.webp" alt="SPLiDER comparison — scene 4" width="900">
</p>

**Notes**
- GitHub renders animated `.webp` as an image (often animated).
- If you want a true video player UI, consider exporting to `.mp4` and linking it (or uploading as a Release asset).

---

## Code Structure

```
.
├── README.md
├── Video/
│   ├── mosaic_2x3.webp
│   ├── mosaic_2x2.webp
│   ├── mosaic_2x4_part1.webp
│   ├── mosaic_2x4_part2.webp
│   ├── mosaic_2x4_part3.webp
│   └── mosaic_2x4_part4.webp
└── code/
    ├── train_SPNADF.py
    ├── input_args.py
    ├── dataloader.py
    ├── model/
    └── utils/
```

---

## Environment

### Dependencies (minimal)
- torch, torchvision
- timm
- tensorboardX
- numpy, scipy
- opencv-python
- Pillow
- scikit-image
- tqdm, matplotlib

### Install
```bash
pip install -r requirements.txt
```

**Notes**
- `torch` / `torchvision` must be version-matched (common failure mode: missing torchvision operators).
- If you use CUDA, install PyTorch following the official instructions for your CUDA version.

---

## Data

This code expects ToA/event inputs and corresponding supervision (depth / reflectivity) in the format defined by:
- `code/dataloader.py`

Common arguments are defined in:
- `code/input_args.py`

At minimum you will typically need:
- `--gtr_data_dir` : ground-truth reflectivity (or supervision target) root
- `--gtd_data_dir` : ground-truth depth (or supervision target) root
- plus any ToA / detection-mask directories required by your loader

**Tip:** If you plan to publish the repo, add a “Dataset layout” section here with an example directory tree.

---

## Training

Run training from the repository root:

```bash
python code/train_SPNADF.py \
  --gtr_data_dir /path/to/reflectivity_gt \
  --gtd_data_dir /path/to/depth_gt
```

Other useful flags (see `code/input_args.py`):
- learning rate / batch size / epochs
- logging directory
- checkpoint saving
- data augmentation switches

---

## Outputs

Training typically writes:
- checkpoints (model weights)
- TensorBoard logs (if enabled)
- qualitative reconstructions / saved results (if enabled)

Search in `code/train_SPNADF.py` for output directory settings.

---

## Citation

If you use this code, please cite the SP-NADF paper:
- *Single-Photon Neural Assumed-Density Filter for Dynamic Scene Reconstruction*

(BibTeX can be added here once you decide the final bib entry.)

---

## License

Specify your license here (e.g., MIT / Apache-2.0 / research-only).
