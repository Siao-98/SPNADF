SP-NADF — Single-Photon Neural Assumed-Density Filter
==========================================================

This repository provides the code for SP-NADF (Single-Photon Neural Assumed-Density Filter): an online method for
joint depth + reflectivity reconstruction from single-photon LiDAR time-of-arrival (ToA) events, with pixel-wise
uncertainty for risk-aware recursion.

Repo layout note: README and code/ are at the same level. Result visualizations are under Video/.


1) Method summary
-----------------
SP-NADF targets dynamic-scene reconstruction under sparse photons, strong background noise, and fast motion. It:
- operates directly on ToA events (instead of long-exposure histograms),
- integrates photon-statistics physics with spatio-temporal neural priors,
- outputs actionable uncertainty estimates to assess reconstruction risk and guide robust inference.


2) Visual results (videos)
--------------------------

Fig. 7 — Online temporal evolution (depth + intensity)
This mosaic corresponds to Fig. 7: Temporal evolution of observations, reconstructions, and uncertainty in online recursion
(depth + reflectivity/intensity).
- Video/mosaic_2x3.webp

Fig. 7 — Uncertainty + ESP effect
This mosaic corresponds to Fig. 7’s uncertainty and the role of ESP (event-level signal probability).
- Video/mosaic_2x2.webp

Comparison with SPLiDER (four scenes)
The following mosaics are four scenes comparing SP-NADF with SPLiDER:
- Video/mosaic_2x4_part1.webp
- Video/mosaic_2x4_part2.webp
- Video/mosaic_2x4_part3.webp
- Video/mosaic_2x4_part4.webp


3) Code structure
-----------------
.
├── README.txt
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


4) Environment
--------------
Dependencies (minimal):
- torch, torchvision
- timm
- tensorboardX
- numpy, scipy
- opencv-python
- Pillow
- scikit-image
- tqdm, matplotlib

Install (example):
  pip install -r requirements.txt

Notes:
- torch / torchvision must be version-matched (common failure mode: torchvision operators missing).
- If you use CUDA, install PyTorch following the official instructions for your CUDA version.


5) Data
-------
This code expects ToA/event inputs and corresponding supervision (depth / reflectivity) in the format defined by:
- code/dataloader.py

Common arguments are defined in:
- code/input_args.py

At minimum you will typically need:
- --gtr_data_dir : ground-truth reflectivity (or supervision target) root
- --gtd_data_dir : ground-truth depth (or supervision target) root
- plus any ToA / detection-mask directories your loader requires

Tip:
If you are publishing the repo, add a short “Dataset layout” section here with a tree example (paths + file extensions).


6) Training
-----------
Run training from the repository root:

  python code/train_SPNADF.py \
    --gtr_data_dir /path/to/reflectivity_gt \
    --gtd_data_dir /path/to/depth_gt

Other useful flags (see code/input_args.py):
- learning rate / batch size / epochs
- logging directory
- checkpoint saving
- data augmentation switches


7) Outputs
----------
Training typically writes:
- checkpoints (model weights)
- TensorBoard logs (if enabled)
- qualitative reconstructions / saved results (if enabled)

Search in code/train_SPNADF.py for the output directory settings.


8) Citation
-----------
If you use this code, please cite the SP-NADF paper:
- Single-Photon Neural Assumed-Density Filter for Dynamic Scene Reconstruction

(You can add BibTeX here once you decide the final venue / bib entry.)


9) License
----------
Specify your license here (e.g., MIT / Apache-2.0 / research-only).

