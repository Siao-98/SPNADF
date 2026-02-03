# SP-NADF — Single-Photon Neural Assumed-Density Filter

This repository provides the code for **SP-NADF (Single-Photon Neural Assumed-Density Filter)**: an online method for
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

> GitHub Markdown does not support “center align” for the `![]()` syntax.
> Use HTML `<p align="center">...</p>` to center images reliably.

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

