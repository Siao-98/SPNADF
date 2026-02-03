# SPAD_data_generator.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2

_EPS_DIV = 1e-12
_EPS_SCALE = 1e-6
_MED_KSIZE = 3


def _as_f64(x):
    return np.asarray(x, dtype=np.float64)


def _rng_or_default(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng()


def depth_scale(depth, scale_min, scale_max):
    d = _as_f64(depth)
    d_blur = cv2.medianBlur(d.astype(np.float32), _MED_KSIZE).astype(np.float64)
    dmin = float(np.min(d_blur))
    dmax = float(np.max(d_blur))

    if not np.isfinite(dmin) or not np.isfinite(dmax) or (abs(dmax - dmin) < _EPS_DIV):
        mid = 0.5 * (float(scale_min) + float(scale_max))
        return np.full_like(d_blur, mid, dtype=np.float64)

    scaled = float(scale_min) + (d_blur - dmin) / (dmax - dmin + _EPS_SCALE) * (float(scale_max) - float(scale_min))
    return scaled.astype(np.float64)


def timetravel(depth, scale_min, scale_max, expand=1.0, c_speed=3e8):
    depth_m = depth_scale(depth, scale_min, scale_max)
    tof = 2.0 * depth_m / float(c_speed)
    return _as_f64(tof * float(expand))


def signal_energy(reflectivity,
                  energy_per_pulse, spad_q_efficiency, target_range, illum_radius,
                  effictive_pix_size_x, effictive_pix_size_y, f_no,
                  solar_background_per_meter, C_atm, wavelength, plank, c_speed):

    illum_area = np.pi * (float(illum_radius) ** 2)
    f_no2 = float(f_no) ** 2
    pix_area = float(effictive_pix_size_x) * float(effictive_pix_size_y)

    hc_inv = float(wavelength) / (float(plank) * float(c_speed))  # lambda / (h c)

    refl = _as_f64(reflectivity)

    common_geom = (pix_area / max(illum_area, _EPS_DIV)) / (8.0 * max(f_no2, _EPS_DIV))

    signal_ppp = hc_inv * float(energy_per_pulse) * (float(spad_q_efficiency) * refl * (float(C_atm) ** (2.0 * float(target_range)))) * common_geom
    rate_of_bck_photons = hc_inv * float(solar_background_per_meter) * 1.0 * (float(spad_q_efficiency) * 1.0 * (float(C_atm) ** (float(target_range)))) * common_geom

    return _as_f64(signal_ppp), _as_f64(rate_of_bck_photons)


def make_clicks(reflectivity, args, rng=None, allow_zero_pulses=False,
                solar_background_per_meter_override=None,
                energy_per_pulse_override=None):

    rng = _rng_or_default(rng)

    exp_time = 1.0 / float(args.fps)
    num_pulses_per_frame = int(np.round(exp_time * float(args.rep_rate)))
    if not allow_zero_pulses:
        num_pulses_per_frame = max(1, num_pulses_per_frame)

    illum_radius = ((float(args.target_range) / float(args.illum_lens)) * float(args.fibre_core)) / 2.0

    solar_bkg = float(args.solar_background_per_meter) if solar_background_per_meter_override is None else float(solar_background_per_meter_override)
    e_pulse = float(args.energy_per_pulse) if energy_per_pulse_override is None else float(energy_per_pulse_override)

    signal_ppp, bck_rate = signal_energy(
        reflectivity, e_pulse, args.spad_q_base_efficiency, args.target_range,
        illum_radius, args.effictive_pix_size_x, args.effictive_pix_size_y, args.f_no,
        solar_bkg, args.C_atm, args.wavelength, args.plank, args.c_speed
    )

    dc_ppp = float(args.dark_count_rate) / float(args.rep_rate)
    bck_ppp = bck_rate / float(args.rep_rate)
    noise_ppp = _as_f64(bck_ppp + dc_ppp)

    total_ppf = (signal_ppp + noise_ppp) * float(num_pulses_per_frame)

    clicks = rng.poisson(total_ppf)
    binary_img = (clicks > 0).astype(np.int32)

    return binary_img, signal_ppp, noise_ppp


def timestamp(time_travel, reflectivity, args, rng=None, allow_zero_pulses=False,
              solar_background_per_meter_override=None,
              energy_per_pulse_override=None):
    """
    Returns:
      binary_image (H,W) int32
      stamps       (H,W) float64 seconds, zero where no click
      sig_prob     (H,W) float64
      B_divP       (H,W) float64  (kept consistent with your current usage)
    """
    rng = _rng_or_default(rng)

    tof = _as_f64(time_travel)
    H, W = tof.shape

    binary_image, signal_ppp, noise_ppp = make_clicks(
        reflectivity, args, rng=rng, allow_zero_pulses=allow_zero_pulses,
        solar_background_per_meter_override=solar_background_per_meter_override,
        energy_per_pulse_override=energy_per_pulse_override
    )

    sigma_t = np.sqrt(float(args.pulse_width) ** 2 + float(args.jitter) ** 2)
    pulse_stamps = rng.normal(loc=tof, scale=sigma_t, size=(H, W))

    t_min = float(args.min_depth_in_time)
    t_max = 1.0 / float(args.rep_rate)
    if t_max <= t_min:
        t_max = t_min + 1.0 / max(float(args.rep_rate), 1.0)
    bck_stamps = rng.uniform(low=t_min, high=t_max, size=(H, W))

    sig_prob = signal_ppp / np.maximum(signal_ppp + noise_ppp, _EPS_DIV)
    sig_prob = np.clip(sig_prob, 0.0, 1.0)

    # keep consistent with what you used in training code
    B_divP = noise_ppp * _as_f64(reflectivity) / np.maximum(signal_ppp, _EPS_DIV)

    bern = rng.binomial(n=1, p=sig_prob, size=(H, W))
    stamps = (bern * pulse_stamps + (1 - bern) * bck_stamps) * binary_image

    return binary_image, _as_f64(stamps), _as_f64(sig_prob), _as_f64(B_divP)
