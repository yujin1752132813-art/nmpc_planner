from __future__ import annotations

from typing import List

import numpy as np

from .types import RefPoint


def _gaussian_kernel1d(sigma_pts: float) -> np.ndarray:
    sigma_pts = max(float(sigma_pts), 1.0)
    radius = int(3.0 * sigma_pts)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_pts) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _smooth_array(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return out


def smooth_reference_curvature(
    path: List[RefPoint],
    sigma_m: float = 0.8,
    passes: int = 2,
) -> List[RefPoint]:
    """Smooth the curvature profile kappa(s), then integrate back yaw/x/y.

    This is more principled than directly smoothing x/y for your problem:
    it specifically targets the straight-to-arc curvature jump that creates
    small steering/lateral-error spikes at arc entry/exit.
    """
    if len(path) < 5:
        return path

    s = np.array([p.s for p in path], dtype=float)
    x = np.array([p.x for p in path], dtype=float)
    y = np.array([p.y for p in path], dtype=float)
    yaw = np.array([p.yaw for p in path], dtype=float)
    kappa = np.array([p.kappa for p in path], dtype=float)
    v_ref = np.array([p.v_ref for p in path], dtype=float)

    ds_nom = float(np.mean(np.diff(s)))
    sigma_pts = max(sigma_m / max(ds_nom, 1e-6), 1.0)
    kernel = _gaussian_kernel1d(sigma_pts)

    kappa_sm = kappa.copy()
    for _ in range(max(int(passes), 1)):
        kappa_sm = _smooth_array(kappa_sm, kernel)

    # preserve start/end curvature approximately
    kappa_sm[0] = kappa[0]
    kappa_sm[-1] = kappa[-1]

    # integrate yaw from smoothed curvature
    yaw_sm = np.zeros_like(yaw)
    yaw_sm[0] = yaw[0]
    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]
        yaw_sm[i] = yaw_sm[i - 1] + 0.5 * (kappa_sm[i - 1] + kappa_sm[i]) * ds

    # integrate x,y using smoothed yaw
    x_sm = np.zeros_like(x)
    y_sm = np.zeros_like(y)
    x_sm[0] = x[0]
    y_sm[0] = y[0]
    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]
        x_sm[i] = x_sm[i - 1] + 0.5 * (np.cos(yaw_sm[i - 1]) + np.cos(yaw_sm[i])) * ds
        y_sm[i] = y_sm[i - 1] + 0.5 * (np.sin(yaw_sm[i - 1]) + np.sin(yaw_sm[i])) * ds

    smoothed = []
    for i in range(len(s)):
        smoothed.append(
            RefPoint(
                s=float(s[i]),
                x=float(x_sm[i]),
                y=float(y_sm[i]),
                yaw=float(yaw_sm[i]),
                kappa=float(kappa_sm[i]),
                v_ref=float(v_ref[i]),
            )
        )
    return smoothed