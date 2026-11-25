"""Signal-processing building blocks used by the diagnoser."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def hampel(x: np.ndarray, half_window: int = 5, n_sigmas: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Hampel filter: flag samples far from the rolling median, measured in robust sigmas (MAD).

    Returns (cleaned, outlier_mask). Outliers are replaced with the rolling median.
    A sustained step is not flagged once it fills more than half the window.
    """
    x = np.asarray(x, dtype=float)
    size = 2 * half_window + 1
    med = median_filter(x, size=size, mode="nearest")
    mad = median_filter(np.abs(x - med), size=size, mode="nearest")
    sigma = 1.4826 * mad
    floor = 1.4826 * np.median(np.abs(x - np.median(x))) * 0.1 + 1e-12  # avoid sigma == 0 on flat data
    outliers = np.abs(x - med) > n_sigmas * np.maximum(sigma, floor)
    cleaned = np.where(outliers, med, x)
    return cleaned, outliers


def kalman_bias(z: np.ndarray, q: float, r: float, x0: float = 0.0, p0: float = 1.0) -> np.ndarray:
    """Scalar Kalman filter for a slowly varying bias observed in white noise.

    State model:       b[k] = b[k-1] + w,   w ~ N(0, q)
    Measurement model: z[k] = b[k] + v,     v ~ N(0, r)
    """
    z = np.asarray(z, dtype=float)
    est = np.empty_like(z)
    x, p = x0, p0
    for k, zk in enumerate(z):
        p = p + q                 # predict
        gain = p / (p + r)        # update
        x = x + gain * (zk - x)
        p = (1.0 - gain) * p
        est[k] = x
    return est


def cusum_up(z: np.ndarray, k: float, h: float) -> tuple[int | None, np.ndarray]:
    """One-sided (upward) CUSUM on a standardized series. Returns (first alarm index or None, statistic)."""
    s = np.empty(len(z))
    acc = 0.0
    first = None
    for i, zi in enumerate(z):
        acc = max(0.0, acc + zi - k)
        s[i] = acc
        if first is None and acc > h:
            first = i
    return first, s


def band_power(windows: np.ndarray, fs: float, f_center: float, half_bw: float) -> np.ndarray:
    """Power in [f_center - half_bw, f_center + half_bw] for each row, via a Hann-windowed FFT."""
    windows = np.atleast_2d(windows)
    m = windows.shape[1]
    w = np.hanning(m)
    spec = np.abs(np.fft.rfft((windows - windows.mean(axis=1, keepdims=True)) * w, axis=1)) ** 2
    spec /= (w ** 2).sum() * fs
    freqs = np.fft.rfftfreq(m, d=1.0 / fs)
    band = (freqs >= f_center - half_bw) & (freqs <= f_center + half_bw)
    return spec[:, band].sum(axis=1) * (freqs[1] - freqs[0])


def thermal_prediction(current: np.ndarray, t0: float, ambient: float, r_thermal: float,
                       tau: float, dt: float) -> np.ndarray:
    """Temperature the first-order thermal model predicts from a current trace, starting at `t0`."""
    pred = np.empty(len(current))
    pred[0] = t0
    alpha = dt / tau
    for k in range(1, len(current)):
        target = ambient + r_thermal * current[k - 1] ** 2
        pred[k] = pred[k - 1] + alpha * (target - pred[k - 1])
    return pred


def first_sustained(mask: np.ndarray, length: int) -> int | None:
    """Index at which `mask` has been True for `length` consecutive samples, or None."""
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= length:
            return i
    return None
