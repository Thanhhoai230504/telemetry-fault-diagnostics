"""Turn one run of telemetry into a diagnostic report with classified fault codes."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter

from .signals import band_power, cusum_up, first_sustained, hampel, kalman_bias, thermal_prediction
from .simulate import Run

FAULT_CODES = {
    "F01": ("TEMP_SENSOR_DRIFT", "drift", "temperature",
            "temperature departs from the thermal model while current is normal: the sensor, not the motor"),
    "F02": ("CURRENT_IMPULSE_NOISE", "spikes", "current",
            "isolated one-sample current impulses: EMI or loose wiring on the CT channel"),
    "F03": ("BEARING_WEAR", "bearing", "vibration",
            "energy at the outer-race defect frequency rises above its baseline"),
    "F04": ("OVERLOAD", "overload", "current",
            "sustained current above rating, with temperature following the thermal model"),
}
CODE_FOR_FAULT = {v[1]: k for k, v in FAULT_CODES.items()}


@dataclass(frozen=True)
class Thresholds:
    calibration_s: float = 300.0     # fault-free start of every run used to learn the baseline
    # temperature sensor drift (F01)
    kalman_q: float = 1e-4
    kalman_r: float = 0.03
    drift_limit_degC: float = 1.5
    drift_hold_s: int = 10
    # current impulses (F02)
    hampel_half_window: int = 5
    hampel_sigmas: float = 5.0
    impulse_min_A: float = 1.5
    impulse_count: int = 3
    impulse_span_s: float = 60.0
    # bearing wear (F03)
    bpfo_half_bw_hz: float = 6.0
    cusum_k: float = 1.0
    cusum_h: float = 25.0
    # overload (F04)
    overload_ratio: float = 1.15
    overload_hold_s: int = 10


def diagnose(run: Run, th: Thresholds | None = None) -> dict:
    th = th or Thresholds()
    spec = run.spec
    t = run.t
    cal = t < th.calibration_s
    alarms: list[dict] = []

    def raise_alarm(code: str, idx: int, **evidence) -> None:
        name, _, channel, why = FAULT_CODES[code]
        alarms.append({"code": code, "name": name, "channel": channel, "time_s": float(t[idx]),
                       "reason": why, "evidence": {k: _round(v) for k, v in evidence.items()}})

    # --- F02: isolated current impulses ------------------------------------------
    clean_i, outliers = hampel(run.current, th.hampel_half_window, th.hampel_sigmas)
    x = run.current
    prev = np.r_[x[0], x[:-1]]
    nxt = np.r_[x[1:], x[-1]]
    dev = x - clean_i
    # an impulse stands out from *both* neighbours in the same direction; a ramp or step does not
    impulse = (outliers & (np.abs(dev) >= th.impulse_min_A)
               & (np.sign(x - prev) == np.sign(x - nxt)) & (np.sign(x - prev) != 0))
    hits = np.flatnonzero(impulse)
    for j in range(th.impulse_count - 1, len(hits)):
        if t[hits[j]] - t[hits[j - th.impulse_count + 1]] <= th.impulse_span_s:
            raise_alarm("F02", hits[j], impulses_in_window=th.impulse_count,
                        largest_impulse_A=float(np.max(np.abs(dev[hits[: j + 1]]))))
            break

    # --- F04: overload ------------------------------------------------------------
    i_cal = float(np.median(clean_i[cal]))
    i_smooth = median_filter(clean_i, size=31, mode="nearest")
    idx = first_sustained(i_smooth > th.overload_ratio * i_cal, th.overload_hold_s)
    if idx is not None:
        raise_alarm("F04", idx, baseline_A=i_cal, current_A=float(i_smooth[idx]),
                    ratio=float(i_smooth[idx] / i_cal))

    # --- F01: temperature sensor drift (model residual + Kalman) --------------------
    t0 = float(np.median(run.temp[:5]))
    pred = thermal_prediction(clean_i, t0, spec.ambient, spec.r_thermal, spec.tau, spec.dt)
    resid = run.temp - pred
    resid = resid - np.mean(resid[cal])
    bias = kalman_bias(resid, q=th.kalman_q, r=th.kalman_r)
    idx = first_sustained((np.abs(bias) > th.drift_limit_degC) & ~cal, th.drift_hold_s)
    if idx is not None:
        raise_alarm("F01", idx, residual_degC=float(bias[idx]), measured_degC=float(run.temp[idx]),
                    model_degC=float(pred[idx]))

    # --- F03: bearing defect tone (FFT band energy + CUSUM) ----------------------------
    p = np.log10(band_power(run.vib, spec.vib_fs, spec.bpfo_hz, th.bpfo_half_bw_hz) + 1e-12)
    mu, sd = float(np.mean(p[cal])), float(np.std(p[cal]) + 1e-9)
    z = (p - mu) / sd
    first, s = cusum_up(np.where(cal, 0.0, z), th.cusum_k, th.cusum_h)
    if first is not None:
        raise_alarm("F03", first, band_hz=[spec.bpfo_hz - th.bpfo_half_bw_hz, spec.bpfo_hz + th.bpfo_half_bw_hz],
                    band_power_vs_baseline_db=float(10 * (p[first] - mu)), cusum=float(s[first]))

    alarms.sort(key=lambda a: a["time_s"])
    return {
        "diagnosis": alarms[0]["code"] if alarms else "OK",
        "alarms": alarms,
        "summary": {
            "duration_s": float(t[-1] + spec.dt),
            "temp_max_degC": round(float(np.max(run.temp)), 3),
            "current_mean_A": round(float(np.mean(clean_i)), 3),
            "current_impulses": int(impulse.sum()),
        },
        "thresholds": asdict(th),
    }


def _round(v):
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, list):
        return [_round(i) for i in v]
    return v
