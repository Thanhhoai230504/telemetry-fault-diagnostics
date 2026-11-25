import numpy as np

from teldiag.signals import band_power, cusum_up, first_sustained, hampel, kalman_bias, thermal_prediction


def test_hampel_flags_impulses_but_not_a_step():
    rng = np.random.default_rng(0)
    x = 10 + rng.normal(0, 0.05, 400)
    x[200:] += 3.0                      # sustained step: a real change, not noise
    x[[50, 120, 300]] += [6.0, -5.0, 7.0]
    _, outliers = hampel(x, half_window=5, n_sigmas=5)
    assert set(np.flatnonzero(outliers)) >= {50, 120, 300}
    assert not outliers[205:400].any() or set(np.flatnonzero(outliers[205:])) == {300 - 205}


def test_kalman_converges_to_constant_bias():
    rng = np.random.default_rng(1)
    z = 2.0 + rng.normal(0, 0.2, 2000)
    est = kalman_bias(z, q=1e-5, r=0.04)
    assert abs(est[-1] - 2.0) < 0.1


def test_cusum_alarms_on_mean_shift_only_after_it():
    z = np.r_[np.zeros(100), np.full(100, 2.0)]
    first, _ = cusum_up(z, k=1.0, h=10.0)
    assert first is not None and 100 <= first < 115


def test_band_power_picks_up_the_tone():
    fs, m = 2048.0, 256
    t = np.arange(m) / fs
    tone = np.sin(2 * np.pi * 107 * t)[None, :]
    other = np.sin(2 * np.pi * 300 * t)[None, :]
    assert band_power(tone, fs, 107, 6)[0] > 100 * band_power(other, fs, 107, 6)[0]


def test_thermal_prediction_reaches_steady_state():
    pred = thermal_prediction(np.full(3000, 10.0), t0=30.0, ambient=30.0, r_thermal=0.25, tau=300.0, dt=1.0)
    assert abs(pred[-1] - 55.0) < 0.1


def test_first_sustained():
    assert first_sustained(np.array([1, 1, 0, 1, 1, 1], bool), 3) == 5
    assert first_sustained(np.array([1, 0, 1], bool), 2) is None
