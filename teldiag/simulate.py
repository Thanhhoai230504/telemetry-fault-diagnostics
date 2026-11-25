"""Synthetic telemetry for a motor-driven pump unit, with injectable faults.

The unit reports, once per second:
  * winding temperature (degC) from a thermistor,
  * phase current (A) from a current transformer,
  * one 125 ms vibration window from an accelerometer on the bearing housing.

Temperature follows a first-order thermal model driven by I^2 losses, so a real
overload moves current *and* temperature together, while a drifting sensor moves
temperature alone. That difference is what the diagnoser uses for root cause.

Fault types (at most one per run, starting at `onset`):
  drift     temperature sensor reading drifts away linearly
  spikes    impulse noise on the current channel (EMI / loose wiring)
  bearing   outer-race bearing defect tone grows in the vibration spectrum
  overload  sustained load increase: current steps up, temperature follows
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

FAULTS = ("none", "drift", "spikes", "bearing", "overload")


@dataclass(frozen=True)
class UnitSpec:
    """Physical parameters of the simulated unit (known from commissioning)."""

    dt: float = 1.0                # s between telemetry frames
    ambient: float = 30.0          # degC
    r_thermal: float = 0.25        # degC per A^2 of steady-state rise
    tau: float = 300.0             # s, thermal time constant
    i_nominal: float = 10.0        # A
    vib_fs: float = 2048.0         # Hz, accelerometer sample rate
    vib_window: int = 256          # samples per frame (125 ms)
    shaft_hz: float = 25.0         # 1500 rpm
    bpfo_hz: float = 107.0         # outer-race defect frequency for this bearing


@dataclass
class Run:
    t: np.ndarray              # (n,) seconds
    temp: np.ndarray           # (n,) measured temperature, degC
    current: np.ndarray        # (n,) measured current, A
    vib: np.ndarray            # (n, vib_window) acceleration, g
    fault: str = "none"
    onset: float | None = None
    params: dict = field(default_factory=dict)
    seed: int = 0
    spec: UnitSpec = field(default_factory=UnitSpec)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, t=self.t, temp=self.temp, current=self.current, vib=self.vib,
            fault=self.fault, onset=np.nan if self.onset is None else self.onset,
            seed=self.seed, params=np.array(json.dumps(self.params, default=float)),
            spec=np.array(json.dumps(asdict(self.spec))),
        )

    @staticmethod
    def load(path: str) -> "Run":
        d = np.load(path, allow_pickle=False)
        onset = float(d["onset"])
        spec = UnitSpec(**json.loads(str(d["spec"])))
        return Run(
            t=d["t"], temp=d["temp"], current=d["current"], vib=d["vib"],
            fault=str(d["fault"]), onset=None if np.isnan(onset) else onset,
            params=json.loads(str(d["params"])), seed=int(d["seed"]), spec=spec,
        )


def simulate(fault: str = "none", seed: int = 0, duration: float = 1800.0,
             onset: float | None = None, spec: UnitSpec | None = None,
             severity: float = 1.0) -> Run:
    """Simulate one run. If `onset` is None a fault starts at a random time in [600, 1200) s.

    `severity` scales the fault's size (drift rate, impulse height, defect growth, extra load):
    1.0 is the default fault, smaller values make it harder to detect.
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    spec = spec or UnitSpec()
    rng = np.random.default_rng(seed)
    n = int(round(duration / spec.dt))
    t = np.arange(n) * spec.dt
    if fault == "none":
        onset = None
    elif onset is None:
        onset = float(rng.uniform(600.0, 1200.0))
    after = np.zeros(n, dtype=bool) if onset is None else t >= onset
    params: dict = {"severity": severity}

    # --- load current: nominal + slow wander + measurement noise -------------
    wander = np.cumsum(rng.normal(0.0, 0.01, n))
    wander -= np.linspace(0.0, wander[-1], n)          # keep the wander bounded
    i_true = spec.i_nominal + 0.3 * np.tanh(wander)
    if fault == "overload":
        factor = 1.0 + float(rng.uniform(0.25, 0.40)) * severity
        params["load_factor"] = round(factor, 3)
        ramp = np.clip((t - onset) / 5.0, 0.0, 1.0)     # 5 s ramp to the new load
        i_true = i_true * (1.0 + (factor - 1.0) * ramp)

    # --- winding temperature: first-order thermal model ------------------------
    ambient = spec.ambient + 0.5 * np.sin(2 * np.pi * t / 3600.0 + rng.uniform(0, 2 * np.pi))
    temp_true = np.empty(n)
    temp_true[0] = ambient[0] + spec.r_thermal * i_true[0] ** 2
    alpha = spec.dt / spec.tau
    for k in range(1, n):
        target = ambient[k] + spec.r_thermal * i_true[k - 1] ** 2
        temp_true[k] = temp_true[k - 1] + alpha * (target - temp_true[k - 1])
    temp = temp_true + rng.normal(0.0, 0.15, n)
    if fault == "drift":
        rate = float(rng.uniform(0.008, 0.02)) * severity * rng.choice([-1.0, 1.0])
        params["drift_degC_per_s"] = round(float(rate), 4)
        temp = temp + np.where(after, rate * (t - (onset or 0.0)), 0.0)

    current = i_true + rng.normal(0.0, 0.08, n)
    if fault == "spikes":
        rate = float(rng.uniform(1 / 40, 1 / 15))      # spikes per second after onset
        params["spikes_per_s"] = round(rate, 4)
        hits = after & (rng.random(n) < rate)
        current = current + np.where(hits, rng.uniform(3.0, 8.0, n) * severity * rng.choice([-1.0, 1.0], n), 0.0)
        params["n_spikes"] = int(hits.sum())

    # --- vibration windows ------------------------------------------------------
    m = spec.vib_window
    tau_s = np.arange(m) / spec.vib_fs
    phase = rng.uniform(0, 2 * np.pi, (n, 2))
    tt = t[:, None] + tau_s[None, :]
    vib = (1.0 * np.sin(2 * np.pi * spec.shaft_hz * tt + phase[:, :1])
           + 0.4 * np.sin(2 * np.pi * 2 * spec.shaft_hz * tt + phase[:, 1:])
           + rng.normal(0.0, 0.3, (n, m)))
    if fault == "bearing":
        growth = float(rng.uniform(0.0008, 0.002)) * severity     # g of defect tone per second
        params["defect_growth_g_per_s"] = round(growth, 5)
        amp = np.where(after, growth * (t - onset), 0.0)[:, None]
        vib = vib + amp * np.sin(2 * np.pi * spec.bpfo_hz * tt + rng.uniform(0, 2 * np.pi, (n, 1)))

    return Run(t=t, temp=temp, current=current, vib=vib, fault=fault,
               onset=onset, params=params, seed=seed, spec=spec)
