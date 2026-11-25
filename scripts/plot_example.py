"""Draw docs/example.png: one run per fault with the diagnoser's alarm marked. Needs matplotlib."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from teldiag.diagnose import diagnose
from teldiag.signals import band_power
from teldiag.simulate import simulate

panels = [("drift", "temp", "Temperature sensor drift (F01)", "degC"),
          ("spikes", "current", "Current impulse noise (F02)", "A"),
          ("bearing", "vib", "Bearing wear (F03): power at 107 Hz", "g$^2$"),
          ("overload", "current", "Overload (F04)", "A")]
fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
for ax, (fault, channel, title, unit) in zip(axes, panels):
    run = simulate(fault, seed=21, onset=800)
    y = band_power(run.vib, run.spec.vib_fs, run.spec.bpfo_hz, 6.0) if channel == "vib" else getattr(run, channel)
    ax.plot(run.t, y, lw=0.7, color="#3b6ea5")
    ax.axvline(run.onset, color="#888", ls="--", lw=1, label="fault onset")
    alarm = diagnose(run)["alarms"][0]
    ax.axvline(alarm["time_s"], color="#c0392b", lw=1.2, label=f"alarm {alarm['code']} (+{alarm['time_s'] - run.onset:.0f} s)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(unit)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
axes[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig("docs/example.png", dpi=110)
print("wrote docs/example.png")
