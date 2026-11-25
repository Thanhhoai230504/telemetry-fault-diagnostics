import json

import numpy as np
import pytest

from teldiag.cli import main
from teldiag.diagnose import CODE_FOR_FAULT, diagnose
from teldiag.evaluate import score_run, summarize
from teldiag.simulate import Run, simulate


def test_simulation_is_reproducible():
    a, b = simulate("bearing", seed=3), simulate("bearing", seed=3)
    assert a.onset == b.onset
    np.testing.assert_array_equal(a.vib, b.vib)


@pytest.mark.parametrize("seed", range(5))
def test_healthy_runs_raise_no_alarm(seed):
    report = diagnose(simulate("none", seed=100 + seed))
    assert report["diagnosis"] == "OK" and report["alarms"] == []


@pytest.mark.parametrize("fault", ["drift", "spikes", "bearing", "overload"])
@pytest.mark.parametrize("seed", range(3))
def test_each_fault_gets_its_own_code_after_onset(fault, seed):
    run = simulate(fault, seed=200 + seed)
    report = diagnose(run)
    assert report["diagnosis"] == CODE_FOR_FAULT[fault]
    assert report["alarms"][0]["time_s"] >= run.onset


def test_overload_is_not_blamed_on_the_temperature_sensor():
    # temperature rises a lot under overload, but it follows the thermal model
    run = simulate("overload", seed=11, onset=600)
    codes = {a["code"] for a in diagnose(run)["alarms"]}
    assert codes == {"F04"}
    assert run.temp[-60:].mean() > run.temp[:300].mean() + 10


def test_drift_is_not_blamed_on_load():
    codes = {a["code"] for a in diagnose(simulate("drift", seed=12, onset=600))["alarms"]}
    assert codes == {"F01"}


def test_scoring_counts_early_alarms_as_false():
    report = {"diagnosis": "F03", "alarms": [{"code": "F03", "time_s": 500.0}]}
    r = score_run("bearing", onset=800.0, duration=1800.0, report=report)
    assert r["false_alarms"] == 1 and r["latency_s"] is None
    s = summarize([r])
    assert s["per_class"]["F03"]["detected"] == 0


def test_cli_round_trip(tmp_path, capsys):
    path = tmp_path / "run.npz"
    assert main(["simulate", "--fault", "drift", "--seed", "5", "--onset", "700", "--out", str(path)]) == 0
    run = Run.load(str(path))
    assert run.fault == "drift" and run.onset == 700.0
    capsys.readouterr()
    assert main(["diagnose", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["diagnosis"] == "F01"


def test_cli_evaluate_gate(tmp_path):
    out = tmp_path / "metrics.json"
    assert main(["evaluate", "--runs-per-class", "3", "--min-accuracy", "0.9",
                 "--max-fp-rate", "0.0", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["accuracy"] >= 0.9
