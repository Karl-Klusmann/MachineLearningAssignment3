import json

import numpy as np

import experiments
from experiments import best_row, convergence_status, lower_edges, run_one, tune_run


def test_tuning_run_never_sees_the_test_set(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "N_EPOCHS", 5)
    for algorithm, settings in [("sgd", {"lam": 1e-3, "lr": 0.1}),
                                ("qpso", {"lam": 1e-3, "v_max": 0.5, "r_cloud": 0.2, "n_quantum": 5})]:
        summary = tune_run("wine", algorithm, 8, 100, settings, tmp_path)
        assert "test" not in json.dumps(summary)

    r = run_one("wine", "qpso", 8, seed=100, n_epochs=3, with_test=False)
    assert not any("test" in key for key in r["history"])
    assert "test" not in r["best"]


def test_finished_tuning_runs_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "N_EPOCHS", 5)
    settings = {"lam": 1e-3, "lr": 0.1}
    first = tune_run("wine", "sgd", 4, 100, settings, tmp_path)
    second = tune_run("wine", "sgd", 4, 100, settings, tmp_path)
    assert first == second
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_tuning_run_saves_the_training_objective(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "N_EPOCHS", 60)
    summary = tune_run("wine", "sgd", 4, 100, {"lam": 1e-3, "lr": 0.1}, tmp_path)
    assert len(summary["train_error"]) == 60
    assert np.isclose(summary["train_objective_last50"], np.mean(summary["train_error"][-50:]))


def test_lower_edges_finds_the_next_value_along_one_parameter():
    def row(v_max, r_cloud):
        return {"dataset": "wine", "algorithm": "qpso", "n_batches": 32,
                "settings": {"lam": 1e-3, "v_max": v_max, "r_cloud": r_cloud, "n_quantum": 15}}

    rows = [row(None, 0.2), row(0.1, 0.2), row(0.001, 0.2), row(0.001, 0.05), row(0.001, 1.0)]
    assert lower_edges(rows, row(0.001, 0.05)) == [("v_max", 0.0003), ("r_cloud", 0.0125)]
    assert lower_edges(rows, row(None, 0.2)) == []
    rows += [row(0.0001, 0.003)]
    assert lower_edges(rows, row(0.0001, 0.003)) == [("v_max", None), ("r_cloud", None)]


def test_convergence_status_uses_one_percent_either_way():
    assert convergence_status(-0.02) == "still improving"
    assert convergence_status(-0.005) == "converged"
    assert convergence_status(0.005) == "converged"
    assert convergence_status(0.02) == "late drift"


def test_diverged_configuration_is_never_chosen():
    rows = [{"lr": 243, "score": np.nan}, {"lr": 1, "score": 0.5}, {"lr": 3, "score": 0.2}]
    assert best_row(rows, "score")["lr"] == 3
