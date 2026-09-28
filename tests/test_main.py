import gzip
import json

import numpy as np

from experiments import count_nonfinite, main_run


def test_main_run_saves_full_history_and_is_skipped_when_done(tmp_path):
    settings = {"lam": 1e-3, "v_max": 0.01, "r_cloud": 0.05, "n_quantum": 5}
    first = main_run("wine", "qpso", 8, 0, settings, 4, tmp_path)
    files = list(tmp_path.glob("*.json.gz"))
    assert len(files) == 1

    record = json.loads(gzip.decompress(files[0].read_bytes()))
    for key in ["train_error", "train_ce", "val_ce", "test_acc", "test_f1", "W1_norm", "diversity",
                "gbest_updates", "gbest_err_stored", "clamp_fraction", "n_evals"]:
        assert len(record["history"][key]) == 4, key
    assert set(record["best"]) == set(record["final"]) == {"train", "val", "test"}

    assert main_run("wine", "qpso", 8, 0, settings, 4, tmp_path) == first
    assert first["n_nonfinite"] == 0


def test_count_nonfinite_looks_inside_nested_records():
    assert count_nonfinite({"a": [1.0, np.nan], "b": {"c": [np.inf, 2.0]}, "d": "text", "e": None}) == 2
