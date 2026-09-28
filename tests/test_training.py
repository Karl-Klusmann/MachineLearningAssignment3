import numpy as np

from data import epoch_batches, load_dataset
from pso import sample_ball, train_pso
from sgd import train_sgd

LAM, LR = 1e-3, 0.1
VARIANTS = {
    "static pso": {"reevaluate": False, "n_quantum": 0},
    "qpso": {"reevaluate": True, "n_quantum": 10},
    "reevaluate only": {"reevaluate": True, "n_quantum": 0},
}


def run(name, data, n_batches, n_epochs, seed):
    if name == "sgd":
        return train_sgd(data, LAM, LR, n_batches, n_epochs, seed)
    return train_pso(data, LAM, n_batches, n_epochs, seed, **VARIANTS[name])


def test_epoch_batches_use_every_row_once_with_near_equal_sizes():
    for n_train in [106, 341, 128]:
        for n_batches in [32, 16, 8, 4, 2, 1]:
            batches = epoch_batches(n_train, n_batches, np.random.default_rng(0))
            sizes = [len(b) for b in batches]
            assert len(batches) == n_batches
            assert np.array_equal(np.sort(np.concatenate(batches)), np.arange(n_train))
            assert max(sizes) - min(sizes) <= 1


def test_epoch_batches_same_seed_gives_same_batches():
    a = epoch_batches(106, 8, np.random.default_rng(1))
    b = epoch_batches(106, 8, np.random.default_rng(1))
    assert all(np.array_equal(i, j) for i, j in zip(a, b))


def test_all_algorithms_see_the_same_batches():
    data = load_dataset("wine")
    checksums = {name: run(name, data, 8, 2, seed=7)["batch_checksum"]
                 for name in ["sgd", "static pso", "qpso"]}
    assert len(set(checksums.values())) == 1
    assert run("sgd", data, 8, 2, seed=8)["batch_checksum"] != checksums["sgd"]


def test_each_algorithm_is_reproducible():
    data = load_dataset("wine")
    for name in ["sgd", *VARIANTS]:
        a, b = run(name, data, 8, 3, seed=3), run(name, data, 8, 3, seed=3)
        assert np.array_equal(a["best_weights"], b["best_weights"])
        assert a["history"] == b["history"]
        assert a["n_evals"] == b["n_evals"] and a["best_epoch"] == b["best_epoch"]


def test_quantum_samples_lie_inside_the_cloud():
    n_weights, r_cloud = 343, 0.5
    rng = np.random.default_rng(4)
    gbest = rng.normal(size=n_weights)
    points = gbest + sample_ball(2000, n_weights, r_cloud, rng)
    dist = np.linalg.norm(points - gbest, axis=1)
    assert dist.max() <= r_cloud
    assert np.isclose(dist.mean(), r_cloud * n_weights / (n_weights + 1), rtol=1e-3)


def test_reevaluation_costs_extra_evals_only_with_several_batches():
    data = load_dataset("wine")
    for n_batches in [8, 1]:
        static = train_pso(data, LAM, n_batches, 2, seed=0)["n_evals"]
        reeval = train_pso(data, LAM, n_batches, 2, seed=0, reevaluate=True)["n_evals"]
        if n_batches > 1:
            assert reeval > static
        else:
            assert reeval == static


def test_n_evals_counts_rows_times_vectors():
    data = load_dataset("wine")
    assert train_sgd(data, LAM, LR, 8, 2, seed=0)["n_evals"] == 2 * 106
    assert train_pso(data, LAM, 8, 2, seed=0)["n_evals"] == 30 * (14 + 2 * 106)
    assert train_pso(data, LAM, 8, 2, seed=0, reevaluate=True)["n_evals"] == 30 * (14 + 2 * 106 + 2 * 106 - 14)


def test_v_max_clamps_every_velocity_component():
    data = load_dataset("wine")
    for variant in VARIANTS.values():
        max_abs_v = train_pso(data, LAM, 8, 5, seed=0, v_max=0.05, **variant)["history"]["max_abs_v"]
        assert max(max_abs_v) <= 0.05
    assert max(train_pso(data, LAM, 8, 5, seed=0)["history"]["max_abs_v"]) > 0.05


def test_clamp_fraction_is_a_fraction_and_zero_without_clamping():
    data = load_dataset("wine")
    for variant in VARIANTS.values():
        clamped = train_pso(data, LAM, 8, 5, seed=0, v_max=0.01, **variant)["history"]["clamp_fraction"]
        assert all(0 <= f <= 1 for f in clamped)
        assert max(clamped) > 0
        free = train_pso(data, LAM, 8, 5, seed=0, **variant)["history"]["clamp_fraction"]
        assert all(f == 0 for f in free)


def test_gbest_updates_counted_once_per_iteration_at_most():
    data = load_dataset("wine")
    for variant in VARIANTS.values():
        updates = train_pso(data, LAM, 8, 5, seed=0, **variant)["history"]["gbest_updates"]
        assert len(updates) == 5
        assert all(0 <= u <= 8 for u in updates)
        assert sum(updates) > 0


def test_smoke_run_on_wine():
    data = load_dataset("wine")
    for name in ["sgd", *VARIANTS]:
        result = run(name, data, 8, 5, seed=0)
        for key, values in result["history"].items():
            assert len(values) == 5, key
            assert np.all(np.isfinite(values)), key
        assert np.all(np.isfinite(result["best_weights"]))
        assert 1 <= result["best_epoch"] <= 5
        print(f"\n{name}: val ce at epoch 5 = {result['history']['val_ce'][-1]:.4f}, "
              f"best epoch {result['best_epoch']}")
