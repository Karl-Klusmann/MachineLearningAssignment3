import numpy as np

from data import load_dataset, one_hot
from experiments import ALGORITHMS, run_one
from network import forward, get_sizes, gradient, init_weights

DATASETS = ["wine", "wdbc", "glass"]


def test_sgd_can_overfit_20_rows():
    lr, max_steps = 0.5, 5000
    for name in DATASETS:
        d = load_dataset(name)
        sizes = get_sizes(d)
        X, y = d["X_train"][:20], d["y_train"][:20]
        T = one_hot(y, sizes[2])
        weights = init_weights(sizes, np.random.default_rng(0))
        for step in range(1, max_steps + 1):
            weights = weights - lr * gradient(weights, X, T, 0.0, sizes)
            acc = np.mean(forward(weights, X, sizes)[1].argmax(axis=1) == y)
            if acc == 1.0:
                break
        print(f"\n{name}: training accuracy {acc:.3f} after {step} steps")
        assert acc == 1.0


def test_static_pso_stored_gbest_err_never_increases_on_a_static_problem():
    h = run_one("wine", "pso", n_batches=1, seed=0, n_epochs=300)["history"]
    assert np.all(np.diff(h["gbest_err_stored"]) <= 0)
    print(f"\nstatic pso, wine, full batch, 300 epochs: train ce {h['train_ce'][-1]:.4f}, "
          f"train acc {h['train_acc'][-1]:.3f}")


def test_stored_gbest_err_matches_full_training_objective_with_one_batch():
    for name in DATASETS:
        for algorithm in ["pso", "qpso", "pso_reeval"]:
            h = run_one(name, algorithm, n_batches=1, seed=1, n_epochs=20)["history"]
            assert np.allclose(h["gbest_err_stored"], h["train_error"], rtol=0, atol=1e-9)


def test_new_history_entries_have_one_value_per_epoch():
    for algorithm in ALGORITHMS:
        h = run_one("glass", algorithm, n_batches=4, seed=2, n_epochs=7)["history"]
        for key in ["weight_norm", "W1_norm", "b1_norm", "W2_norm", "b2_norm", "max_abs_b1"]:
            assert len(h[key]) == 7, key
        if algorithm != "sgd":
            assert len(h["gbest_err_stored"]) == 7
        blocks = np.array([h["W1_norm"], h["b1_norm"], h["W2_norm"], h["b2_norm"]])
        assert np.allclose(np.sqrt(np.sum(blocks**2, axis=0)), h["weight_norm"])
