import numpy as np

from data import load_dataset, one_hot
from network import (cross_entropy, error, evaluate, forward, get_sizes, gradient, init_weights,
                     pack, swarm_error, unpack)

DATASETS = ["wine", "wdbc", "glass"]


def train_batch(name, n_rows=None):
    d = load_dataset(name)
    sizes = get_sizes(d)
    X, y = d["X_train"][:n_rows], d["y_train"][:n_rows]
    return X, one_hot(y, sizes[2]), sizes


def test_unpack_shapes_and_pack_roundtrip():
    sizes = (13, 20, 3)
    weights = np.arange(343, dtype=float)
    W1, b1, W2, b2 = unpack(weights, sizes)
    assert W1.shape == (13, 20) and b1.shape == (20,) and W2.shape == (20, 3) and b2.shape == (3,)
    assert W1[0, 0] == 0 and b1[0] == 260 and W2[0, 0] == 280 and b2[0] == 340
    assert np.array_equal(pack(W1, b1, W2, b2), weights)

    x = init_weights(sizes, np.random.default_rng(0), n_vectors=5)
    W1, b1, W2, b2 = unpack(x, sizes)
    assert W1.shape == (5, 13, 20) and b1.shape == (5, 20) and W2.shape == (5, 20, 3) and b2.shape == (5, 3)
    assert np.array_equal(pack(W1[2], b1[2], W2[2], b2[2]), x[2])


def test_gradient_matches_central_differences():
    lam, eps = 1e-3, 1e-5
    for name in DATASETS:
        X, T, sizes = train_batch(name, n_rows=10)
        weights = init_weights(sizes, np.random.default_rng(1))
        numeric = np.zeros_like(weights)
        for i in range(len(weights)):
            step = np.zeros_like(weights)
            step[i] = eps
            numeric[i] = (error(weights + step, X, T, lam, sizes)
                          - error(weights - step, X, T, lam, sizes)) / (2 * eps)
        backprop = gradient(weights, X, T, lam, sizes)
        rel_err = np.linalg.norm(numeric - backprop) / (np.linalg.norm(numeric) + np.linalg.norm(backprop))
        print(f"\n{name}: gradient check relative error {rel_err:.2e}")
        assert rel_err < 1e-7


def test_penalty_covers_all_weights_including_biases():
    lam = 0.1
    X, T, sizes = train_batch("glass")
    weights = init_weights(sizes, np.random.default_rng(2))
    penalty = error(weights, X, T, lam, sizes) - error(weights, X, T, 0.0, sizes)
    assert np.isclose(penalty, lam / 2 * np.sum(weights**2))


def test_swarm_error_matches_error_per_vector():
    lam = 1e-3
    for name in DATASETS:
        X, T, sizes = train_batch(name)
        x = init_weights(sizes, np.random.default_rng(3), n_vectors=5)
        expected = np.array([error(weights, X, T, lam, sizes) for weights in x])
        assert np.allclose(swarm_error(x, X, T, lam, sizes), expected, rtol=0, atol=1e-12)


def test_softmax_rows_sum_to_one_and_huge_weights_stay_finite():
    X, T, sizes = train_batch("wdbc")
    rng = np.random.default_rng(4)
    for scale in [1, 1000]:
        weights = scale * init_weights(sizes, rng)
        H, P = forward(weights, X, sizes)
        assert np.allclose(P.sum(axis=1), 1)
        assert np.all(np.isfinite(H)) and np.all(np.isfinite(P))
        assert np.isfinite(cross_entropy(weights, X, T, sizes))
        assert np.all(np.isfinite(gradient(weights, X, T, 1e-3, sizes)))
        assert np.all(np.isfinite(swarm_error(weights[None], X, T, 1e-3, sizes)))


def test_init_weights_within_layer_bounds_and_reproducible():
    sizes = (30, 20, 2)
    x = init_weights(sizes, np.random.default_rng(5), n_vectors=50)
    W1, b1, W2, b2 = unpack(x, sizes)
    bound1, bound2 = 1 / np.sqrt(30), 1 / np.sqrt(20)
    assert np.abs(W1).max() <= bound1 and np.abs(b1).max() <= bound1
    assert np.abs(W2).max() <= bound2 and np.abs(b2).max() <= bound2
    assert np.abs(W2).max() > bound1

    assert np.array_equal(init_weights(sizes, np.random.default_rng(5), n_vectors=50), x)
    assert init_weights(sizes, np.random.default_rng(5)).shape == (662,)


def test_evaluate_counts_unpredicted_classes_as_zero_f1():
    d = load_dataset("glass")
    sizes = get_sizes(d)
    X, y = d["X_test"], d["y_test"]
    W1, b1, W2, b2 = np.zeros((9, 18)), np.zeros(18), np.zeros((18, 6)), np.zeros(6)
    b2[0] = 10
    result = evaluate(pack(W1, b1, W2, b2), X, y, sizes)
    n0 = np.sum(y == 0)
    assert np.isclose(result["acc"], n0 / len(y))
    assert np.isclose(result["f1"], 2 * n0 / (n0 + len(y)) / 6)


def test_gradient_descent_lowers_training_cross_entropy():
    lr, n_steps = 0.5, 200
    X, T, sizes = train_batch("wine")
    weights = init_weights(sizes, np.random.default_rng(6))
    start = cross_entropy(weights, X, T, sizes)
    for _ in range(n_steps):
        weights = weights - lr * gradient(weights, X, T, 0.0, sizes)
    end = cross_entropy(weights, X, T, sizes)
    print(f"\nwine full-batch gd: cross-entropy {start:.4f} -> {end:.4f}")
    assert end < start
