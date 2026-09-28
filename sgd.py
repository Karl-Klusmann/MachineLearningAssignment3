import hashlib
import time

from data import epoch_batches, make_rngs, one_hot
from network import get_sizes, gradient, init_weights, metrics_at, record_epoch


def train_sgd(data, lam, lr, n_batches, n_epochs, seed):
    start = time.perf_counter()
    sizes = get_sizes(data)
    batch_rng, init_rng, _ = make_rngs(seed)
    X_train, T_train = data["X_train"], one_hot(data["y_train"], sizes[2])

    weights = init_weights(sizes, init_rng)
    history = {}
    best_weights, best_epoch = None, 0
    n_evals = 0
    checksum = hashlib.sha1()

    for epoch in range(1, n_epochs + 1):
        for idx in epoch_batches(len(X_train), n_batches, batch_rng):
            checksum.update(idx.tobytes())
            weights = weights - lr * gradient(weights, X_train[idx], T_train[idx], lam, sizes)
            n_evals += len(idx)

        record_epoch(history, weights, data, lam, sizes)
        history.setdefault("n_evals", []).append(n_evals)
        if best_epoch == 0 or history["val_ce"][-1] < history["val_ce"][best_epoch - 1]:
            best_weights, best_epoch = weights.copy(), epoch

    return {
        "history": history,
        "best_weights": best_weights,
        "best_epoch": best_epoch,
        "best": metrics_at(history, best_epoch),
        "n_evals": n_evals,
        "batch_checksum": checksum.hexdigest(),
        "time_seconds": time.perf_counter() - start,
    }
