import hashlib
import time

import numpy as np

from data import epoch_batches, make_rngs, one_hot
from network import get_sizes, init_weights, metrics_at, record_epoch, swarm_error


def sample_ball(n, n_dims, radius, rng):
    direction = rng.standard_normal((n, n_dims))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    r = radius * rng.uniform(size=(n, 1)) ** (1 / n_dims)
    return r * direction


def train_pso(data, lam, n_batches, n_epochs, seed, n_particles=30, inertia=0.729844,
              c1=1.496180, c2=1.496180, reevaluate=False, n_quantum=0, r_cloud=0.5, v_max=None):
    start = time.perf_counter()
    sizes = get_sizes(data)
    batch_rng, init_rng, algo_rng = make_rngs(seed)
    X_train, T_train = data["X_train"], one_hot(data["y_train"], sizes[2])
    n_neutral = n_particles - n_quantum

    x = init_weights(sizes, init_rng, n_particles)
    v = np.zeros_like(x)
    n_weights = x.shape[1]
    pbest = x.copy()
    pbest_err = None
    gbest = None
    history = {}
    best_weights, best_epoch = None, 0
    n_evals = 0
    checksum = hashlib.sha1()

    for epoch in range(1, n_epochs + 1):
        gbest_updates = 0
        max_abs_v = 0.0
        n_clamped, n_components = 0, 0
        for idx in epoch_batches(len(X_train), n_batches, batch_rng):
            checksum.update(idx.tobytes())
            X_b, T_b = X_train[idx], T_train[idx]
            old_gbest = gbest

            if pbest_err is None or (reevaluate and n_batches > 1):
                pbest_err = swarm_error(pbest, X_b, T_b, lam, sizes)
                n_evals += n_particles * len(idx)
                i = np.argmin(pbest_err)
                gbest, gbest_err = pbest[i].copy(), pbest_err[i]

            n = n_neutral
            r1 = algo_rng.uniform(size=(n, n_weights))
            r2 = algo_rng.uniform(size=(n, n_weights))
            v[:n] = inertia * v[:n] + c1 * r1 * (pbest[:n] - x[:n]) + c2 * r2 * (gbest - x[:n])
            if v_max is not None:
                n_clamped += np.count_nonzero(np.abs(v[:n]) > v_max)
                v[:n] = np.clip(v[:n], -v_max, v_max)
            n_components += v[:n].size
            max_abs_v = max(max_abs_v, float(np.abs(v).max()))
            x[:n] = x[:n] + v[:n]

            if n_quantum > 0:
                x[n:] = gbest + sample_ball(n_quantum, n_weights, r_cloud, algo_rng)
                v[n:] = 0

            err = swarm_error(x, X_b, T_b, lam, sizes)
            n_evals += n_particles * len(idx)
            improved = err < pbest_err
            pbest[improved] = x[improved]
            pbest_err[improved] = err[improved]
            i = np.argmin(pbest_err)
            gbest, gbest_err = pbest[i].copy(), pbest_err[i]
            if old_gbest is not None and not np.array_equal(gbest, old_gbest):
                gbest_updates += 1

        record_epoch(history, gbest, data, lam, sizes)
        history.setdefault("n_evals", []).append(n_evals)
        history.setdefault("gbest_err_stored", []).append(float(gbest_err))
        history.setdefault("gbest_updates", []).append(gbest_updates)
        history.setdefault("max_abs_v", []).append(max_abs_v)
        history.setdefault("clamp_fraction", []).append(n_clamped / n_components)
        history.setdefault("diversity", []).append(float(np.mean(np.linalg.norm(x - x.mean(axis=0), axis=1))))
        if best_epoch == 0 or history["val_ce"][-1] < history["val_ce"][best_epoch - 1]:
            best_weights, best_epoch = gbest.copy(), epoch

    return {
        "history": history,
        "best_weights": best_weights,
        "best_epoch": best_epoch,
        "best": metrics_at(history, best_epoch),
        "n_evals": n_evals,
        "batch_checksum": checksum.hexdigest(),
        "time_seconds": time.perf_counter() - start,
    }
