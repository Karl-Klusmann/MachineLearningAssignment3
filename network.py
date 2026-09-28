import numpy as np
from sklearn.metrics import f1_score

from data import one_hot

SPLITS = ["train", "val", "test"]


def get_sizes(data):
    return (data["X_train"].shape[1], data["n_hidden"], len(data["class_names"]))


def unpack(weights, sizes):
    n_inputs, n_hidden, n_outputs = sizes
    lead = weights.shape[:-1]
    i1 = n_inputs * n_hidden
    i2 = i1 + n_hidden
    i3 = i2 + n_hidden * n_outputs
    W1 = weights[..., :i1].reshape(lead + (n_inputs, n_hidden))
    b1 = weights[..., i1:i2]
    W2 = weights[..., i2:i3].reshape(lead + (n_hidden, n_outputs))
    b2 = weights[..., i3:]
    return W1, b1, W2, b2


def pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])


def init_weights(sizes, rng, n_vectors=None):
    n_inputs, n_hidden, n_outputs = sizes
    shape = () if n_vectors is None else (n_vectors,)
    bound1 = 1 / np.sqrt(n_inputs)
    bound2 = 1 / np.sqrt(n_hidden)
    layer1 = rng.uniform(-bound1, bound1, size=shape + ((n_inputs + 1) * n_hidden,))
    layer2 = rng.uniform(-bound2, bound2, size=shape + ((n_hidden + 1) * n_outputs,))
    return np.concatenate([layer1, layer2], axis=-1)


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def log_softmax(A):
    A = A - A.max(axis=-1, keepdims=True)
    return A - np.log(np.exp(A).sum(axis=-1, keepdims=True))


def log_forward(weights, X, sizes):
    W1, b1, W2, b2 = unpack(weights, sizes)
    if weights.ndim == 2:
        b1, b2 = b1[:, None, :], b2[:, None, :]
    H = sigmoid(X @ W1 + b1)
    return H, log_softmax(H @ W2 + b2)


def forward(weights, X, sizes):
    H, log_P = log_forward(weights, X, sizes)
    return H, np.exp(log_P)


def cross_entropy(weights, X, T, sizes):
    _, log_P = log_forward(weights, X, sizes)
    return -np.sum(T * log_P) / len(X)


def error(weights, X, T, lam, sizes):
    return cross_entropy(weights, X, T, sizes) + lam / 2 * np.sum(weights**2)


def gradient(weights, X, T, lam, sizes):
    W1, b1, W2, b2 = unpack(weights, sizes)
    H, P = forward(weights, X, sizes)
    batch_size = len(X)

    dA2 = (P - T) / batch_size
    dW2 = H.T @ dA2 + lam * W2
    db2 = dA2.sum(axis=0) + lam * b2
    dA1 = (dA2 @ W2.T) * H * (1 - H)
    dW1 = X.T @ dA1 + lam * W1
    db1 = dA1.sum(axis=0) + lam * b1
    return pack(dW1, db1, dW2, db2)


def swarm_error(x, X, T, lam, sizes):
    _, log_P = log_forward(x, X, sizes)
    ce = -np.sum(T * log_P, axis=(1, 2)) / len(X)
    return ce + lam / 2 * np.sum(x**2, axis=1)


def evaluate(weights, X, y, sizes):
    n_outputs = sizes[2]
    _, P = forward(weights, X, sizes)
    y_pred = P.argmax(axis=1)
    f1 = f1_score(y, y_pred, average="macro", labels=np.arange(n_outputs), zero_division=0)
    return {
        "ce": float(cross_entropy(weights, X, one_hot(y, n_outputs), sizes)),
        "acc": float(np.mean(y_pred == y)),
        "f1": float(f1),
    }


def record_epoch(history, weights, data, lam, sizes):
    for split in SPLITS:
        if f"X_{split}" not in data:
            continue
        result = evaluate(weights, data[f"X_{split}"], data[f"y_{split}"], sizes)
        for metric, value in result.items():
            history.setdefault(f"{split}_{metric}", []).append(value)
    T_train = one_hot(data["y_train"], sizes[2])
    history.setdefault("train_error", []).append(float(error(weights, data["X_train"], T_train, lam, sizes)))
    history.setdefault("weight_norm", []).append(float(np.linalg.norm(weights)))
    W1, b1, W2, b2 = unpack(weights, sizes)
    for block, values in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        history.setdefault(f"{block}_norm", []).append(float(np.linalg.norm(values)))
    history.setdefault("max_abs_b1", []).append(float(np.abs(b1).max()))


def metrics_at(history, epoch):
    return {split: {m: history[f"{split}_{m}"][epoch - 1] for m in ["ce", "acc", "f1"]}
            for split in SPLITS if f"{split}_ce" in history}
