import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
GLASS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

SPLIT_SEED = 2026

N_HIDDEN = {"wine": 20, "wdbc": 20, "glass": 18}

GLASS_FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
GLASS_CLASSES = ["building float", "building non-float", "vehicle float",
                 "containers", "tableware", "headlamps"]
GLASS_TYPE_TO_CLASS = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5}
GLASS_OPENML_CLASSES = ["build wind float", "build wind non-float", "vehic wind float",
                        "containers", "tableware", "headlamps"]


def load_glass():
    path = DATA_DIR / "glass.data"
    if not path.exists():
        try:
            with urllib.request.urlopen(GLASS_URL, timeout=30) as response:
                content = response.read()
            DATA_DIR.mkdir(exist_ok=True)
            path.write_bytes(content)
        except OSError as e:
            print(f"glass download failed ({e}), using openml instead")

    if path.exists():
        raw = np.loadtxt(path, delimiter=",")
        X = raw[:, 1:10]
        y = np.array([GLASS_TYPE_TO_CLASS[t] for t in raw[:, 10].astype(int)])
    else:
        glass = fetch_openml(name="glass", version=1, as_frame=True, data_home=DATA_DIR)
        X = glass.data[GLASS_FEATURES].to_numpy(float)
        y = glass.target.map({c: i for i, c in enumerate(GLASS_OPENML_CLASSES)}).to_numpy(int)

    order = np.lexsort(X.T)
    return X[order], y[order]


def load_raw(name):
    if name == "glass":
        X, y = load_glass()
        return X, y, GLASS_FEATURES, GLASS_CLASSES
    loaders = {"wine": load_wine, "wdbc": load_breast_cancer}
    d = loaders[name]()
    return d.data, d.target, [str(f) for f in d.feature_names], [str(c) for c in d.target_names]


def split_indices(y):
    idx = np.arange(len(y))
    idx_rest, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    idx_train, idx_val = train_test_split(idx_rest, test_size=0.25, stratify=y[idx_rest],
                                          random_state=SPLIT_SEED)
    return idx_train, idx_val, idx_test


def load_dataset(name):
    X, y, feature_names, class_names = load_raw(name)
    idx_train, idx_val, idx_test = split_indices(y)

    mean = X[idx_train].mean(axis=0)
    std = X[idx_train].std(axis=0)
    X = (X - mean) / std

    n_inputs = X.shape[1]
    n_hidden = N_HIDDEN[name]
    n_outputs = len(class_names)
    return {
        "X_train": X[idx_train], "y_train": y[idx_train],
        "X_val": X[idx_val], "y_val": y[idx_val],
        "X_test": X[idx_test], "y_test": y[idx_test],
        "feature_names": feature_names,
        "class_names": class_names,
        "n_hidden": n_hidden,
        "n_weights": (n_inputs + 1) * n_hidden + (n_hidden + 1) * n_outputs,
    }


def one_hot(y, n_outputs):
    return np.eye(n_outputs)[y]


def make_rngs(seed):
    batch_seed, init_seed, algo_seed = np.random.SeedSequence(seed).spawn(3)
    return (np.random.default_rng(batch_seed), np.random.default_rng(init_seed),
            np.random.default_rng(algo_seed))


def epoch_batches(n_train, n_batches, rng):
    return np.array_split(rng.permutation(n_train), n_batches)
