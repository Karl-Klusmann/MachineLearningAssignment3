import numpy as np

from data import load_dataset, load_raw, one_hot, split_indices

EXPECTED = {
    "wine": {"shape": (178, 13), "counts": [59, 71, 48], "splits": (106, 36, 36), "n_weights": 343},
    "wdbc": {"shape": (569, 30), "counts": [212, 357], "splits": (341, 114, 114), "n_weights": 662},
    "glass": {"shape": (214, 9), "counts": [70, 76, 17, 13, 9, 29], "splits": (128, 43, 43), "n_weights": 294},
}
ARRAYS = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]


def test_shapes_and_class_counts():
    for name, expected in EXPECTED.items():
        X, y, _, _ = load_raw(name)
        assert X.shape == expected["shape"]
        assert np.bincount(y).tolist() == expected["counts"]


def test_wdbc_malignant_is_class_zero():
    _, _, _, class_names = load_raw("wdbc")
    assert class_names == ["malignant", "benign"]


def test_glass_has_no_id_column():
    _, _, feature_names, _ = load_raw("glass")
    assert feature_names == ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]


def test_splits_are_disjoint_with_expected_sizes():
    for name, expected in EXPECTED.items():
        _, y, _, _ = load_raw(name)
        idx_train, idx_val, idx_test = split_indices(y)
        assert (len(idx_train), len(idx_val), len(idx_test)) == expected["splits"]
        assert len(np.unique(np.concatenate([idx_train, idx_val, idx_test]))) == len(y)

        d = load_dataset(name)
        assert (len(d["y_train"]), len(d["y_val"]), len(d["y_test"])) == expected["splits"]


def test_training_features_are_standardised():
    for name in EXPECTED:
        X_train = load_dataset(name)["X_train"]
        assert np.allclose(X_train.mean(axis=0), 0)
        assert np.allclose(X_train.std(axis=0), 1)


def test_loading_twice_gives_identical_arrays():
    for name in EXPECTED:
        a, b = load_dataset(name), load_dataset(name)
        for key in ARRAYS:
            assert np.array_equal(a[key], b[key])


def test_n_weights():
    for name, expected in EXPECTED.items():
        assert load_dataset(name)["n_weights"] == expected["n_weights"]


def test_one_hot():
    T = one_hot(np.array([0, 2, 1]), 3)
    assert np.array_equal(T, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
