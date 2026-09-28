from pathlib import Path

import numpy as np
import pandas as pd

from data import load_dataset, load_raw

DATASETS = ["wine", "wdbc", "glass"]
DISPLAY_NAMES = {"wine": "Wine", "wdbc": "WDBC", "glass": "Glass"}
RESULTS_DIR = Path(__file__).parent / "results"

summary, split_counts, feature_ranges = [], [], []
for name in DATASETS:
    X, y, feature_names, class_names = load_raw(name)
    d = load_dataset(name)
    counts = np.bincount(y)
    n_train, n_val, n_test = len(d["y_train"]), len(d["y_val"]), len(d["y_test"])

    summary.append({
        "dataset": name,
        "samples": X.shape[0],
        "features": X.shape[1],
        "classes": len(class_names),
        "class_counts": "/".join(str(c) for c in counts),
        "imbalance_ratio": round(counts.max() / counts.min(), 2),
        "missing_values": int(np.isnan(X).sum()),
        "duplicates": int(pd.DataFrame(X).duplicated().sum()),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "architecture": f"{X.shape[1]}-{d['n_hidden']}-{len(class_names)}",
        "n_weights": d["n_weights"],
    })

    for c, class_name in enumerate(class_names):
        split_counts.append({
            "dataset": name,
            "class": c,
            "class_name": class_name,
            "train": int(np.sum(d["y_train"] == c)),
            "val": int(np.sum(d["y_val"] == c)),
            "test": int(np.sum(d["y_test"] == c)),
            "total": int(counts[c]),
        })

    for j, feature in enumerate(feature_names):
        feature_ranges.append({"dataset": name, "feature": feature,
                               "min": X[:, j].min(), "max": X[:, j].max()})

summary = pd.DataFrame(summary)
split_counts = pd.DataFrame(split_counts)
feature_ranges = pd.DataFrame(feature_ranges)

pd.set_option("display.width", 200)
print(summary.to_string(index=False), end="\n\n")
print(split_counts.to_string(index=False), end="\n\n")
print(feature_ranges.to_string(index=False))

RESULTS_DIR.mkdir(exist_ok=True)
summary.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
split_counts.to_csv(RESULTS_DIR / "split_class_counts.csv", index=False)
feature_ranges.to_csv(RESULTS_DIR / "feature_ranges.csv", index=False)

table = pd.DataFrame({
    "Dataset": summary["dataset"].map(DISPLAY_NAMES),
    "Samples": summary["samples"],
    "Features": summary["features"],
    "Classes": summary["classes"],
    "Class counts": summary["class_counts"],
    "Imbalance": summary["imbalance_ratio"],
    "Train/val/test": [f"{a}/{b}/{c}" for a, b, c in zip(summary["n_train"], summary["n_val"], summary["n_test"])],
    "Network": summary["architecture"],
    "Weights": summary["n_weights"],
})
table.to_latex(RESULTS_DIR / "dataset_table.tex", index=False, float_format="%.2f",
               caption="Datasets, splits and network architectures.", label="tab:datasets")
print(f"\nsaved tables to {RESULTS_DIR}")
