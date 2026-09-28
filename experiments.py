import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[var] = "1"

import gzip
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data import load_dataset
from network import metrics_at
from pso import train_pso
from sgd import train_sgd

N_PARTICLES = 30
INERTIA = 0.729844
C1 = 1.496180
C2 = 1.496180

LAM = 1e-3
LR = 0.1
N_QUANTUM = 10
R_CLOUD = 0.5

DATASETS = ["wine", "wdbc", "glass"]
ALGORITHMS = ["sgd", "pso", "qpso", "pso_reeval"]
N_BATCHES = [32, 16, 8, 4, 2, 1]
N_EPOCHS = 1000
N_RUNS = 30
N_JOBS = -1

TUNING_SEEDS = [100, 101, 102, 103, 104]
MAIN_SEEDS = list(range(N_RUNS))
STAGE1_N_BATCHES = 4
LAM_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LR_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
V_MAX_GRID = [None, 0.5, 0.1, 0.03, 0.01]
V_MAX_MIN = 0.001
V_MAX_EXTENSIONS = [0.003, 0.001, 0.0003, 0.0001]
R_CLOUD_EXTENSIONS = [0.0125, 0.003]
EDGE_ROUNDS = 3
R_CLOUD_GRID = [0.05, 0.2, 1.0]
N_QUANTUM_GRID = [5, 15]

PILOT_SEEDS = [0, 1, 2]
PILOT_N_BATCHES = [32, 1]
PILOT_EPOCHS = 300
TIMING_EPOCHS = 10

RESULTS_DIR = Path(__file__).parent / "results"
PILOT_DIR = RESULTS_DIR / "pilot"
TUNING_DIR = RESULTS_DIR / "tuning"
TUNING_RUNS_DIR = TUNING_DIR / "runs"
OBJECTIVE_RUNS_DIR = TUNING_DIR / "runs_objective"
CHOSEN = TUNING_DIR / "chosen.json"

VAL_CRITERION = "mean_best_val_ce"
OBJECTIVE_CRITERION = "mean_train_objective_last50"
LAST_EPOCHS = 50
LR_MAX = 243
ROBUSTNESS_LR = 1
CONVERGENCE_WINDOWS = ((851, 900), (951, 1000))
CONVERGENCE_TOL = 0.01

MAIN_DIR = RESULTS_DIR / "raw" / "main"
MAIN_CSV = RESULTS_DIR / "main_runs.csv"
GLASS_LONG_EPOCHS = 2000
GLASS_LONG_DIR = RESULTS_DIR / "raw" / "glass2000"
GLASS_LONG_CSV = RESULTS_DIR / "glass2000_runs.csv"

PSO_VARIANTS = {"pso": (False, False), "qpso": (True, True), "pso_reeval": (True, False)}
LABELS = {"sgd": "SGD", "pso": "static PSO", "qpso": "QPSO", "pso_reeval": "PSO + re-evaluation"}
COLORS = {"sgd": "#2a78d6", "pso": "#eb6834", "qpso": "#1baf7a", "pso_reeval": "#eda100"}


def run_one(dataset, algorithm, n_batches, seed, n_epochs, with_test=True, **overrides):
    if algorithm == "sgd":
        settings = {"lam": LAM, "lr": LR}
    else:
        reevaluate, quantum = PSO_VARIANTS[algorithm]
        settings = {"lam": LAM, "n_particles": N_PARTICLES, "inertia": INERTIA, "c1": C1, "c2": C2,
                    "reevaluate": reevaluate, "n_quantum": N_QUANTUM if quantum else 0}
        if quantum:
            settings["r_cloud"] = R_CLOUD
    settings.update(overrides)

    data = load_dataset(dataset)
    if not with_test:
        data = {key: value for key, value in data.items() if not key.endswith("_test")}
    train = train_sgd if algorithm == "sgd" else train_pso
    result = train(data, n_batches=n_batches, n_epochs=n_epochs, seed=seed, **settings)
    result.update(dataset=dataset, algorithm=algorithm, n_batches=n_batches, seed=seed,
                  n_epochs=n_epochs, settings=settings)
    return result


def save_result(result, path):
    path.write_text(json.dumps(dict(result, best_weights=result["best_weights"].tolist())))


def mean_curve(results, key, **match):
    runs = [r for r in results if all(r[k] == v for k, v in match.items())]
    return np.mean([r["history"][key] for r in runs], axis=0)


def plot_objective(results):
    epochs = np.arange(1, PILOT_EPOCHS + 1)
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, n_batches in zip(axes, PILOT_N_BATCHES):
            for algorithm in ALGORITHMS:
                curve = mean_curve(results, "train_error", dataset=dataset, algorithm=algorithm, n_batches=n_batches)
                ax.plot(epochs, curve, color=COLORS[algorithm], lw=1.5, label=LABELS[algorithm])
            ax.set_yscale("log")
            ax.set_title(f"{dataset}, n_batches = {n_batches}")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("training objective (mean of 3 seeds)")
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(PILOT_DIR / f"{dataset}_objective.png", dpi=150)
        plt.close(fig)


def plot_stale_memory(results):
    epochs = np.arange(1, PILOT_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, algorithm in zip(axes, ["pso", "qpso"]):
        match = {"dataset": "wine", "algorithm": algorithm, "n_batches": 32}
        ax.plot(epochs, mean_curve(results, "train_error", **match), color=COLORS[algorithm], lw=1.5,
                label="actual (full training set)")
        ax.plot(epochs, mean_curve(results, "gbest_err_stored", **match), color=COLORS[algorithm], lw=1.5,
                ls="--", label="stored gbest_err")
        ax.set_yscale("log")
        ax.set_title(f"{LABELS[algorithm]}, wine, n_batches = 32")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    axes[0].set_ylabel("objective of gbest (mean of 3 seeds)")
    fig.tight_layout()
    fig.savefig(PILOT_DIR / "wine_stale_memory.png", dpi=150)
    plt.close(fig)


def pilot():
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    configs = [(dataset, algorithm, n_batches, seed) for dataset in DATASETS for n_batches in PILOT_N_BATCHES
               for algorithm in ALGORITHMS for seed in PILOT_SEEDS]
    print(f"running {len(configs)} pilot runs of {PILOT_EPOCHS} epochs")
    start = time.perf_counter()
    results = Parallel(n_jobs=N_JOBS)(delayed(run_one)(*config, PILOT_EPOCHS) for config in configs)
    print(f"finished in {time.perf_counter() - start:.0f} s\n")
    for r in results:
        save_result(r, PILOT_DIR / f"{r['dataset']}_{r['algorithm']}_nb{r['n_batches']}_seed{r['seed']}.json")

    rows = []
    for r in results:
        rows.append({
            "dataset": r["dataset"],
            "n_batches": r["n_batches"],
            "algorithm": r["algorithm"],
            "best_val_ce": r["best"]["val"]["ce"],
            "test_acc_at_best": r["best"]["test"]["acc"],
            "best_epoch": r["best_epoch"],
            "final_weight_norm": r["history"]["weight_norm"][-1],
            **{f"final_{key}": r["history"][key][-1]
               for key in ["W1_norm", "b1_norm", "W2_norm", "b2_norm", "max_abs_b1"]},
            "final_stored": r["history"].get("gbest_err_stored", [np.nan])[-1],
            "final_actual": r["history"]["train_error"][-1],
        })
    table = pd.DataFrame(rows).groupby(["dataset", "n_batches", "algorithm"], sort=False).mean()
    table.to_csv(PILOT_DIR / "pilot_table.csv")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())

    plot_objective(results)
    plot_stale_memory(results)
    print(f"\nsaved results and figures to {PILOT_DIR}")


def seconds_per_epoch(with_test=True):
    run_one("wine", "qpso", 8, seed=0, n_epochs=2, with_test=with_test)
    return {(dataset, algorithm, n_batches):
            run_one(dataset, algorithm, n_batches, seed=0, n_epochs=TIMING_EPOCHS,
                    with_test=with_test)["time_seconds"] / TIMING_EPOCHS
            for dataset in DATASETS for n_batches in N_BATCHES for algorithm in ALGORITHMS}


def run_name(dataset, algorithm, n_batches, seed, settings):
    tuned = "_".join(f"{key}{value}" for key, value in sorted(settings.items()))
    return f"{dataset}_{algorithm}_nb{n_batches}_{tuned}_seed{seed}"


def tune_run(dataset, algorithm, n_batches, seed, settings, runs_dir):
    path = runs_dir / f"{run_name(dataset, algorithm, n_batches, seed, settings)}.json"
    if path.exists():
        return json.loads(path.read_text())

    r = run_one(dataset, algorithm, n_batches, seed, N_EPOCHS, with_test=False, **settings)
    h = r["history"]
    summary = {
        "dataset": dataset,
        "algorithm": algorithm,
        "n_batches": n_batches,
        "seed": seed,
        "n_epochs": N_EPOCHS,
        "settings": settings,
        "best_val_ce": r["best"]["val"]["ce"],
        "best_epoch": r["best_epoch"],
        "train_objective_last50": float(np.mean(h["train_error"][-LAST_EPOCHS:])),
        "train_error": h["train_error"],
        "final_W1_norm": h["W1_norm"][-1],
        "mean_gbest_updates": float(np.mean(h["gbest_updates"])) if "gbest_updates" in h else None,
        "n_evals": r["n_evals"],
        "time_seconds": r["time_seconds"],
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary))
    tmp.replace(path)
    return summary


def run_many(jobs, runs_dir):
    return Parallel(n_jobs=N_JOBS)(delayed(tune_run)(*job, runs_dir) for job in jobs)


def summarise(summaries):
    groups = {}
    for s in summaries:
        key = (s["dataset"], s["algorithm"], s["n_batches"], tuple(sorted(s["settings"].items())))
        groups.setdefault(key, []).append(s)

    rows = []
    for (dataset, algorithm, n_batches, settings), runs in groups.items():
        best_val_ce = [r["best_val_ce"] for r in runs]
        best_epochs = [r["best_epoch"] for r in runs]
        row = {"dataset": dataset, "algorithm": algorithm, "n_batches": n_batches, **dict(settings),
               "settings": dict(settings),
               "n_runs": len(runs),
               "mean_best_val_ce": np.mean(best_val_ce),
               "std_best_val_ce": np.std(best_val_ce, ddof=1),
               "mean_best_epoch": np.mean(best_epochs),
               "n_best_epoch_over_900": sum(e > 900 for e in best_epochs)}
        if all("train_objective_last50" in r for r in runs):
            objective = [r["train_objective_last50"] for r in runs]
            row["mean_train_objective_last50"] = np.mean(objective)
            row["std_train_objective_last50"] = np.std(objective, ddof=1)
        if algorithm != "sgd":
            row["mean_final_W1_norm"] = np.mean([r["final_W1_norm"] for r in runs])
            row["mean_gbest_updates"] = np.mean([r["mean_gbest_updates"] for r in runs])
        rows.append(row)
    return rows


def best_row(rows, criterion, **match):
    candidates = [row for row in rows if all(row[key] == value for key, value in match.items())]
    return min(candidates, key=lambda row: np.inf if np.isnan(row[criterion]) else row[criterion])


def lam_jobs():
    return [(dataset, "sgd", STAGE1_N_BATCHES, seed, {"lam": lam, "lr": lr})
            for dataset in DATASETS for lam in [0.0] + LAM_GRID for lr in LR_GRID for seed in TUNING_SEEDS]


def lr_jobs(lams, lrs):
    return [(dataset, "sgd", n_batches, seed, {"lam": lams[dataset], "lr": lr})
            for (dataset, n_batches), grid in lrs.items() for lr in grid for seed in TUNING_SEEDS]


def pso_configs():
    return [(dataset, algorithm, n_batches) for dataset in DATASETS for n_batches in N_BATCHES
            for algorithm in ["pso", "pso_reeval", "qpso"] if not (algorithm == "pso_reeval" and n_batches == 1)]


def pso_jobs(lams, v_maxes):
    jobs = []
    for (dataset, algorithm, n_batches), grid in v_maxes.items():
        for v_max in grid:
            base = {"lam": lams[dataset], "v_max": v_max}
            if algorithm == "qpso":
                settings = [dict(base, r_cloud=r_cloud, n_quantum=n_quantum)
                            for r_cloud in R_CLOUD_GRID for n_quantum in N_QUANTUM_GRID]
            else:
                settings = [base]
            jobs += [(dataset, algorithm, n_batches, seed, s) for s in settings for seed in TUNING_SEEDS]
    return jobs


def tune_lambda():
    rows = summarise(run_many(lam_jobs(), TUNING_RUNS_DIR))
    lams = {dataset: best_row([row for row in rows if row["lam"] > 0], VAL_CRITERION, dataset=dataset)["lam"]
            for dataset in DATASETS}
    return lams, rows


def tune_lr(lams):
    lrs = {(dataset, n_batches): list(LR_GRID) for dataset in DATASETS for n_batches in N_BATCHES}
    pending = list(lrs)
    rows, extensions, capped = [], [], []
    while pending:
        round_rows = summarise(run_many(lr_jobs(lams, {key: lrs[key] for key in pending}), OBJECTIVE_RUNS_DIR))
        still_pending = []
        for dataset, n_batches in pending:
            grid = sorted(lrs[dataset, n_batches])
            best_lr = best_row(round_rows, OBJECTIVE_CRITERION, dataset=dataset, n_batches=n_batches)["lr"]
            new_lr = None
            if best_lr == grid[0]:
                new_lr = float(f"{best_lr / 3:.3g}")
            elif best_lr == grid[-1]:
                new_lr = float(f"{best_lr * 3:.3g}")
            if new_lr is not None and new_lr > LR_MAX:
                capped.append((dataset, n_batches, best_lr))
                new_lr = None
            if new_lr is None:
                rows += [row for row in round_rows if row["dataset"] == dataset and row["n_batches"] == n_batches]
            else:
                lrs[dataset, n_batches].append(new_lr)
                extensions.append((dataset, n_batches, best_lr, new_lr))
                still_pending.append((dataset, n_batches))
        pending = still_pending
    return rows, extensions, capped


def tune_pso(lams):
    v_maxes = {config: list(V_MAX_GRID) for config in pso_configs()}
    pending = list(v_maxes)
    rows, extensions, capped = [], [], []
    while pending:
        round_rows = summarise(run_many(pso_jobs(lams, {config: v_maxes[config] for config in pending}),
                                        OBJECTIVE_RUNS_DIR))
        still_pending = []
        for config in pending:
            dataset, algorithm, n_batches = config
            smallest = min(v for v in v_maxes[config] if v is not None)
            best_v_max = best_row(round_rows, OBJECTIVE_CRITERION, dataset=dataset, algorithm=algorithm,
                                  n_batches=n_batches)["v_max"]
            new_v_max = float(f"{smallest / 3:.1g}") if best_v_max == smallest else None
            if new_v_max is not None and new_v_max < V_MAX_MIN:
                capped.append((config, best_v_max))
                new_v_max = None
            if new_v_max is None:
                rows += [row for row in round_rows
                         if (row["dataset"], row["algorithm"], row["n_batches"]) == config]
            else:
                v_maxes[config].append(new_v_max)
                extensions.append((config, best_v_max, new_v_max))
                still_pending.append(config)
        pending = still_pending
    rows += [dict(row, algorithm="pso_reeval", note="copied from pso")
             for row in rows if row["algorithm"] == "pso" and row["n_batches"] == 1]
    return rows, extensions, capped


def lower_edges(rows, pick):
    config = (pick["dataset"], pick["algorithm"], pick["n_batches"])
    same = [r for r in rows if (r["dataset"], r["algorithm"], r["n_batches"]) == config]
    edges = []
    for param, extensions in [("v_max", V_MAX_EXTENSIONS), ("r_cloud", R_CLOUD_EXTENSIONS)]:
        if pick["settings"].get(param) is None:
            continue
        tried = [r["settings"][param] for r in same if r["settings"].get(param) is not None]
        if pick["settings"][param] == min(tried):
            smaller = [v for v in extensions if v < pick["settings"][param]]
            edges.append((param, max(smaller) if smaller else None))
    return edges


def close_edges(lams, rows):
    log = []
    for round_number in range(1, EDGE_ROUNDS + 1):
        jobs = []
        for dataset, algorithm, n_batches in pso_configs():
            pick = best_row(rows, OBJECTIVE_CRITERION, dataset=dataset, algorithm=algorithm, n_batches=n_batches)
            for param, new_value in lower_edges(rows, pick):
                if new_value is None:
                    continue
                settings = dict(pick["settings"], **{param: new_value})
                log.append((round_number, (dataset, algorithm, n_batches), param, pick["settings"][param], new_value))
                jobs += [(dataset, algorithm, n_batches, seed, settings) for seed in TUNING_SEEDS]
        if not jobs:
            break
        new_rows = summarise(run_many(jobs, OBJECTIVE_RUNS_DIR))
        new_rows += [dict(row, algorithm="pso_reeval", note="copied from pso")
                     for row in new_rows if row["algorithm"] == "pso" and row["n_batches"] == 1]
        rows = rows + new_rows

    remaining = []
    for dataset, algorithm, n_batches in pso_configs():
        pick = best_row(rows, OBJECTIVE_CRITERION, dataset=dataset, algorithm=algorithm, n_batches=n_batches)
        remaining += [((dataset, algorithm, n_batches), param, pick["settings"][param], new_value)
                      for param, new_value in lower_edges(rows, pick)]
    return rows, log, remaining


def convergence(chosen):
    (a1, b1), (a2, b2) = CONVERGENCE_WINDOWS
    rows = []
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            for n_batches in N_BATCHES:
                settings = chosen[dataset][algorithm][str(n_batches)]
                run_algorithm = "pso" if algorithm == "pso_reeval" and n_batches == 1 else algorithm
                early, late, changes, best_epochs = [], [], [], []
                for seed in TUNING_SEEDS:
                    path = OBJECTIVE_RUNS_DIR / f"{run_name(dataset, run_algorithm, n_batches, seed, settings)}.json"
                    s = json.loads(path.read_text())
                    early.append(np.mean(s["train_error"][a1 - 1:b1]))
                    late.append(np.mean(s["train_error"][a2 - 1:b2]))
                    changes.append((late[-1] - early[-1]) / early[-1])
                    best_epochs.append(s["best_epoch"])
                change = np.mean(changes)
                rows.append({"dataset": dataset, "algorithm": algorithm, "n_batches": n_batches,
                             "settings": describe(settings),
                             f"mean_objective_{a1}_{b1}": np.mean(early), f"mean_objective_{a2}_{b2}": np.mean(late),
                             "relative_change": change, "status": convergence_status(change),
                             "mean_best_epoch": np.mean(best_epochs)})
    return pd.DataFrame(rows)


def convergence_status(change):
    if change <= -CONVERGENCE_TOL:
        return "still improving"
    if change > CONVERGENCE_TOL:
        return "late drift"
    return "converged"


def projected_hours(jobs, runs_dir, per_epoch):
    todo = [job for job in jobs if not (runs_dir / f"{run_name(*job)}.json").exists()]
    seconds = sum(per_epoch[dataset, algorithm, n_batches] * N_EPOCHS for dataset, algorithm, n_batches, _, _ in todo)
    return seconds / 3600, len(todo)


def sort_rows(rows, extra=()):
    order = {"dataset": DATASETS.index, "algorithm": ALGORITHMS.index}

    def key(col):
        if col.name == "n_batches":
            return -col
        return col.map(order[col.name]) if col.name in order else col

    df = pd.DataFrame(rows).drop(columns="settings")
    df = df.sort_values(["dataset", "n_batches", "algorithm", *extra], key=key, kind="stable")
    first = ["dataset", "algorithm", "n_batches"] + [c for c in ["lam", "lr", "v_max", "r_cloud", "n_quantum"] if c in df]
    df = df[first + [c for c in df if c not in first]]
    if "v_max" in df:
        df["v_max"] = df["v_max"].astype(object).where(df["v_max"].notna(), "none")
    return df


def describe(settings):
    return ", ".join(f"{key}={'none' if value is None else f'{value:g}'}"
                     for key, value in settings.items() if key != "lam")


def tune():
    n_cpus = os.cpu_count()
    per_epoch = seconds_per_epoch(with_test=False)

    def print_projection(what, jobs, runs_dir):
        hours, n_todo = projected_hours(jobs, runs_dir, per_epoch)
        print(f"projected {what}: {n_todo} of {len(jobs)} runs still to do, {hours:.2f} h serial, "
              f"{hours / n_cpus:.2f} h on {n_cpus} cpus (ideal speed-up)")

    start = time.perf_counter()
    print_projection("stage 1", lam_jobs(), TUNING_RUNS_DIR)
    print("stage 1: weight decay")
    lams, lam_rows = tune_lambda()
    base_lrs = {(dataset, n_batches): list(LR_GRID) for dataset in DATASETS for n_batches in N_BATCHES}
    base_v_maxes = {config: list(V_MAX_GRID) for config in pso_configs()}
    print_projection("stages 2 and 3 (before any extensions)",
                     lr_jobs(lams, base_lrs) + pso_jobs(lams, base_v_maxes), OBJECTIVE_RUNS_DIR)
    print("stage 2: sgd learning rate")
    lr_rows, extensions, capped = tune_lr(lams)
    print("stage 3: pso variants")
    pso_rows, v_extensions, _ = tune_pso(lams)
    print("stage 3 edges: extending v_max and r_cloud one at a time")
    pso_rows, edge_log, edge_remaining = close_edges(lams, pso_rows)
    print(f"tuning finished in {(time.perf_counter() - start) / 60:.1f} min\n")

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    sort_rows(lam_rows, extra=["lam", "lr"]).to_csv(TUNING_DIR / "stage1_lambda.csv", index=False, float_format="%.6g")
    sort_rows(lr_rows, extra=["lr"]).to_csv(TUNING_DIR / "stage2_sgd_lr.csv", index=False, float_format="%.6g")
    sort_rows(pso_rows).to_csv(TUNING_DIR / "stage3_pso.csv", index=False, float_format="%.6g")

    robustness = sort_rows([row for row in lam_rows if row["lr"] == ROBUSTNESS_LR], extra=["lam"])
    robustness = robustness[["dataset", "lam", "lr", "mean_best_val_ce", "std_best_val_ce", "mean_best_epoch"]]
    robustness["chosen"] = [lam == lams[dataset] for dataset, lam in zip(robustness["dataset"], robustness["lam"])]
    robustness.to_csv(TUNING_DIR / "stage1_robustness.csv", index=False, float_format="%.6g")

    chosen, chosen_rows = {}, []
    for dataset in DATASETS:
        chosen[dataset] = {"lam": lams[dataset]}
        for algorithm in ALGORITHMS:
            chosen[dataset][algorithm] = {}
            for n_batches in N_BATCHES:
                rows = lr_rows if algorithm == "sgd" else pso_rows
                row = best_row(rows, OBJECTIVE_CRITERION, dataset=dataset, algorithm=algorithm, n_batches=n_batches)
                chosen[dataset][algorithm][str(n_batches)] = row["settings"]
                chosen_rows.append(row)
    old = json.loads(CHOSEN.read_text()) if CHOSEN.exists() else chosen
    CHOSEN.write_text(json.dumps(chosen, indent=2))
    convergence_table = convergence(chosen)
    convergence_table.to_csv(TUNING_DIR / "convergence.csv", index=False, float_format="%.6g")

    def compare(dataset, algorithm, n_batches, text):
        new, previous = chosen[dataset][algorithm][str(n_batches)], old[dataset][algorithm][str(n_batches)]
        if {k: v for k, v in new.items() if k != "lam"} == {k: v for k, v in previous.items() if k != "lam"}:
            return text
        return f"{text} * (was {describe(previous)})" if algorithm != "sgd" else f"{text} * (was {previous['lr']:g})"

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print("chosen lam (stage 1, best validation cross-entropy): "
          + ", ".join(f"{dataset} {lams[dataset]:g}" + ("" if lams[dataset] == old[dataset]["lam"]
                                                           else f" * (was {old[dataset]['lam']:g})")
                      for dataset in DATASETS))
    print(f"\nstage 1 robustness: best validation cross-entropy of every lam at lr = {ROBUSTNESS_LR}")
    print(robustness.to_string(index=False))

    wine = pd.DataFrame({n_batches: {row["lr"]: f"{row[OBJECTIVE_CRITERION]:.4g}" for row in lr_rows
                                     if row["dataset"] == "wine" and row["n_batches"] == n_batches}
                         for n_batches in N_BATCHES}).sort_index().fillna("-")
    print("\nwine: mean training objective over the last 50 epochs (rows lr, columns n_batches)")
    print(wine.to_string())

    lr_table = pd.DataFrame({n_batches: {dataset: compare(dataset, "sgd", n_batches,
                                                          f"{chosen[dataset]['sgd'][str(n_batches)]['lr']:g}")
                                         for dataset in DATASETS} for n_batches in N_BATCHES})
    print("\nchosen sgd lr (columns are n_batches; * = changed from the previous chosen.json)")
    print(lr_table.to_string())

    print("\nchosen pso settings (* = changed from the previous chosen.json)")
    for dataset in DATASETS:
        for n_batches in N_BATCHES:
            for algorithm in ["pso", "pso_reeval", "qpso"]:
                text = describe(chosen[dataset][algorithm][str(n_batches)])
                print(f"  {dataset:5s} {n_batches:2d} {algorithm:10s} {compare(dataset, algorithm, n_batches, text)}")

    print("\nchosen settings on a grid edge")
    for dataset in DATASETS:
        if lams[dataset] in (LAM_GRID[0], LAM_GRID[-1]):
            print(f"  stage 1: {dataset} lam = {lams[dataset]:g}")
    for dataset, n_batches, best_lr in capped:
        print(f"  stage 2: {dataset} n_batches {n_batches}: best lr {best_lr:g} is at the top, "
              f"and extending would pass lr = {LR_MAX}, so the extension was stopped")
    for dataset, n_batches, best_lr, new_lr in extensions:
        print(f"  stage 2 (extended): {dataset} n_batches {n_batches}: best lr {best_lr:g} was at an end, added {new_lr:g}")
    for (dataset, algorithm, n_batches), best_v_max, new_v_max in v_extensions:
        print(f"  stage 3 (extended): {dataset} {algorithm} n_batches {n_batches}: best v_max {best_v_max:g} "
              f"was the smallest tried, added {new_v_max:g}")
    for round_number, (dataset, algorithm, n_batches), param, old_value, new_value in edge_log:
        print(f"  edge rule round {round_number}: {dataset} {algorithm} n_batches {n_batches}: chosen {param} "
              f"{old_value:g} was on the lower edge, tried {new_value:g}")
    for (dataset, algorithm, n_batches), param, value, new_value in edge_remaining:
        reason = "no smaller value allowed" if new_value is None else f"out of rounds ({EDGE_ROUNDS})"
        print(f"  STILL ON AN EDGE: {dataset} {algorithm} n_batches {n_batches}: {param} = {value:g} ({reason})")
    for row in chosen_rows:
        if row["settings"].get("r_cloud") == max(R_CLOUD_GRID):
            print(f"  stage 3: {row['dataset']} qpso n_batches {row['n_batches']}: r_cloud = {max(R_CLOUD_GRID):g} (top)")
    n_quantum_counts = pd.Series([row["settings"]["n_quantum"] for row in chosen_rows if row["algorithm"] == "qpso"])
    print(f"  stage 3: qpso n_quantum chosen: {n_quantum_counts.value_counts().sort_index().to_dict()}")

    v_max_counts = pd.Series(["none" if row["settings"]["v_max"] is None else f"{row['settings']['v_max']:g}"
                              for row in chosen_rows if row["algorithm"] != "sgd"]).value_counts().to_dict()
    r_cloud_counts = pd.Series([row["settings"]["r_cloud"] for row in chosen_rows
                                if row["algorithm"] == "qpso"]).value_counts().sort_index().to_dict()
    print(f"\nchosen v_max counts: {v_max_counts}; chosen r_cloud counts: {r_cloud_counts}")

    changed = [(dataset, algorithm, n_batches) for dataset in DATASETS for algorithm in ALGORITHMS
               for n_batches in N_BATCHES
               if chosen[dataset][algorithm][str(n_batches)] != old[dataset][algorithm][str(n_batches)]]
    print(f"\npicks changed from the previous chosen.json: {len(changed)}")
    for dataset, algorithm, n_batches in changed:
        print(f"  {dataset:5s} {algorithm:10s} n_batches {n_batches:2d}: "
              f"{describe(old[dataset][algorithm][str(n_batches)])} -> {describe(chosen[dataset][algorithm][str(n_batches)])}")

    (a1, b1), (a2, b2) = CONVERGENCE_WINDOWS
    failed = convergence_table[convergence_table["status"] != "converged"]
    print(f"\nconvergence: relative change of the training objective from epochs {a1}-{b1} to {a2}-{b2} "
          f"(converged if it improved by less than {CONVERGENCE_TOL:.0%}); "
          f"{len(failed)} of {len(convergence_table)} chosen configurations fail")
    for row in failed.itertuples():
        print(f"  {row.dataset:5s} {row.algorithm:10s} n_batches {row.n_batches:2d} ({row.settings}): {row.status}, "
              f"relative change {row.relative_change:+.2%}, mean best epoch {row.mean_best_epoch:.0f}")

    print("\nweight growth for the chosen settings at n_batches = 32 and 16")
    for row in chosen_rows:
        if row["n_batches"] in (32, 16) and row["algorithm"] in ["qpso", "pso_reeval"]:
            print(f"  {row['dataset']:5s} {row['algorithm']:10s} n_batches {row['n_batches']:2d} ({describe(row['settings'])}):"
                  f" mean final W1 norm {row['mean_final_W1_norm']:.1f}, mean gbest updates "
                  f"{row['mean_gbest_updates']:.1f} of {row['n_batches']} per epoch")


def main_jobs(chosen, datasets, n_epochs):
    return [(dataset, algorithm, n_batches, seed, chosen[dataset][algorithm][str(n_batches)], n_epochs)
            for dataset in datasets for n_batches in N_BATCHES for algorithm in ALGORITHMS
            if not (algorithm == "pso_reeval" and n_batches == 1) for seed in MAIN_SEEDS]


def count_nonfinite(value):
    if isinstance(value, dict):
        return sum(count_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_nonfinite(v) for v in value)
    if isinstance(value, float):
        return int(not np.isfinite(value))
    return 0


def main_row(record):
    h = record["history"]
    row = {key: record[key] for key in ["dataset", "algorithm", "n_batches", "seed", "n_epochs"]}
    for key in ["lam", "lr", "v_max", "r_cloud", "n_quantum"]:
        if key in record["settings"]:
            value = record["settings"][key]
            row[key] = "none" if value is None else value
    row["best_epoch"] = record["best_epoch"]
    for when in ["best", "final"]:
        for split, metrics in record[when].items():
            for metric, value in metrics.items():
                row[f"{when}_{split}_{metric}"] = value
    row["best_train_error"] = h["train_error"][record["best_epoch"] - 1]
    row["final_train_error"] = h["train_error"][-1]
    row["n_evals"] = record["n_evals"]
    row["time_seconds"] = record["time_seconds"]
    row["batch_checksum"] = record["batch_checksum"]
    row["n_nonfinite"] = count_nonfinite(record)
    return row


def main_run(dataset, algorithm, n_batches, seed, settings, n_epochs, out_dir):
    path = out_dir / f"{run_name(dataset, algorithm, n_batches, seed, settings)}.json.gz"
    if path.exists():
        return main_row(json.loads(gzip.decompress(path.read_bytes())))

    r = run_one(dataset, algorithm, n_batches, seed, n_epochs, **settings)
    record = {
        "dataset": dataset,
        "algorithm": algorithm,
        "n_batches": n_batches,
        "seed": seed,
        "n_epochs": n_epochs,
        "settings": settings,
        "history": r["history"],
        "best_epoch": r["best_epoch"],
        "best": r["best"],
        "final": metrics_at(r["history"], n_epochs),
        "n_evals": r["n_evals"],
        "time_seconds": r["time_seconds"],
        "batch_checksum": r["batch_checksum"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(record).encode()))
    tmp.replace(path)
    return main_row(record)


def check_main(df, jobs, out_dir, name):
    copied = df["note"].eq("copied from pso").sum() if "note" in df else 0
    files = list(out_dir.glob("*.json.gz"))
    print(f"  runs finished: {len(df) - copied} of {len(jobs)} expected ({len(files)} files in {out_dir.name}/), "
          f"plus {copied} pso_reeval rows at n_batches 1 copied from pso")
    print(f"  nan/inf values: {df['n_nonfinite'].sum()} (in {(df['n_nonfinite'] > 0).sum()} runs)")
    groups = df.groupby(["dataset", "n_batches", "seed"])
    bad = [key for key, g in groups if g["batch_checksum"].nunique() != 1 or set(g["algorithm"]) != set(ALGORITHMS)]
    print(f"  batch checksums identical across all four algorithms: {groups.ngroups - len(bad)} of {groups.ngroups} "
          f"(dataset, n_batches, seed) groups")
    for key in bad:
        print(f"    MISMATCH: {key}")
    print(f"  disk: {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")


def main_experiment():
    chosen = json.loads(CHOSEN.read_text())
    n_cpus = os.cpu_count()
    parts = [("main", main_jobs(chosen, DATASETS, N_EPOCHS), MAIN_DIR, MAIN_CSV),
             (f"glass{GLASS_LONG_EPOCHS}", main_jobs(chosen, ["glass"], GLASS_LONG_EPOCHS), GLASS_LONG_DIR, GLASS_LONG_CSV)]

    per_epoch = seconds_per_epoch()
    n_todo = {}
    for name, jobs, out_dir, _ in parts:
        todo = [job for job in jobs if not (out_dir / f"{run_name(*job[:5])}.json.gz").exists()]
        hours = sum(per_epoch[dataset, algorithm, n_batches] * n_epochs
                    for dataset, algorithm, n_batches, _, _, n_epochs in todo) / 3600
        n_todo[name] = len(todo)
        print(f"projected {name}: {len(todo)} of {len(jobs)} runs still to do, {hours:.2f} h serial, "
              f"{hours / n_cpus:.2f} h on {n_cpus} cpus (ideal speed-up)")

    order = {"dataset": DATASETS.index, "algorithm": ALGORITHMS.index}
    for name, jobs, out_dir, csv_path in parts:
        start = time.perf_counter()
        rows = Parallel(n_jobs=N_JOBS)(delayed(main_run)(*job, out_dir) for job in jobs)
        rows += [dict(row, algorithm="pso_reeval", note="copied from pso")
                 for row in rows if row["algorithm"] == "pso" and row["n_batches"] == 1]
        df = pd.DataFrame(rows).sort_values(
            ["dataset", "n_batches", "algorithm", "seed"], kind="stable",
            key=lambda col: col.map(order[col.name]) if col.name in order else (-col if col.name == "n_batches" else col))
        first = [c for c in ["dataset", "algorithm", "n_batches", "seed", "n_epochs",
                             "lam", "lr", "v_max", "r_cloud", "n_quantum"] if c in df]
        df = df[first + [c for c in df if c not in first]]
        df.to_csv(csv_path, index=False)
        print(f"\n{name}: finished in {(time.perf_counter() - start) / 60:.1f} min "
              f"({n_todo[name]} runs executed now, {len(jobs) - n_todo[name]} loaded from earlier)")
        check_main(df, jobs, out_dir, name)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pilot":
        pilot()
    elif mode == "tune":
        tune()
    elif mode == "main":
        main_experiment()
    else:
        print("usage: python experiments.py pilot|tune|main")
