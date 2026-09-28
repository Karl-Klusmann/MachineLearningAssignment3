import gzip
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import pandas as pd
from scipy import stats

from data import load_dataset
from experiments import (CHOSEN, COLORS, DATASETS, GLASS_LONG_CSV, LABELS, MAIN_CSV, MAIN_DIR, MAIN_SEEDS,
                         N_BATCHES, N_EPOCHS, RESULTS_DIR, run_name)

ALPHA = 0.05
MIN_BETTER = 8
MINI_BATCHES = [32, 16, 8, 4, 2]
BEHAVIOUR_N_BATCHES = [32, 1]
CLAMP_EPOCHS = 100

ORDER = ["sgd", "pso", "pso_reeval", "qpso"]
COMPARISONS = {"C1": ("qpso", "pso"), "C2": ("pso_reeval", "pso"), "C3": ("qpso", "pso_reeval"), "C4": ("qpso", "sgd")}
EQUAL_COST_COMPARISONS = {c: COMPARISONS[c] for c in ["C1", "C2", "C3"]}
FULL_BATCH_COMPARISONS = {c: COMPARISONS[c] for c in ["C1", "C4"]}
MEASURES = {"test_ce": ("best_test_ce", True), "test_f1": ("best_test_f1", False),
            "train_objective": ("final_train_error", True)}
OUTCOMES = ["better", "worse", "no difference"]

ANALYSIS_DIR = RESULTS_DIR / "analysis"
EXPLORATORY_DIR = ANALYSIS_DIR / "exploratory"
FIGURES_DIR = Path(__file__).parent / "figures"
DISPLAY = {"wine": "Wine", "wdbc": "WDBC", "glass": "Glass"}
MARKERS = {"sgd": "s", "pso": "o", "pso_reeval": "^", "qpso": "D"}
FIG_SIZE = (3.5, 2.3)
PLOT_STYLE = {"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8,
              "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
              "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42}


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def wilcoxon_p(diff):
    if np.all(diff == 0):
        return 1.0
    return float(stats.wilcoxon(diff).pvalue)


def outcome(diff, p_holm, lower_is_better):
    if not p_holm < ALPHA:
        return "no difference"
    centre = np.median(diff)
    if centre == 0:
        centre = np.mean(diff)
    if centre == 0:
        return "no difference"
    return "better" if (centre < 0) == lower_is_better else "worse"


def compare(values, comparisons, lower_is_better):
    algorithms = [a for a in ORDER if any(a in pair for pair in comparisons.values())]
    friedman_p = float(stats.friedmanchisquare(*[values[a] for a in algorithms]).pvalue)
    diffs = {c: values[a] - values[b] for c, (a, b) in comparisons.items()}
    if friedman_p < ALPHA:
        p_raw = [wilcoxon_p(d) for d in diffs.values()]
        p_holm = holm(p_raw)
    else:
        p_raw = p_holm = [np.nan] * len(diffs)
    return [{"comparison": c, "first": a, "second": b, "friedman_p": friedman_p, "p_raw": p, "p_holm": p_adj,
             "outcome": outcome(diffs[c], p_adj, lower_is_better), "median_diff": float(np.median(diffs[c]))}
            for (c, (a, b)), p, p_adj in zip(comparisons.items(), p_raw, p_holm)]


def paired(runs, dataset, n_batches, column):
    config = runs[(runs["dataset"] == dataset) & (runs["n_batches"] == n_batches)]
    return {algorithm: g.sort_values("seed")[column].to_numpy() for algorithm, g in config.groupby("algorithm")}


def load_record(runs_dir, chosen, dataset, algorithm, n_batches, seed):
    settings = chosen[dataset][algorithm][str(n_batches)]
    path = runs_dir / f"{run_name(dataset, algorithm, n_batches, seed, settings)}.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def equal_cost_epoch(n_evals, val_ce, budget):
    n_window = int(np.searchsorted(n_evals, budget, side="right"))
    return int(np.argmin(val_ce[:n_window])) + 1


def equal_cost_runs(main, chosen):
    budgets = main[main["algorithm"] == "pso"].set_index(["dataset", "n_batches", "seed"])["n_evals"]
    rows = []
    for dataset in DATASETS:
        for n_batches in MINI_BATCHES:
            for seed in MAIN_SEEDS:
                budget = int(budgets[dataset, n_batches, seed])
                pso = main[(main["dataset"] == dataset) & (main["n_batches"] == n_batches)
                           & (main["algorithm"] == "pso") & (main["seed"] == seed)].iloc[0]
                rows.append({"dataset": dataset, "n_batches": n_batches, "algorithm": "pso", "seed": seed,
                             "budget": budget, "window_epochs": N_EPOCHS, "epoch": pso["best_epoch"],
                             "n_evals_at_epoch": np.nan, "best_test_ce": pso["best_test_ce"]})
                for algorithm in ["pso_reeval", "qpso"]:
                    h = load_record(MAIN_DIR, chosen, dataset, algorithm, n_batches, seed)["history"]
                    epoch = equal_cost_epoch(h["n_evals"], h["val_ce"], budget)
                    rows.append({"dataset": dataset, "n_batches": n_batches, "algorithm": algorithm, "seed": seed,
                                 "budget": budget,
                                 "window_epochs": int(np.searchsorted(h["n_evals"], budget, side="right")),
                                 "epoch": epoch, "n_evals_at_epoch": h["n_evals"][epoch - 1],
                                 "best_test_ce": h["test_ce"][epoch - 1]})
    return pd.DataFrame(rows)


def all_comparisons(main, equal_cost, glass_long):
    checks = [("primary", main, "test_ce", DATASETS, COMPARISONS),
              ("train_objective", main, "train_objective", DATASETS, COMPARISONS),
              ("a_test_f1", main, "test_f1", DATASETS, COMPARISONS),
              ("b_equal_cost", equal_cost, "test_ce", DATASETS, EQUAL_COST_COMPARISONS),
              ("c_glass2000", glass_long, "test_ce", ["glass"], COMPARISONS)]
    rows = []
    for check, runs, measure, datasets, comparisons in checks:
        column, lower_is_better = MEASURES[measure]
        for dataset in datasets:
            for n_batches in MINI_BATCHES:
                for row in compare(paired(runs, dataset, n_batches, column), comparisons, lower_is_better):
                    rows.append({"check": check, "measure": measure, "dataset": dataset, "n_batches": n_batches, **row})
    for measure, (column, lower_is_better) in MEASURES.items():
        for dataset in DATASETS:
            for row in compare(paired(main, dataset, 1, column), FULL_BATCH_COMPARISONS, lower_is_better):
                rows.append({"check": "full_batch", "measure": measure, "dataset": dataset, "n_batches": 1, **row})
    return pd.DataFrame(rows)


def count_outcomes(comparisons):
    rows = []
    for (check, measure), g in comparisons.groupby(["check", "measure"], sort=False):
        for dataset in ["all", *g["dataset"].unique()]:
            part = g if dataset == "all" else g[g["dataset"] == dataset]
            for c, s in part.groupby("comparison"):
                counts = s["outcome"].value_counts()
                rows.append({"check": check, "measure": measure, "dataset": dataset, "comparison": c,
                             "first": s["first"].iloc[0], "second": s["second"].iloc[0], "n_configs": len(s),
                             **{o.replace(" ", "_"): int(counts.get(o, 0)) for o in OUTCOMES}})
    return pd.DataFrame(rows)


def n_better(part, comparison):
    return int((part.loc[part["comparison"] == comparison, "outcome"] == "better").sum())


def rq2_rho(part):
    rho = {}
    for dataset in part["dataset"].unique():
        c1 = part[(part["dataset"] == dataset) & (part["comparison"] == "C1")].set_index("n_batches")["median_diff"]
        rho[dataset] = float(stats.spearmanr(MINI_BATCHES, -c1[MINI_BATCHES].to_numpy()).statistic)
    return rho


def main_mechanism(part):
    c2, c3 = n_better(part, "C2"), n_better(part, "C3")
    if c2 > c3:
        return "re-evaluation (C2)"
    if c3 > c2:
        return "quantum particles (C3)"
    return "tie (neither mechanism has more 'better' outcomes)"


def verdicts(comparisons):
    primary = comparisons[comparisons["check"] == "primary"]
    better = {c: n_better(primary, c) for c in COMPARISONS}
    rq1, rq4 = better["C1"] >= MIN_BETTER, better["C4"] >= MIN_BETTER
    rho = rq2_rho(primary)
    rq2 = all(r > 0 for r in rho.values())
    hypothesis = "supported" if rq1 and rq4 else "partially supported" if rq1 else "not supported"
    result = {
        "procedure": "docs/analysis_procedure.md",
        "n_configurations": int(primary.groupby(["dataset", "n_batches"]).ngroups),
        "better_counts": better,
        "RQ1": {"verdict": "yes" if rq1 else "no", "rule": f"C1 (QPSO vs static PSO) 'better' in >= {MIN_BETTER} of 15",
                "C1_better": better["C1"]},
        "RQ2": {"verdict": "supported" if rq2 else "not supported",
                "rule": "Spearman rho(n_batches, median of static PSO minus QPSO test CE) > 0 on all three problems",
                "rho": rho},
        "RQ3": {"verdict": main_mechanism(primary),
                "rule": "larger 'better' count of C2 (re-evaluation) and C3 (quantum particles)",
                "C2_better": better["C2"], "C3_better": better["C3"]},
        "RQ4": {"verdict": "yes" if rq4 else "no", "rule": f"C4 (QPSO vs SGD) 'better' in >= {MIN_BETTER} of 15",
                "C4_better": better["C4"]},
        "hypothesis": {"verdict": hypothesis,
                       "rule": "supported if RQ1 and RQ4 are yes; partially supported if only RQ1; "
                               "not supported if RQ1 is no"},
        "checks": {},
    }
    for check in ["train_objective", "a_test_f1", "b_equal_cost", "c_glass2000"]:
        part = comparisons[comparisons["check"] == check]
        summary = {"n_configurations": int(part.groupby(["dataset", "n_batches"]).ngroups),
                   "better_counts": {c: n_better(part, c) for c in part["comparison"].unique()},
                   "RQ3_mechanism": main_mechanism(part)}
        if summary["n_configurations"] == 15:
            summary["RQ1_rule"] = "yes" if n_better(part, "C1") >= MIN_BETTER else "no"
            if "C4" in summary["better_counts"]:
                summary["RQ4_rule"] = "yes" if n_better(part, "C4") >= MIN_BETTER else "no"
        if check in ["b_equal_cost", "c_glass2000"]:
            summary["RQ2_rho"] = rq2_rho(part)
        result["checks"][check] = summary
    return result


def ordered(df):
    key = {"dataset": DATASETS.index, "algorithm": ORDER.index}
    columns = [c for c in ["n_epochs", "dataset", "n_batches", "algorithm"] if c in df]
    return df.sort_values(columns, kind="stable", key=lambda col: col.map(key[col.name]) if col.name in key
                          else (-col if col.name == "n_batches" else col)).reset_index(drop=True)


def summary_table(runs, column):
    g = runs.groupby(["n_epochs", "dataset", "n_batches", "algorithm"])[column]
    table = g.agg(["mean", "std", "median"]).reset_index()
    table["q1"] = g.quantile(0.25).to_numpy()
    table["q3"] = g.quantile(0.75).to_numpy()
    table["n_runs"] = g.size().to_numpy()
    return ordered(table)


def behaviour(chosen):
    rows = []
    for dataset in DATASETS:
        for n_batches in BEHAVIOUR_N_BATCHES:
            for algorithm in ["pso", "pso_reeval", "qpso"]:
                if algorithm == "pso_reeval" and n_batches == 1:
                    continue
                per_run = []
                for seed in MAIN_SEEDS:
                    h = load_record(MAIN_DIR, chosen, dataset, algorithm, n_batches, seed)["history"]
                    per_run.append({"gbest_changes_per_epoch": np.mean(h["gbest_updates"]),
                                    "actual_over_stored": h["train_error"][-1] / h["gbest_err_stored"][-1],
                                    "clamp_fraction_last100": np.mean(h["clamp_fraction"][-CLAMP_EPOCHS:]),
                                    "final_diversity": h["diversity"][-1],
                                    "final_W1_norm": h["W1_norm"][-1]})
                rows.append({"dataset": dataset, "n_batches": n_batches, "algorithm": algorithm,
                             **pd.DataFrame(per_run).median().to_dict()})
    return pd.DataFrame(rows)


def write_table(path, caption, label, columns, head, body):
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule", *head, r"\midrule"]
    lines += [r"\midrule" if row is None else " & ".join(row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def latex_summary(table, path, measure_name, label):
    table = table.set_index(["dataset", "n_batches", "algorithm"])
    head = [" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{LABELS[a]}}}" for a in ORDER) + r" \\",
            " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(ORDER))),
            "Problem & Batches & " + " & ".join([r"Mean $\pm$ std & Median"] * len(ORDER)) + r" \\"]
    body = []
    for dataset in DATASETS:
        if body:
            body.append(None)
        for n_batches in N_BATCHES:
            row = [DISPLAY[dataset] if n_batches == N_BATCHES[0] else "", str(n_batches)]
            for algorithm in ORDER:
                s = table.loc[(dataset, n_batches, algorithm)]
                row += [rf"${s['mean']:.3f} \pm {s['std']:.3f}$", f"{s['median']:.3f}"]
            body.append(row)
    caption = (f"Test {measure_name} at the best-validation checkpoint over 30 seeds. "
               "At one batch, PSO + re-evaluation is identical to static PSO.")
    write_table(path, caption, label, "lr" + "cr" * len(ORDER), head, body)


def latex_behaviour(table, path):
    names = ["gbest changes / epoch", "Actual / stored", "Clamp fraction", "Diversity", r"$\|W_1\|$"]
    head = ["Problem & Batches & Algorithm & " + " & ".join(names) + r" \\"]
    body = []
    for dataset in DATASETS:
        if body:
            body.append(None)
        part = table[table["dataset"] == dataset]
        for i, (_, r) in enumerate(part.iterrows()):
            row = [DISPLAY[dataset] if i == 0 else "", str(r["n_batches"]), LABELS[r["algorithm"]]]
            row += [f"{r[c]:.3g}" for c in ["gbest_changes_per_epoch", "actual_over_stored", "clamp_fraction_last100",
                                            "final_diversity", "final_W1_norm"]]
            body.append(row)
    caption = ("Behaviour of the PSO variants (medians over 30 seeds): gbest changes per epoch, ratio of the actual "
               "to the stored gbest objective at the final epoch, clamp fraction over the last 100 epochs, "
               "and final diversity and $\\|W_1\\|$.")
    write_table(path, caption, "tab:behaviour", "lrl" + "r" * len(names), head, body)


def plot_test_ce(main):
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for algorithm in ORDER:
            g = main[(main["dataset"] == dataset) & (main["algorithm"] == algorithm)].groupby("n_batches")["best_test_ce"]
            x = np.array(sorted(g.groups))
            ax.fill_between(x, g.quantile(0.25).loc[x], g.quantile(0.75).loc[x], color=COLORS[algorithm],
                            alpha=0.15, lw=0)
            ax.plot(x, g.median().loc[x], color=COLORS[algorithm], marker=MARKERS[algorithm], ms=3.5, lw=1.2,
                    label=LABELS[algorithm])
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_BATCHES, [str(n) for n in N_BATCHES])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("batches per epoch")
        ax.set_ylabel("test cross-entropy")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(frameon=False)
        fig.tight_layout(pad=0.3)
        fig.savefig(FIGURES_DIR / f"test_ce_{dataset}.pdf")
        plt.close(fig)


def plot_stored_vs_actual(chosen):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for algorithm in ["pso", "qpso"]:
        runs = [load_record(MAIN_DIR, chosen, "wine", algorithm, 32, seed)["history"] for seed in MAIN_SEEDS]
        epochs = np.arange(1, N_EPOCHS + 1)
        ax.plot(epochs, np.median([h["train_error"] for h in runs], axis=0), color=COLORS[algorithm], lw=1.2,
                label=f"{LABELS[algorithm]}, actual")
        ax.plot(epochs, np.median([h["gbest_err_stored"] for h in runs], axis=0), color=COLORS[algorithm], lw=1.2,
                ls="--", label=f"{LABELS[algorithm]}, stored")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("objective of gbest")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES_DIR / "stored_vs_actual_wine.pdf")
    plt.close(fig)


def print_results(comparisons, counts, result, table):
    print("verdicts (primary measure: test cross-entropy at the checkpoint)")
    for key in ["RQ1", "RQ2", "RQ3", "RQ4", "hypothesis"]:
        details = {k: v for k, v in result[key].items() if k not in ["verdict", "rule"]}
        print(f"  {key}: {result[key]['verdict']}  [{result[key]['rule']}]  {details}")

    print("\noutcome counts over all problems (better / worse / no difference)")
    for (check, measure), g in counts[counts["dataset"] == "all"].groupby(["check", "measure"], sort=False):
        cells = [f"{r['comparison']} {r['better']}/{r['worse']}/{r['no_difference']}" for _, r in g.iterrows()]
        print(f"  {check:16s} {measure:16s} " + "   ".join(cells))

    print("\nfull batch (n_batches = 1)")
    full = comparisons[comparisons["check"] == "full_batch"]
    for _, r in full.iterrows():
        print(f"  {r['measure']:16s} {r['dataset']:6s} friedman p = {r['friedman_p']:.3g}  {r['comparison']} "
              f"p_holm = {r['p_holm']:.3g}  median diff = {r['median_diff']:+.4g}  {r['outcome']}")

    print("\nbehaviour (medians over 30 seeds)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def main():
    chosen = json.loads(CHOSEN.read_text())
    main_runs = pd.read_csv(MAIN_CSV)
    glass_long = pd.read_csv(GLASS_LONG_CSV)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    equal_cost = equal_cost_runs(main_runs, chosen)
    equal_cost.to_csv(ANALYSIS_DIR / "equal_cost_runs.csv", index=False)
    comparisons = all_comparisons(main_runs, equal_cost, glass_long)
    comparisons.to_csv(ANALYSIS_DIR / "comparisons.csv", index=False)
    counts = count_outcomes(comparisons)
    counts.to_csv(ANALYSIS_DIR / "counts.csv", index=False)
    result = verdicts(comparisons)
    (ANALYSIS_DIR / "verdicts.json").write_text(json.dumps(result, indent=2) + "\n")

    both = pd.concat([main_runs, glass_long])
    for measure, name, label in [("test_ce", "cross-entropy", "tab:test_ce"), ("test_f1", "macro-F1", "tab:test_f1")]:
        table = summary_table(both, MEASURES[measure][0])
        table.to_csv(ANALYSIS_DIR / f"table_{measure}.csv", index=False)
        latex_summary(table[table["n_epochs"] == N_EPOCHS], ANALYSIS_DIR / f"table_{measure}.tex", name, label)

    behaviour_table = behaviour(chosen)
    behaviour_table.to_csv(ANALYSIS_DIR / "behaviour.csv", index=False)
    latex_behaviour(behaviour_table, ANALYSIS_DIR / "behaviour.tex")

    with plt.rc_context(PLOT_STYLE):
        plot_test_ce(main_runs)
        plot_stored_vs_actual(chosen)

    print_results(comparisons, counts, result, behaviour_table)
    print(f"\nsaved {', '.join(sorted(p.name for p in ANALYSIS_DIR.iterdir()))} to {ANALYSIS_DIR}")
    print(f"saved {', '.join(sorted(p.name for p in FIGURES_DIR.glob('*.pdf')))} to {FIGURES_DIR}")


def init_W1_norm(n_hidden):
    return np.sqrt(n_hidden / 3)


def run_stats(chosen):
    rows = []
    for dataset in DATASETS:
        for n_batches in N_BATCHES:
            for algorithm in ORDER:
                source = "pso" if algorithm == "pso_reeval" and n_batches == 1 else algorithm
                for seed in MAIN_SEEDS:
                    h = load_record(MAIN_DIR, chosen, dataset, source, n_batches, seed)["history"]
                    row = {"dataset": dataset, "n_batches": n_batches, "algorithm": algorithm,
                           "final_W1_norm": h["W1_norm"][-1]}
                    if "clamp_fraction" in h:
                        row["clamp_first100"] = np.mean(h["clamp_fraction"][:CLAMP_EPOCHS])
                        row["clamp_last100"] = np.mean(h["clamp_fraction"][-CLAMP_EPOCHS:])
                    rows.append(row)
    return pd.DataFrame(rows)


def c4_by_problem(main):
    rows = []
    for measure in ["test_f1", "train_objective"]:
        column, lower_is_better = MEASURES[measure]
        for dataset in DATASETS:
            for n_batches in N_BATCHES:
                comparisons = COMPARISONS if n_batches > 1 else FULL_BATCH_COMPARISONS
                c4 = [r for r in compare(paired(main, dataset, n_batches, column), comparisons, lower_is_better)
                      if r["comparison"] == "C4"][0]
                rows.append({"measure": measure, "dataset": dataset, "n_batches": n_batches,
                             "procedure": "mini-batch" if n_batches > 1 else "full batch",
                             **{k: c4[k] for k in ["friedman_p", "p_raw", "p_holm", "median_diff", "outcome"]}})
    return pd.DataFrame(rows)


def exploratory():
    chosen = json.loads(CHOSEN.read_text())
    main_runs = pd.read_csv(MAIN_CSV)
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    key = ["dataset", "n_batches", "algorithm"]
    medians = run_stats(chosen).groupby(key).median().reset_index()

    clamp = ordered(medians[medians["algorithm"] != "sgd"][key + ["clamp_first100", "clamp_last100"]])
    clamp.insert(3, "v_max", [chosen[d][a][str(n)]["v_max"] for d, n, a in clamp[key].itertuples(index=False)])
    clamp["v_max"] = clamp["v_max"].fillna("none")
    clamp.to_csv(EXPLORATORY_DIR / "clamp_fraction.csv", index=False)

    w1 = ordered(medians.pivot(index=["dataset", "n_batches"], columns="algorithm", values="final_W1_norm")[ORDER]
                 .reset_index())
    w1["init"] = [init_W1_norm(load_dataset(d)["n_hidden"]) for d in w1["dataset"]]
    w1.to_csv(EXPLORATORY_DIR / "final_W1_norm.csv", index=False)

    c4 = c4_by_problem(main_runs)
    c4.to_csv(EXPLORATORY_DIR / "c4_by_problem.csv", index=False)
    mini = c4[c4["procedure"] == "mini-batch"]
    c4_counts = (mini.groupby(["measure", "dataset"], sort=False)["outcome"].value_counts().unstack(fill_value=0)
                 .reindex(columns=OUTCOMES, fill_value=0).reset_index())
    c4_counts.to_csv(EXPLORATORY_DIR / "c4_counts.csv", index=False)

    evals = ordered(main_runs.groupby(["dataset", "n_batches", "algorithm"])["n_evals"].median().astype(int)
                    .unstack()[ORDER].reset_index())
    evals.to_csv(EXPLORATORY_DIR / "n_evals.csv", index=False)

    with pd.option_context("display.width", 200):
        print("1. median clamp fraction, epochs 1-100 and 901-1000\n" + clamp.to_string(index=False) + "\n")
        print("2. median final W1 norm, and sqrt(n_hidden/3) at initialisation\n"
              + w1.to_string(index=False, float_format=lambda v: f"{v:.3g}") + "\n")
        print("3. C4 (qpso vs sgd) per configuration\n" + c4.to_string(index=False) + "\n")
        print("   mini-batch outcome counts per problem\n" + c4_counts.to_string(index=False) + "\n")
        print("4. median n_evals per run\n" + evals.to_string(index=False))
    print(f"\nsaved {', '.join(sorted(p.name for p in EXPLORATORY_DIR.iterdir()))} to {EXPLORATORY_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "exploratory":
        exploratory()
    else:
        main()
