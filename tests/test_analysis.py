import numpy as np

from analysis import COMPARISONS, compare, equal_cost_epoch, holm, init_W1_norm, outcome, wilcoxon_p
from data import load_dataset
from network import init_weights, unpack
from pso import train_pso


def test_holm_matches_a_worked_example():
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_outcome_direction_depends_on_whether_lower_is_better():
    diff = np.array([-0.2, -0.1, 0.05])
    assert outcome(diff, 0.01, lower_is_better=True) == "better"
    assert outcome(diff, 0.01, lower_is_better=False) == "worse"
    assert outcome(diff, 0.05, lower_is_better=True) == "no difference"
    assert outcome(diff, np.nan, lower_is_better=True) == "no difference"


def test_outcome_falls_back_to_the_mean_when_the_median_is_zero():
    diff = np.array([-3.0, 0.0, 0.0, 0.0, 1.0])
    assert outcome(diff, 0.01, lower_is_better=True) == "better"
    assert outcome(np.zeros(5), 0.01, lower_is_better=True) == "no difference"


def test_all_zero_differences_give_p_one():
    assert wilcoxon_p(np.zeros(30)) == 1.0


def test_significant_friedman_leads_to_holm_corrected_wilcoxon_tests():
    base = np.random.default_rng(0).normal(size=30)
    values = {"sgd": base + 1, "pso": base + 0.5, "pso_reeval": base + 0.5, "qpso": base}
    rows = {r["comparison"]: r for r in compare(values, COMPARISONS, lower_is_better=True)}
    assert rows["C1"]["friedman_p"] < 0.05
    for c in ["C1", "C3", "C4"]:
        assert rows[c]["outcome"] == "better"
        assert rows[c]["p_holm"] >= rows[c]["p_raw"]
    assert rows["C2"]["p_raw"] == 1.0 and rows["C2"]["outcome"] == "no difference"
    assert rows["C4"]["median_diff"] == -1.0


def test_non_significant_friedman_means_no_difference_everywhere():
    values = {a: np.array([(seed + k) % 4 for seed in range(32)], dtype=float)
              for k, a in enumerate(["sgd", "pso", "pso_reeval", "qpso"])}
    rows = compare(values, COMPARISONS, lower_is_better=True)
    assert all(r["friedman_p"] > 0.05 for r in rows)
    assert all(r["outcome"] == "no difference" and np.isnan(r["p_raw"]) for r in rows)


def test_equal_cost_window_never_exceeds_the_budget():
    n_evals = [100, 250, 400, 550, 700]
    val_ce = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert equal_cost_epoch(n_evals, val_ce, 400) == 3
    assert equal_cost_epoch(n_evals, val_ce, 399) == 2

    data = load_dataset("wine")
    budget = train_pso(data, 1e-3, 8, 6, seed=0)["n_evals"]
    h = train_pso(data, 1e-3, 8, 6, seed=0, reevaluate=True, n_quantum=5, r_cloud=0.05)["history"]
    assert h["n_evals"][-1] > budget
    for b in [budget, budget // 2, h["n_evals"][0]]:
        assert h["n_evals"][equal_cost_epoch(h["n_evals"], h["val_ce"], b) - 1] <= b


def test_initial_W1_norm_matches_init_weights():
    for sizes in [(13, 20, 3), (30, 20, 2), (9, 18, 6)]:
        W1 = unpack(init_weights(sizes, np.random.default_rng(0), n_vectors=4000), sizes)[0]
        norms = np.linalg.norm(W1.reshape(4000, -1), axis=1)
        assert np.isclose(np.sqrt(np.mean(norms ** 2)), init_W1_norm(sizes[1]), rtol=0.01)
