# Analysis procedure

- Unit of analysis: the 30 runs (seeds 0-29) of each configuration. Each run is represented by its checkpoint: the epoch with the lowest validation cross-entropy.
- Measures:
  - Primary: test cross-entropy at the checkpoint (lower is better).
  - Secondary: test macro-F1 at the checkpoint (higher is better), and the training objective f(theta, D_T) at the final epoch.
- Mini-batch configurations: 3 problems x n_batches {32, 16, 8, 4, 2} = 15.
- Per configuration:
  - Run a Friedman test over the four algorithms, with seeds as blocks, at alpha = 0.05.
  - If it is significant, run four planned two-sided Wilcoxon signed-rank tests paired by seed:
    - C1: QPSO vs static PSO
    - C2: re-evaluating PSO vs static PSO
    - C3: QPSO vs re-evaluating PSO
    - C4: QPSO vs SGD
  - Apply the Holm correction over the four p-values. Implement Holm by hand.
  - If the Friedman test is not significant, all four outcomes are "no difference".
- Outcomes:
  - "better" or "worse" for the first algorithm of the pair when the Holm-adjusted p < 0.05. The direction comes from the sign of the median paired difference, or from the mean if the median is 0.
  - Otherwise "no difference".
  - If all paired differences are 0, use p = 1.
  - Effect size: the median paired difference, first minus second.
- Verdicts:
  - RQ1 = yes if C1 is "better" in at least 8 of the 15 configurations.
  - RQ4 = yes if C4 is "better" in at least 8 of the 15 configurations.
  - RQ2: for each problem, compute the Spearman correlation between n_batches and the median paired difference (static PSO minus QPSO, test cross-entropy) over the five mini-batch settings. RQ2 is supported if the correlation is positive on all three problems.
  - RQ3: count the "better" outcomes for C2 (re-evaluation) and C3 (quantum particles). The larger count names the main mechanism.
  - Hypothesis: supported if RQ1 and RQ4 are yes; partially supported if only RQ1 is yes; not supported if RQ1 is no.
- Robustness checks, each using the same procedure:
  - (a) Test macro-F1.
  - (b) Equal cost: restrict each QPSO and re-evaluating PSO run to the epochs whose cumulative n_evals are at most the final n_evals of the static PSO run with the same seed. Take the best-validation epoch within that window. Compare C1-C3 only.
  - (c) Glass with 2000 epochs.
- Full batch (n_batches = 1), reported separately and not part of the verdicts: a Friedman test over SGD, static PSO and QPSO, then C1 and C4 with Holm over the two p-values.
- Behaviour measures for the PSO variants at n_batches 32 and 1:
  - mean gbest changes per epoch;
  - the ratio of actual to stored gbest objective at the final epoch;
  - mean clamp fraction over the last 100 epochs;
  - final diversity;
  - final W1 norm.

## Exploratory analysis (added after unblinding)

These analyses were chosen after the main results had been read. The rules above were committed first, in 0283b07 ("fix analysis procedure"). The analyses below are not part of that procedure and do not change any verdict. Run them with `python analysis.py exploratory`; the outputs go to `results/analysis/exploratory/`. At n_batches = 1, re-evaluating PSO is the static PSO run.

1. Clamp fraction: for every PSO variant and configuration, the mean clamp fraction of each run over epochs 1-100 and over epochs 901-1000, then the median over the 30 seeds.
2. Final W1 norm: the median over seeds of ||W1|| at the final epoch, for SGD and the three PSO variants in every configuration. It is shown next to the expected norm at initialisation, sqrt(n_hidden/3). Each W1 entry is uniform in +-1/sqrt(n_inputs), so E||W1||^2 = n_hidden/3.
3. C4 (QPSO vs SGD) for test macro-F1 and for the training objective at the final epoch, per problem:
   - the outcome of every configuration;
   - the mini-batch configurations use the four-algorithm Friedman test and Holm over C1-C4;
   - full batch uses the full-batch procedure;
   - the mini-batch outcomes are counted per problem.
4. Evaluation cost: the median over seeds of the final n_evals of each run, for SGD, static PSO, re-evaluating PSO and QPSO in every configuration.
