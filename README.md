# Machine Learning 741: Assignment 3

Is mini-batch training of a neural network a dynamic optimisation problem?
This project trains a small neural network with SGD, PSO and quantum PSO
(QPSO) on the Wine, Breast Cancer and Glass datasets, using different
mini-batch sizes, and compares the results.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run these from the project folder, in this order:

```bash
python -m pytest                 # tests
python data_summary.py           # dataset tables
python experiments.py pilot      # short pilot runs
python experiments.py tune       # parameter tuning (takes a few hours)
python experiments.py main       # main experiment (about 1.5 hours)
python analysis.py               # statistical tests, tables and figures
python analysis.py exploratory   # extra analysis
```

Results are saved in `results/` and figures in `figures/`.