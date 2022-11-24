#! /bin/bash

n_trials=$1

python3.10 models/bin/optimization/optimize_logistic_regression.py --n_trials=$n_trials &&
python3.10 models/bin/optimization/optimize_random_forest.py --n_trials=$n_trials &&
python3.10 models/bin/optimization/optimize_xgboost.py --n_trials=$n_trials