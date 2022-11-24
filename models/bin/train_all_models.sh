#! /bin/bash

python models/bin/training/train_gaussian_nb.py &&
python models/bin/training/train_lda.py &&
python models/bin/training/train_logistic_regression.py &&
python models/bin/training/train_neural_network.py &&
python models/bin/training/train_random_forest.py &&
python models/bin/training/train_xgboost.py