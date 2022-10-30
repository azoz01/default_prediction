import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

from sklearn.linear_model import LogisticRegression
from models.lib.utils.optimize_model import optimize_model


def define_model(trial):
    params = {
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "C": trial.suggest_float("C", 0.1, 100),
        "fit_intercept": False,
        "solver": "saga",
        "max_iter": 500,
        "random_state": 42,
    }
    return LogisticRegression(**params)


def main():
    optimize_model(
        default_model_name="logistic_regression", define_model_fun=define_model
    )


if __name__ == "__main__":
    main()
