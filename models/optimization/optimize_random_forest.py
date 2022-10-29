import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from models.utils.optimize_model import optimize_model


def define_model(trial):
    params: Dict[str, str] = {
        "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100),
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
    }
    return RandomForestClassifier(**params)


def main():
    optimize_model(
        default_model_name="random_forest", define_model_fun=define_model
    )


if __name__ == "__main__":
    main()
