import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

from typing import Dict
from xgboost import XGBClassifier
from models.utils.optimize_model import optimize_model


def define_model(trial):
    params: Dict[str, str] = {
        "eta": trial.suggest_float("eta", 0, 1),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "lambda": trial.suggest_float("lambda", 1e-3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        "tree_method": "gpu_hist",
        "scale_pos_weight": 11.381618,
    }
    return XGBClassifier(**params)


def main():
    optimize_model(default_model_name="xgboost", define_model_fun=define_model)


if __name__ == "__main__":
    main()
