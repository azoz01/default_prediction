from sklearn.linear_model import LogisticRegression
from models.lib.utils.optimize_model import optimize_model


def define_model(trial):
    params = {
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "C": trial.suggest_float("C", 1e-4, 1),
        "fit_intercept": True,
        "solver": "saga",
        "max_iter": 500,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    return LogisticRegression(**params)


def main() -> None:
    optimize_model(
        default_model_name="logistic_regression", define_model_fun=define_model
    )


if __name__ == "__main__":
    main()
