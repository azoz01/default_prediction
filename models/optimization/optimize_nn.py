import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

from sklearn.neural_network import MLPClassifier
from models.utils.optimize_model import optimize_model


def define_model(trial):
    hidden_layers_count = trial.suggest_int("hidden_layers_count", 1, 5)
    hidden_layers_sizes = [
        trial.suggest_int(f"hidden_layer_size_{i}", 10, 100)
        for i in range(hidden_layers_count)
    ]
    alpha = trial.suggest_float("alpha", 1e-5, 1e-3)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-3)
    max_iter = trial.suggest_int("max_iter", 100, 200)
    random_state = 42
    learning_rate = "adaptive"
    params = {
        "hidden_layer_sizes": hidden_layers_sizes,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "learning_rate_init": learning_rate_init,
        "random_state": random_state,
        "max_iter": max_iter,
    }
    return MLPClassifier(**params)


def main():
    optimize_model(
        default_model_name="neural_network", define_model_fun=define_model
    )


if __name__ == "__main__":
    main()
