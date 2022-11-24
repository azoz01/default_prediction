import logging
import mlflow
from sklearn.linear_model import LogisticRegression
from models.lib.utils.metrics import get_metrics
from utils.parameters import get_data_path, get_models_path
from utils.io import load_data, save_model, read_json

logger = logging.getLogger(__name__)

DATA_PATH_NAME = "reduced"


def main() -> None:
    with mlflow.start_run(run_name="logistic"):
        data_path = get_data_path(DATA_PATH_NAME)
        parameters_path = (
            get_models_path("parameters") / "logistic_regression.json"
        )
        tunable_parameters = read_json(parameters_path)
        constant_parameters = {
            "fit_intercept": True,
            "solver": "saga",
            "max_iter": 500,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        model_parameters = {**constant_parameters, **tunable_parameters}
        model = LogisticRegression(**model_parameters)
        for name, value in model_parameters.items():
            mlflow.log_param(name, value)

        logger.info("Loading data")
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
            path=data_path
        )

        logger.info("Fitting model to train data")
        model.fit(X_train, y_train)

        logger.info("Evaluating model")
        train_metrics = get_metrics(model, X_train, y_train)
        logger.info(f"Metrics on train set: {train_metrics}")
        mlflow.log_param("type", "logistic_regression")
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        logger.info(
            "Metrics on validation set: "
            f"{get_metrics(model, X_valid, y_valid)}"
        )
        test_metrics = get_metrics(model, X_test, y_test)
        logger.info(f"Metrics on test set: {test_metrics}")
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        logger.info("Saving model")
        save_model(
            model,
            "resources/models/serialized/logistic_regression.pkl",
            "resources/models/parameters/logistic_regression.json",
        )
        logger.info("Publishing artifacts")
        mlflow.log_artifact(
            "resources/models/serialized/logistic_regression.pkl"
        )
        mlflow.log_artifact(
            "resources/models/parameters/logistic_regression.json"
        )


if __name__ == "__main__":
    main()
