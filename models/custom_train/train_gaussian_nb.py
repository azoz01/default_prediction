import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
import logging
import mlflow
from sklearn.naive_bayes import GaussianNB
from utils.io import load_data, save_pickle
from models.lib.utils.metrics import get_metrics

logger = logging.getLogger(__name__)


def main():
    data_input_path = "resources/data/reduced"
    model_output_path = "resources/models/serialized/gaussian_nb.pkl"
    mlflow.start_run(run_name="gaussian_nb")
    mlflow.log_param("model_class", "gaussian_nb")

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        data_input_path
    )
    model = GaussianNB()
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    train_metrics = get_metrics(model, X_train, y_train)
    logger.info(f"Metrics on train set: {train_metrics}")
    for metric, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)
    logger.info(
        f"Metrics on validation set: {get_metrics(model, X_valid, y_valid)}"
    )
    test_metrics = get_metrics(model, X_test, y_test)
    logger.info(f"Metrics on test set: {test_metrics}")
    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)
    save_pickle(model, model_output_path)
    mlflow.log_artifact(model_output_path)
    mlflow.end_run()


if __name__ == "__main__":
    main()
