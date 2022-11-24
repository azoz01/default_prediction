import logging
import mlflow
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from models.lib.utils.metrics import get_metrics
from utils.parameters import get_data_path
from utils.io import load_data, save_model

logger = logging.getLogger(__name__)

DATA_PATH_NAME = "reduced"


def main():
    with mlflow.start_run(run_name="lda"):
        data_path = get_data_path(DATA_PATH_NAME)
        model = LinearDiscriminantAnalysis()
        mlflow.log_param("type", "lda")

        logger.info("Loading data")
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
            path=data_path
        )

        logger.info("Fitting model to train data")
        model.fit(X_train, y_train)

        logger.info("Evaluating model")
        train_metrics = get_metrics(model, X_train, y_train)
        logger.info(f"Metrics on train set: {train_metrics}")
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
            "resources/models/serialized/lda.pkl",
            "resources/models/parameters/lda.json",
        )
        logger.info("Publishing artifacts")
        mlflow.log_artifact("resources/models/serialized/lda.pkl")
        mlflow.log_artifact("resources/models/parameters/lda.json")


if __name__ == "__main__":
    main()
