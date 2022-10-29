import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
import logging
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.io import load_data, save_pickle
from models.utils.metrics import get_metrics

logger = logging.getLogger(__name__)


def main():
    data_input_path = "resources/data/reduced"
    model_output_path = "resources/models/serialized/gaussian_nb.pkl"

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        data_input_path
    )
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    logger.info(
        f"Metrics on train set: {get_metrics(model, X_train, y_train)}"
    )
    logger.info(
        f"Metrics on validation set: {get_metrics(model, X_valid, y_valid)}"
    )
    logger.info(f"Metrics on test set: {get_metrics(model, X_test, y_test)}")

    logger.info(f"Saving model")
    save_pickle(model, model_output_path)


if __name__ == "__main__":
    main()
