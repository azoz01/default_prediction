import sys
import os


sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils.paths as paths
from utils.logging import model_logger
from models.utils.metrics import get_metrics
from models.utils.save_model import save_model


def main():

    X_train = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "X_train.parquet")
    )
    y_train = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "y_train.parquet")
    )
    X_test = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "X_test.parquet")
    )
    y_test = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "y_test.parquet")
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    model_logger.info(get_metrics(model, X_train, y_train))
    model_logger.info(get_metrics(model, X_test, y_test))
    model_logger.info(model.get_params())

    model_logger.info("Saving model")
    save_model(model, "logistic_regression")


if __name__ == "__main__":
    main()
