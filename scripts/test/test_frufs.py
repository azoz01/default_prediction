import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import utils.paths as paths
from utils.logging import pipeline_logger
from FRUFS import FRUFS
from sklearn.tree import DecisionTreeRegressor


def main():
    pipeline_logger.info(
        f"Reading input data from {paths.CATEGORICAL_EMBEDDED_DATA_PATH}"
    )
    X_train: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "X_train.parquet")
    )
    y_train: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "y_train.parquet")
    )
    X_test: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "X_test.parquet")
    )
    y_test: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "y_test.parquet")
    )
    X_train = X_train.sample(frac=1e-4)
    X_test = X_test.sample(frac=1e-4)

    pipeline_logger.info("Fitting feature selector")
    feature_selector = FRUFS(n_jobs=-1, k=50, model_r=DecisionTreeRegressor())
    feature_selector.fit(X_train)
    pipeline_logger.info("Selecting features from train sample")
    X_train_transformed = feature_selector.transform(X_train)
    pipeline_logger.info("Selecting features from test sample")
    X_test_transformed = feature_selector.transform(X_test)

    pipeline_logger.info("Checking for nulls")
    assert (
        X_train_transformed.isna().sum().sum() == 0
    ), "Train sample contains nulls"
    assert (
        X_test_transformed.isna().sum().sum() == 0
    ), "Test sample contains nulls"
    assert (
        X_train_transformed.isnull().sum().sum() == 0
    ), "Train sample contains nulls"
    assert (
        X_test_transformed.isnull().sum().sum() == 0
    ), "Test sample contains nulls"

    pipeline_logger.info("Checking for datatypes correctness")
    assert (
        len(
            X_train_transformed.select_dtypes(
                exclude=["float64", "float32"]
            ).columns
        )
        == 0
    ), "There are non-numerical columns in train sample"
    assert (
        len(
            X_test_transformed.select_dtypes(
                exclude=["float64", "float32"]
            ).columns
        )
        == 0
    ), "There are non-numerical columns in test sample"

    pipeline_logger.info("Checking for proper dimensions of output data")
    assert X_train_transformed.shape[1] == 50
    assert X_test_transformed.shape[1] == 50


if __name__ == "__main__":
    main()
