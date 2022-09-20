import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pickle as pkl
import pandas as pd

import utils.paths as paths
from utils.logging import pipeline_logger
from pipelines.categorical import CategoryEmbedder
from pipelines.categorical.category_embedder import CategoryEmbedder


def main():
    pipeline_logger.info(
        f"Reading input data from {paths.NUMERICAL_TRANSFORMED_DATA_PATH}"
    )
    X_train = pd.read_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "X_train.parquet")
    )
    y_train = pd.read_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "y_train.parquet")
    )
    X_test = pd.read_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "X_test.parquet")
    )
    y_test = pd.read_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "y_test.parquet")
    )

    category_embedder = CategoryEmbedder()

    pipeline_logger.info("Fitting embedder")
    category_embedder.fit(X_train, y_train, embedder_n_epochs=1)

    pipeline_logger.info("Transforming train sample")
    X_train_transformed: pd.DataFrame = category_embedder.transform(
        X_train, y_train
    )
    pipeline_logger.info("Transforming test sample")
    X_test_transformed: pd.DataFrame = category_embedder.transform(
        X_test, y_test
    )

    assert (
        X_train_transformed.shape[0] == y_train.shape[0]
    ), "Shapes of train sample are inconsistent"
    assert (
        X_test_transformed.shape[0] == y_test.shape[0]
    ), "Shapes of train sample are inconsistent"
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
        len(X_train_transformed.select_dtypes(exclude=["float32"]).columns)
        == 0
    ), "There are non-numerical columns in train sample"
    assert (
        len(X_test_transformed.select_dtypes(exclude=["float32"]).columns) == 0
    ), "There are non-numerical columns in test sample"


if __name__ == "__main__":
    main()
