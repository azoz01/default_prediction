import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import utils.paths as paths
from pipelines.built import CategoricalEncoder
from utils.logging import pipeline_logger
from pandas.api.types import is_numeric_dtype


def main():
    pipeline_logger.info(
        f"Reading input data from {paths.NUMERICAL_TRANSFORMED_DATA_PATH}"
    )
    X_train: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "X_train.parquet")
    )
    X_test: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "X_test.parquet")
    )

    pipeline_logger.info("Fitting pipeline")
    categorical_encoder = CategoricalEncoder()
    categorical_encoder.fit(X_train)
    pipeline_logger.info("Transforming train sample")
    X_train_transformed = categorical_encoder.transform(X_train)
    pipeline_logger.info("Transforming test sample")
    X_test_transformed = categorical_encoder.transform(X_test)

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
        len(X_train_transformed.select_dtypes(exclude=["float64"]).columns)
        == 0
    ), "There are non-numerical columns in train sample"
    assert (
        len(X_test_transformed.select_dtypes(exclude=["float64"]).columns) == 0
    ), "There are non-numerical columns in test sample"


if __name__ == "__main__":
    main()
