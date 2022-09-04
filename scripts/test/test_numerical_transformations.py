import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
from pipelines.built.numerical_transformations import NumericalTransformations
from pipelines.constants import *
import utils.paths as paths
import pipelines.constants as constants
from utils.logging import pipeline_logger


def main():
    pipeline_logger.info("Started test of numerical transformations")
    pipeline_logger.info(
        f"Reading input data from {paths.NO_MISSING_DATA_PATH}"
    )
    X_train: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.NO_MISSING_DATA_PATH, "X_train.parquet")
    )
    X_test: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.NO_MISSING_DATA_PATH, "X_test.parquet")
    )

    pipeline_logger.info("Fitting pipeline")
    numerical_pipeline = NumericalTransformations()
    numerical_pipeline.fit(X_train)
    pipeline_logger.info("Transforming train sample")
    X_train_transformed = numerical_pipeline.transform(X_train)
    pipeline_logger.info("Transforming test sample")
    X_test_transformed = numerical_pipeline.transform(X_test)

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

    pipeline_logger.info("Checking for zero mean in numerical data")
    assert (
        X_train_transformed[constants.NUMERICAL_COLUMNS].mean(axis=0).abs()
        <= 1e-5
    ).all(), "Numerical in train sample columns has non-zero mean"

    pipeline_logger.info("Checking for unit std in numerical data")
    assert (
        (
            X_train_transformed[constants.NUMERICAL_COLUMNS].std(axis=0) - 1
        ).abs()
        <= 1e-5
    ).all(), "Numerical in train sample columns has std non-equal to 1"

    frac_skew_greater_1 = (
        X_train_transformed[constants.NUMERICAL_COLUMNS].skew().abs() >= 1
    ).mean()

    frac_skew_greater_2 = (
        X_train_transformed[constants.NUMERICAL_COLUMNS].skew().abs() >= 2
    ).mean()

    # TODO: Target is to have zeros here
    pipeline_logger.info(
        f"Fraction of features with skewness >= 1: {frac_skew_greater_1}"
    )
    pipeline_logger.info(
        f"Fraction of features with skewness >= 2: {frac_skew_greater_2}"
    )


if __name__ == "__main__":
    main()
