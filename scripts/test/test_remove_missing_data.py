import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import utils.paths as paths
from pipelines.built import MissingDataRemovalPipeline
from utils.logging import pipeline_logger


def main():
    pipeline_logger.info(f"Reading input data from {paths.SPLITTED_DATA_PATH}")
    X_train = pd.read_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "X_train.parquet")
    )
    y_train = pd.read_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "y_train.parquet")
    )
    X_test = pd.read_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "X_test.parquet")
    )
    y_test = pd.read_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info("Fitting pipeline")
    missing_data_removal_pipeline = MissingDataRemovalPipeline()
    missing_data_removal_pipeline.fit(X_train, y_train)

    pipeline_logger.info("Transforming train sample")
    (
        X_train_transformed,
        y_train_transformed,
    ) = missing_data_removal_pipeline.transform(X_train, y_train)
    pipeline_logger.info("Transforming test sample")
    (
        X_test_transformed,
        y_test_transformed,
    ) = missing_data_removal_pipeline.transform(X_test, y_test)

    pipeline_logger.info("Columns with nulls in training sample: ")
    na_cols_mask = X_train_transformed.isna().any().values
    columns_with_nas = X_train_transformed.columns[na_cols_mask]
    pipeline_logger.info(X_train_transformed.isna().sum()[columns_with_nas])

    pipeline_logger.info("Columns with nulls in testing sample: ")
    na_cols_mask = X_test_transformed.isna().any().values
    columns_with_nas = X_test_transformed.columns[na_cols_mask]
    pipeline_logger.info(X_test_transformed.isna().sum()[columns_with_nas])

    assert X_train_transformed.shape[0] == y_train_transformed.shape[0]
    assert X_test_transformed.shape[0] == y_test_transformed.shape[0]


if __name__ == "__main__":
    main()
