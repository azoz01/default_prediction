import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import utils.paths as paths
from pipelines.built import MissingDataRemovalPipeline
from utils.logging import pipeline_logger


def main():
    pipeline_logger.info(f"Reading input data from {paths.SPLITTED_DATA_PATH}")
    X_train: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "X_train.parquet")
    )
    X_test: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "X_test.parquet")
    )

    pipeline_logger.info("Fitting pipeline")
    missing_data_removal_pipeline = MissingDataRemovalPipeline()
    missing_data_removal_pipeline.fit(X_train)
    pipeline_logger.info("Transforming train sample")
    X_train_transformed = missing_data_removal_pipeline.transform(X_train)
    pipeline_logger.info("Transforming test sample")
    X_test_transformed = missing_data_removal_pipeline.transform(X_test)

    pipeline_logger.info("Columns with nulls in training sample: ")
    na_cols_mask = X_train_transformed.isna().any().values
    columns_with_nas = X_train_transformed.columns[na_cols_mask]
    pipeline_logger.info(X_train_transformed.isna().sum()[columns_with_nas])

    pipeline_logger.info("Columns with nulls in testing sample: ")
    na_cols_mask = X_test_transformed.isna().any().values
    columns_with_nas = X_test_transformed.columns[na_cols_mask]
    pipeline_logger.info(X_test_transformed.isna().sum()[columns_with_nas])

    assert (
        X_train_transformed.isna().sum().sum() == 0
    ), "Train sample contains nulls"
    assert (
        X_test_transformed.isna().sum().sum() == 0
    ), "Test sample contains nulls"


if __name__ == "__main__":
    main()
