import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pickle as pkl
import pandas as pd
from pipelines.built import MissingDataRemovalPipeline
from utils.paths import (
    SPLITTED_DATA_PATH,
    NO_MISSING_DATA_PATH,
    SERIALIZED_PATH,
)
from utils.logging import pipeline_logger


def main():

    pipeline_logger.info(f"Reading input data from {SPLITTED_DATA_PATH}")
    X_train = pd.read_parquet(
        os.path.join(SPLITTED_DATA_PATH, "X_train.parquet")
    )
    y_train = pd.read_parquet(
        os.path.join(SPLITTED_DATA_PATH, "y_train.parquet")
    )
    X_test = pd.read_parquet(
        os.path.join(SPLITTED_DATA_PATH, "X_test.parquet")
    )
    y_test = pd.read_parquet(
        os.path.join(SPLITTED_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info("Fitting pipeline")
    missing_data_removal_pipeline = MissingDataRemovalPipeline()
    missing_data_removal_pipeline.fit(X_train)

    pipeline_logger.info("Transforming train sample")
    X_train_transformed = missing_data_removal_pipeline.transform(X_train)
    pipeline_logger.info("Transforming test sample")
    X_test_transformed = missing_data_removal_pipeline.transform(X_test)

    pipeline_logger.info(f"Saving output data to: {NO_MISSING_DATA_PATH}")
    X_train_transformed.to_parquet(
        os.path.join(NO_MISSING_DATA_PATH, "X_train.parquet")
    )
    X_test_transformed.to_parquet(
        os.path.join(NO_MISSING_DATA_PATH, "X_test.parquet")
    )
    y_train.to_parquet(os.path.join(NO_MISSING_DATA_PATH, "y_train.parquet"))
    y_test.to_parquet(os.path.join(NO_MISSING_DATA_PATH, "y_test.parquet"))

    pipeline_logger.info(f"Saving serialized pipeline to {SERIALIZED_PATH}")
    with open(
        os.path.join(SERIALIZED_PATH, "missing_data_removal_pipeline.pkl"),
        "wb",
    ) as f:
        pkl.dump(missing_data_removal_pipeline, f)


if __name__ == "__main__":
    main()
