import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
from pipelines.built.numerical_pipeline import NumericalPipeline
from pipelines.constants import *
import utils.paths as paths
import pickle as pkl
from utils.logging import pipeline_logger


def main():

    pipeline_logger.info(
        f"Reading input data from {paths.NO_MISSING_DATA_PATH}"
    )
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
    numerical_pipeline = NumericalPipeline()
    numerical_pipeline.fit(X_train)
    pipeline_logger.info("Transforming train sample")
    X_train_transformed = numerical_pipeline.transform(X_train)
    pipeline_logger.info("Transforming test sample")
    X_test_transformed = numerical_pipeline.transform(X_test)

    pipeline_logger.info(
        f"Saving output data to: {paths.NUMERICAL_TRANSFORMED_DATA_PATH}"
    )
    X_train_transformed.to_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "X_train.parquet")
    )
    y_train.to_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "y_train.parquet")
    )
    X_test_transformed.to_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "X_test.parquet")
    )
    y_test.to_parquet(
        os.path.join(paths.NUMERICAL_TRANSFORMED_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info(
        f"Saving serialized pipeline to {paths.SERIALIZED_PATH}"
    )
    with open(
        os.path.join(paths.SERIALIZED_PATH, "numerical_pipeline.pkl"), "wb",
    ) as f:
        pkl.dump(numerical_pipeline, f)


if __name__ == "__main__":
    main()
