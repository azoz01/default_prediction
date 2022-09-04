import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
from pipelines.built.numerical_transformations import NumericalTransformations
from pipelines.constants import *
import utils.paths as paths
import pickle as pkl
from utils.logging import pipeline_logger


def main():
    pipeline_logger.info("started numerical transformations")
    pipeline_logger.info(
        f"Reading input data from {paths.NO_MISSING_DATA_PATH}"
    )
    X_train = pd.read_parquet(
        os.path.join(paths.NO_MISSING_DATA_PATH, "X_train.parquet")
    )
    y_train = pd.read_parquet(
        os.path.join(paths.NO_MISSING_DATA_PATH, "y_train.parquet")
    )
    X_test = pd.read_parquet(
        os.path.join(paths.NO_MISSING_DATA_PATH, "X_test.parquet")
    )
    y_test = pd.read_parquet(
        os.path.join(paths.NO_MISSING_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info("Fitting pipeline")
    numerical_transformations = NumericalTransformations()
    numerical_transformations.fit(X_train)
    pipeline_logger.info("Transforming train sample")
    X_train_transformed = numerical_transformations.transform(X_train)
    pipeline_logger.info("Transforming test sample")
    X_test_transformed = numerical_transformations.transform(X_test)

    pipeline_logger.info(
        f"Saving output data to: {paths.NUMERICAL_TRANSFORMED_DATA_PATH}"
    )
    if not os.path.exists(paths.NUMERICAL_TRANSFORMED_DATA_PATH):
        os.mkdir(paths.NUMERICAL_TRANSFORMED_DATA_PATH)
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
        f"Saving serialized pipeline to {paths.PIPELINES_SERIALIZED_PATH}"
    )
    with open(
        os.path.join(
            paths.PIPELINES_SERIALIZED_PATH, "numerical_pipeline.pkl"
        ),
        "wb",
    ) as f:
        pkl.dump(numerical_transformations, f)


if __name__ == "__main__":
    main()
