import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import pickle as pkl
import utils.paths as paths
from pipelines.built import CategoricalEncoder
from utils.logging import pipeline_logger


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

    pipeline_logger.info("Fitting pipeline")
    categorical_encoder = CategoricalEncoder()
    categorical_encoder.fit(X_train)
    pipeline_logger.info("Transforming train sample")
    X_train_transformed = categorical_encoder.transform(X_train)
    pipeline_logger.info("Transforming test sample")
    X_test_transformed = categorical_encoder.transform(X_test)

    pipeline_logger.info(
        f"Saving output data to: {paths.CATEGORICAL_TRANSFORMED_DATA_PATH}"
    )
    if not os.path.exists(paths.CATEGORICAL_TRANSFORMED_DATA_PATH):
        os.mkdir(paths.CATEGORICAL_TRANSFORMED_DATA_PATH)
    X_train_transformed.to_parquet(
        os.path.join(
            paths.CATEGORICAL_TRANSFORMED_DATA_PATH, "X_train.parquet"
        )
    )
    y_train.to_parquet(
        os.path.join(
            paths.CATEGORICAL_TRANSFORMED_DATA_PATH, "y_train.parquet"
        )
    )
    X_test_transformed.to_parquet(
        os.path.join(paths.CATEGORICAL_TRANSFORMED_DATA_PATH, "X_test.parquet")
    )
    y_test.to_parquet(
        os.path.join(paths.CATEGORICAL_TRANSFORMED_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info(
        f"Saving serialized pipeline to {paths.PIPELINES_SERIALIZED_PATH}"
    )
    with open(
        os.path.join(
            paths.PIPELINES_SERIALIZED_PATH, "categorical_pipeline.pkl"
        ),
        "wb",
    ) as f:
        pkl.dump(categorical_encoder, f)


if __name__ == "__main__":
    main()
