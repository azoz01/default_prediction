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
    category_embedder.fit(X_train, y_train, embedder_n_epochs=20)

    pipeline_logger.info("Transforming train sample")
    X_train_transformed: pd.DataFrame = category_embedder.transform(
        X_train, y_train
    )
    pipeline_logger.info("Transforming test sample")
    X_test_transformed: pd.DataFrame = category_embedder.transform(
        X_test, y_train
    )

    pipeline_logger.info(
        f"Saving output data to: {paths.CATEGORICAL_EMBEDDED_DATA_PATH}"
    )
    if not os.path.exists(paths.CATEGORICAL_EMBEDDED_DATA_PATH):
        os.mkdir(paths.CATEGORICAL_EMBEDDED_DATA_PATH)
    X_train_transformed.to_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "X_train.parquet")
    )
    y_train.to_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "y_train.parquet")
    )
    X_test_transformed.to_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "X_test.parquet")
    )
    y_test.to_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info(
        f"Saving serialized pipeline to {paths.PIPELINES_SERIALIZED_PATH}"
    )
    with open(
        os.path.join(
            paths.PIPELINES_SERIALIZED_PATH, "categorical_embedder.pkl"
        ),
        "wb",
    ) as f:
        pkl.dump(category_embedder, f)


if __name__ == "__main__":
    main()