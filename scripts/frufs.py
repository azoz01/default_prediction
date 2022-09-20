import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
import pickle as pkl
import utils.paths as paths
from utils.logging import pipeline_logger
from FRUFS import FRUFS
from sklearn.tree import DecisionTreeRegressor


def main():
    pipeline_logger.info(
        f"Reading input data from {paths.CATEGORICAL_EMBEDDED_DATA_PATH}"
    )
    X_train: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "X_train.parquet")
    )
    y_train: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "y_train.parquet")
    )
    X_test: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "X_test.parquet")
    )
    y_test: pd.DataFrame = pd.read_parquet(
        os.path.join(paths.CATEGORICAL_EMBEDDED_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info("Fitting feature selector")
    feature_selector = FRUFS(n_jobs=-1, k=50, model_r=DecisionTreeRegressor())
    feature_selector.fit(X_train)
    pipeline_logger.info("Selecting features from train sample")
    X_train_transformed = feature_selector.transform(X_train)
    pipeline_logger.info("Selecting features from test sample")
    X_test_transformed = feature_selector.transform(X_test)

    pipeline_logger.info(
        f"Saving output data to: {paths.FEATURE_SELECTED_DATA_PATH}"
    )
    if not os.path.exists(paths.FEATURE_SELECTED_DATA_PATH):
        os.mkdir(paths.FEATURE_SELECTED_DATA_PATH)
    X_train_transformed.to_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "X_train.parquet")
    )
    y_train.to_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "y_train.parquet")
    )
    X_test_transformed.to_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "X_test.parquet")
    )
    y_test.to_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "y_test.parquet")
    )

    pipeline_logger.info(
        f"Saving serialized pipeline to {paths.PIPELINES_SERIALIZED_PATH}"
    )
    with open(
        os.path.join(paths.PIPELINES_SERIALIZED_PATH, "frufs.pkl"), "wb",
    ) as f:
        pkl.dump(feature_selector, f)


if __name__ == "__main__":
    main()
