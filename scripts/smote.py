import sys
import os


sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import utils.paths as paths
from utils.logging import pipeline_logger
from utils.parameters import CALIBRATION_SIZE


def main():
    pipeline_logger.info("Started SMOTE pipeline")
    pipeline_logger.info(
        f"Reading input data from {paths.FEATURE_SELECTED_DATA_PATH}"
    )
    X_train = pd.read_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "X_train.parquet")
    )
    y_train = pd.read_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "y_train.parquet")
    )
    X_test = pd.read_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "X_test.parquet")
    )
    y_test = pd.read_parquet(
        os.path.join(paths.FEATURE_SELECTED_DATA_PATH, "y_test.parquet")
    )

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=CALIBRATION_SIZE
    )
    oversample = SMOTE(random_state=42)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    pipeline_logger.info(f"Saving output data into {paths.BALANCED_DATA_PATH}")
    if not os.path.exists(paths.BALANCED_DATA_PATH):
        os.mkdir(paths.BALANCED_DATA_PATH)
    X_train.to_parquet(
        os.path.join(paths.BALANCED_DATA_PATH, "X_train.parquet")
    )
    y_train.to_parquet(
        os.path.join(paths.BALANCED_DATA_PATH, "y_train.parquet")
    )
    X_calib.to_parquet(
        os.path.join(paths.BALANCED_DATA_PATH, "X_calib.parquet")
    )
    y_calib.to_parquet(
        os.path.join(paths.BALANCED_DATA_PATH, "y_calib.parquet")
    )
    X_test.to_parquet(os.path.join(paths.BALANCED_DATA_PATH, "X_test.parquet"))
    y_test.to_parquet(os.path.join(paths.BALANCED_DATA_PATH, "y_test.parquet"))


if __name__ == "__main__":
    main()
