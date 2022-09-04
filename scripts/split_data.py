import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
import utils.paths as paths
import utils.parameters as parameters
from utils.logging import pipeline_logger

from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    pipeline_logger.info("Started splitting data pipeline")
    pipeline_logger.info(f"Loading raw data from {paths.RAW_DATA_PATH}")
    raw_data = pd.read_parquet(
        os.path.join(paths.RAW_DATA_PATH, "application_data.parquet")
    )
    y = raw_data[["TARGET"]]
    X = raw_data.drop(columns=["TARGET"])

    pipeline_logger.info(f"Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters.TEST_SIZE, stratify=y
    )

    pipeline_logger.info(
        f"Saving splitted data into {paths.SPLITTED_DATA_PATH}"
    )
    if not os.path.exists(paths.SPLITTED_DATA_PATH):
        os.mkdir(paths.SPLITTED_DATA_PATH)
    X_train.to_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "X_train.parquet")
    )
    X_test.to_parquet(os.path.join(paths.SPLITTED_DATA_PATH, "X_test.parquet"))
    y_train.to_parquet(
        os.path.join(paths.SPLITTED_DATA_PATH, "y_train.parquet")
    )
    y_test.to_parquet(os.path.join(paths.SPLITTED_DATA_PATH, "y_test.parquet"))


if __name__ == "__main__":
    main()
