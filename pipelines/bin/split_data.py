import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

from typing import Dict
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_parameters, get_data_path, save_data

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Started splitting data pipeline")
    parameters: Dict[str, str] = get_parameters("split_data")
    input_path: str = get_data_path("raw")
    output_path: str = get_data_path("splitted")

    logger.info(f"Loading data from {input_path}")
    raw_data: pd.DataFrame = pd.read_parquet(
        os.path.join(input_path, "application_data.parquet")
    )

    y: pd.DataFrame = raw_data[["TARGET"]]
    X: pd.DataFrame = raw_data.drop(columns=["TARGET"])

    logger.info(f"Splitting data")
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"] + parameters["valid_size"],
        stratify=y,
        random_state=parameters["random_state"],
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test,
        y_test,
        test_size=parameters["test_size"]
        / (parameters["test_size"] + parameters["valid_size"]),
        stratify=y_test,
        random_state=42,
    )

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    save_data(
        path=output_path,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
