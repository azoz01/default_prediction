import sys
import os


sys.path.append(os.path.abspath(os.getcwd()))
from typing import Dict
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from utils.parameters import get_data_path, get_parameters
from utils.io import load_data, save_data

logger = logging.getLogger(__name__)


def main():
    logger.info("Started oversampling pipeline")
    params: Dict[str, str] = get_parameters("oversample")
    input_path: str = get_data_path("reduced")
    output_path: str = get_data_path("balanced")

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        path=input_path
    )
    logger.info("Splitting data into train and calibration samples")
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=params["calibration_size"]
    )
    oversample = RandomOverSampler(random_state=42)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    save_data(
        path=output_path,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )
    logger.info(f"Saving calibration sample into {output_path} started")
    X_calib.to_parquet(os.path.join(output_path, "X_calib.parquet"))
    y_calib.to_parquet(os.path.join(output_path, "y_calib.parquet"))
    logger.info(f"Saving calibration sample into {output_path} sucessful")


if __name__ == "__main__":
    main()
