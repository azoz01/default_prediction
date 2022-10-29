import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

from typing import Dict, Any
import logging
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from pipelines.lib.data_balance import SmotencWrapper
from pipelines.lib.dummy import DummyPipeline
from utils.parameters import get_data_path, get_parameters
from utils.io import load_data, save_data, update_data_state

logger = logging.getLogger(__name__)

OVERSAMPLE_METHODS: Dict[str, Any] = {
    "oversample": RandomOverSampler(random_state=42),
    "smotenc": SmotencWrapper(),
    "passthrough": DummyPipeline(),
}


def main():
    logger.info("Started balancing data pipeline")
    input_path: str = get_data_path("transformed_numerical_columns")
    output_path: str = get_data_path("balanced")
    parameters: Dict[str, str] = get_parameters("balance_data")

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        path=input_path
    )
    logger.info("Balancing train data")
    oversample = OVERSAMPLE_METHODS[parameters["method"]]
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
    update_data_state("balance", parameters["method"])


if __name__ == "__main__":
    main()
