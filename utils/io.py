import os
import logging
from typing import Any, Dict, Tuple
import pandas as pd
import json
import pickle as pkl

from utils.parameters import get_data_path

logger = logging.getLogger(__name__)


def load_data(
    path: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    logger.info(f"Loading train, valid and test data from {path} started")
    X_train: pd.DataFrame = pd.read_parquet(
        os.path.join(path, "X_train.parquet")
    )
    y_train: pd.DataFrame = pd.read_parquet(
        os.path.join(path, "y_train.parquet")
    )
    X_valid: pd.DataFrame = pd.read_parquet(
        os.path.join(path, "X_valid.parquet")
    )
    y_valid: pd.DataFrame = pd.read_parquet(
        os.path.join(path, "y_valid.parquet")
    )
    X_test: pd.DataFrame = pd.read_parquet(
        os.path.join(path, "X_test.parquet")
    )
    y_test: pd.DataFrame = pd.read_parquet(
        os.path.join(path, "y_test.parquet")
    )
    logger.info(f"Loading train, valid and test data from {path} successful")
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def save_data(path: str, **kwargs) -> None:
    if not os.path.exists(path):
        logger.info(f"{path} doesn't exist, creating empty")
        os.mkdir(path)
    logger.info(f"Saving data into {path} started")
    for filename, file in kwargs.items():
        logger.info(f"Saving {filename}.parquet")
        file.to_parquet(os.path.join(path, f"{filename}.parquet"))
    logger.info(f"Saving data into {path} successful")


def update_data_state(field_to_update: str, value: str) -> None:
    with open(get_data_path("data_state"), "r") as f:
        try:
            current_state: Dict[str, str] = json.load(f)
        except:
            logger.info(
                "Error during loading data state. Overwriting by empty"
            )
            current_state: Dict[str, str] = {}

    current_state[field_to_update] = value
    with open(get_data_path("data_state"), "w") as f:
        json.dump(current_state, f, indent=4)


def save_model(
    model,
    model_serialized_path: str,
    model_parameters_path: str,
    get_params_func=None,
):
    save_pickle(object=model, output_path=model_serialized_path)
    if get_params_func is not None:
        params = get_params_func(model)
    else:
        params = model.get_params()
    save_json(dictionary=params, output_path=model_parameters_path)


def save_json(dictionary: Dict[str, str], output_path: str) -> None:
    logger.info(f"Saving dictionary into {output_path}")
    with open(output_path, "w") as f:
        json.dump(dictionary, f, indent=4)
        logger.info(f"Saving dictionary into {output_path} suceeded")


def save_pickle(object: Any, output_path: int) -> None:
    logger.info(f"Saving {object} object to {output_path}")
    with open(output_path, "wb") as f:
        pkl.dump(object, f)
    logger.info(f"Saving to {output_path} succeeded")
