import os
import logging
from typing import Any, Dict, Tuple
import pandas as pd
import json
import pickle as pkl

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


def save_data(
    path: str,
    X_train: pd.DataFrame = None,
    y_train: pd.DataFrame = None,
    X_valid: pd.DataFrame = None,
    y_valid: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    y_test: pd.DataFrame = None,
) -> None:
    if not os.path.exists(path):
        logger.info(f"{path} doesn't exist, creating empty")
        os.mkdir(path)
    logger.info(f"Saving data into {path} started")
    if X_train is not None:
        X_train.to_parquet(os.path.join(path, "X_train.parquet"))
    if y_train is not None:
        y_train.to_parquet(os.path.join(path, "y_train.parquet"))
    if X_valid is not None:
        X_valid.to_parquet(os.path.join(path, "X_valid.parquet"))
    if y_valid is not None:
        y_valid.to_parquet(os.path.join(path, "y_valid.parquet"))
    if X_test is not None:
        X_test.to_parquet(os.path.join(path, "X_test.parquet"))
    if y_test is not None:
        y_test.to_parquet(os.path.join(path, "y_test.parquet"))
    logger.info(f"Saving data into {path} successful")


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
