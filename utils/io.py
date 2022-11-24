from typing import Any, Callable, Dict, Tuple
import logging
from pathlib import Path
import pickle as pkl
import pandas as pd
import json

logger = logging.getLogger(__name__)


def load_data(
    path: Path,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Loads train, validation and test samples from specified
    directory. Assumes, that data is stored using parquet
    format.

    Args:
        path (Path): path to read data from

    Returns:
        Tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame
        ]: tuple containing read data
    """
    logger.info(f"Loading train, valid and test data from {str(path)} started")
    X_train = pd.read_parquet(path / "X_train.parquet")
    y_train = pd.read_parquet(path / "y_train.parquet")
    X_valid = pd.read_parquet(path / "X_valid.parquet")
    y_valid = pd.read_parquet(path / "y_valid.parquet")
    X_test = pd.read_parquet(path / "X_test.parquet")
    y_test = pd.read_parquet(path / "y_test.parquet")
    logger.info(
        f"Loading train, valid and test data from {str(path)} successful"
    )
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def save_data(path: Path, **kwargs) -> None:
    """
    Saves specified frames into path.
    Frames are passed to **kwargs, so can pass
    any number of frames. Saves data to parquet
    format

    Args:
        path (Path): path to save
    """
    if not path.exists():
        logger.info(f"{str(path)} doesn't exist, creating empty")
        path.mkdir()
    logger.info(f"Saving data into {str(path)} started")
    for filename, frame in kwargs.items():
        logger.info(f"Saving {filename}.parquet")
        frame.to_parquet(path / f"{filename}.parquet")
    logger.info(f"Saving data into {str(path)} successful")


def save_model(
    model: Any,
    model_serialized_path: Path,
    model_parameters_path: Path,
    get_params_func: Callable = None,
) -> None:
    """
    Serializes model and saves its parameters to json.

    Args:
        model (Any): model object
        model_serialized_path (str): path to save model as pickle
        model_parameters_path (str): path to save model params as json
        get_params_func (Callable, optional): function which retrieves
            parameters from model. If not provided, then model.get_params()
            function is used.
    """
    save_pickle(object=model, output_path=model_serialized_path)
    if get_params_func is not None:
        params = get_params_func(model)
    else:
        params = model.get_params()
    save_json(dictionary=params, output_path=model_parameters_path)


def read_json(input_path: Path) -> None:
    """
    Loads dictionary from json

    Args:
        dictionary (Dict[str, str]): dictionary to save
        output_path (str): path to save to
    """
    logger.info(f"Loading dictionary from {input_path}")
    with open(input_path, "r") as f:
        return json.load(f)


def save_json(dictionary: Dict[str, str], output_path: Path) -> None:
    """
    Saves dictionary to json

    Args:
        dictionary (Dict[str, str]): dictionary to save
        output_path (str): path to save to
    """
    logger.info(f"Saving dictionary into {output_path}")
    with open(output_path, "w") as f:
        json.dump(dictionary, f, indent=4)
        logger.info(f"Saving dictionary into {output_path} suceeded")


def load_pickle(input_path: Path) -> Any:
    """
    Loads object from pickle

    Args:
        input_path (int): path to load data from
    """
    logger.info(f"loading pickle from {input_path}")
    with open(input_path, "rb") as f:
        return pkl.load(f)


def save_pickle(object: Any, output_path: Path) -> None:
    """
    Saves object to pickle file

    Args:
        object (Any): object to serialize
        output_path (int): path to save to
    """
    logger.info(f"Saving {object} object to {output_path}")
    with open(output_path, "wb") as f:
        pkl.dump(object, f)
    logger.info(f"Saving to {output_path} succeeded")
