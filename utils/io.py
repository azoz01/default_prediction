from pathlib import Path
import logging
from typing import Any, Callable, Dict, Tuple
import pickle as pkl
import pandas as pd
import json
import mlflow

from utils.parameters import get_data_path

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


def update_data_state(field_to_update: str, value: str) -> None:
    """
    Updates field in data state file. If data state
    field doesn't exist, then new is created.

    Args:
        field_to_update (str): field in data state.
            If not present, then field is created
        value (str): new value of specified field
    """
    with open(get_data_path("data_state"), "r") as f:
        try:
            current_state = json.load(f)
        except Exception as e:
            logger.warning(
                "Error during loading data state. Overwriting by empty"
            )
            logger.warning(e)
            current_state = {}

    current_state[field_to_update] = value
    with open(get_data_path("data_state"), "w") as f:
        json.dump(current_state, f, indent=4)


def get_data_state() -> Dict[str, str]:
    """
    Gets dictionary from data state file.

    Returns:
        Dict[str, str]: Field containing data state
            read from file.
    """
    with open(get_data_path("data_state"), "r") as f:
        return json.load(f)


def get_current_mlflow_experiment() -> str:
    """
    Returns name of current mlflow experiment from config file.
    If such experiment doesn't exist, then new is created.

    Returns:
        str: name of current experiment
    """
    with open("mlflow/conf.json", "r") as f:
        current_experiment = json.load(f)["current_experiment"]
        mlflow_client = mlflow.tracking.MlflowClient()
        if not mlflow_client.get_experiment_by_name(name=current_experiment):
            logger.info(
                f"Experiment with name {current_experiment} "
                "doesn't exist. Creating new one"
            )
            mlflow.create_experiment(name=current_experiment)
        return current_experiment


def save_model(
    model: Any,
    model_serialized_path: str,
    model_parameters_path: str,
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


def save_json(dictionary: Dict[str, str], output_path: str) -> None:
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


def save_pickle(object: Any, output_path: int) -> None:
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
