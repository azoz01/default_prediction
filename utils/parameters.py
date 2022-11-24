from pathlib import Path
from typing import Dict
import yaml
import logging

logger = logging.getLogger(__name__)
PARAMETERS_FILE_PATH = Path("params.yaml")


def get_parameters(pipeline_name: str) -> Dict[str, str]:
    """
    Returns parameters for specified pipeline
    specified in PARAMETERS_FILE_PATH

    Args:
        pipeline_name (str): name of pipeline

    Returns:
        Dict[str, str]: parameters of this pipeline
    """ ""
    with open(PARAMETERS_FILE_PATH) as f:
        params_dict = yaml.load(f, yaml.CLoader)
    params_for_pipeline = params_dict.get(pipeline_name, {})
    logger.info(
        f"Parameters loaded for {pipeline_name}: {params_for_pipeline}"
    )
    return params_for_pipeline


def get_data_path(path: str) -> Path:
    """
    Returns directory to data specified in params.yaml file

    Args:
        path (str): name of path

    Returns:
        Path: object representing obtained path
    """
    return _get_path(path=path, subpath="data")


def get_preprocessing_path(path: str) -> Path:
    """
    Returns directory to preprocessing resources
    specified in params.yaml file

    Args:
        path (str): name of path

    Returns:
        Path: object representing obtained path
    """
    return _get_path(path=path, subpath="preprocessing")


def get_models_path(path: str) -> Path:
    """
    Returns directory to models specified in params.yaml file

    Args:
        path (str): name of path

    Returns:
        Path: object representing obtained path
    """
    return _get_path(path=path, subpath="models")


def _get_path(path: str, subpath: str) -> Path:
    """
    Gets path by path name and subpath in resources directory

    Args:
        path (str): name of path
        subpath (str): name of subpath

    Returns:
        Path: object representing obtained path
    """
    with open("params.yaml") as f:
        paths = yaml.load(f, yaml.CLoader)["paths"][subpath]
    selected_path = paths[path]
    logger.info(f"Loaded path for {path}: {selected_path}")
    return Path(selected_path)
