from typing import Dict
import yaml
import logging

logger = logging.getLogger(__name__)


def get_parameters(pipeline_name: str) -> Dict[str, str]:
    with open("params.yaml") as f:
        params_dict: Dict[str, Dict[str, str]] = yaml.load(f, yaml.CLoader)
    params_for_pipeline: Dict[str, str] = params_dict[pipeline_name]
    logger.info(f"Parameters for {pipeline_name}: {params_for_pipeline}")
    return params_for_pipeline


def get_data_path(path: str) -> str:
    return _get_path(path=path, subpath="data")


def get_pipelines_path(path: str) -> str:
    return _get_path(path=path, subpath="pipelines")


def get_models_path(path: str) -> str:
    return _get_path(path=path, subpath="models")


def _get_path(path: str, subpath: str) -> str:
    with open("params.yaml") as f:
        paths: Dict[str, str] = yaml.load(f, yaml.CLoader)["paths"][subpath]
    selected_path: str = paths[path]
    logger.info(f"{path}: {selected_path}")
    return selected_path
