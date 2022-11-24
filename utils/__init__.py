import logging
import os
from .parameters import (
    get_parameters,
    get_data_path,
    get_preprocessing_path,
    get_models_path,
)
from .io import load_data, save_data, save_pickle

__all__ = [
    "get_parameters",
    "get_data_path",
    "get_preprocessing_path",
    "get_models_path",
    "load_data",
    "save_data",
    "save_pickle",
]

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(name)s: %(message)s",
)
