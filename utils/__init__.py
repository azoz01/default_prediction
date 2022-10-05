import logging
import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(name)s: %(message)s",
)

__all__ = [
    "get_parameters",
    "get_data_path",
    "get_pipelines_path",
    "get_models_path",
    "load_data",
    "save_data",
    "save_pickle",
]

from .parameters import *
from .io import *
