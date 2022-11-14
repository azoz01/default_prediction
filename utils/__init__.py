import logging
import os

__all__ = [
    "get_parameters",
    "get_data_path",
    "get_preprocessing_path",
    "get_models_path",
    "load_data",
    "save_data",
    "save_pickle",
]

from .parameters import *
from .io import *

# Logging init
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(name)s: %(message)s",
)

# MlFlow init
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "./secured/service_account_key.json"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5000/"
os.environ["MLFLOW_EXPERIMENT_NAME"] = get_current_mlflow_experiment()
