import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))


import mlflow
import utils

mlflow.start_run()

mlflow.log_param("param1", 1)
mlflow.log_metric("accuracy", 1)
mlflow.log_artifact("resources/models/serialized/gaussian_nb_calibrated.pkl")

mlflow.end_run()
