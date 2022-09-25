import sys
import os


sys.path.append(os.path.abspath(os.getcwd()))

import argparse
import pandas as pd
import pickle as pkl
import utils.paths as paths
from utils.logging import pipeline_logger
from sklearn.calibration import CalibratedClassifierCV


def main():
    sys_arguments = get_sys_arguments()
    pipeline_logger.info(f"Loading sys arguments: {sys_arguments}")
    input_model_path = os.path.join(
        paths.MODELS_SERIALIZED_PATH, sys_arguments["model"] + ".pkl"
    )
    output_modeL_path = os.path.join(
        paths.MODELS_SERIALIZED_PATH,
        sys_arguments["model"] + "_calibrated.pkl",
    )
    pipeline_logger.info(f"Loading input model from {input_model_path}")
    with open(input_model_path, "rb",) as f:
        model = pkl.load(f)

    pipeline_logger.info(
        f"Loading calibration data from {paths.DATA_TO_TRAIN_PATH}"
    )
    X_calib = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "X_calib.parquet")
    )
    y_calib = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "y_calib.parquet")
    )

    pipeline_logger.info("Calibrating model")
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X_calib, y_calib)

    pipeline_logger.info(f"Saving calibrated model to {output_modeL_path}")
    with open(output_modeL_path, "wb",) as f:
        model = pkl.dump(calibrated, f)


def get_sys_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Name of file of model inside model directory (without extension)",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main()
