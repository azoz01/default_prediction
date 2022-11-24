import logging
import pickle as pkl
from utils.parameters import get_parameters, get_data_path
from utils.io import load_data
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


def main():
    parameters = get_parameters("calibrate_model")
    model_input_path = parameters["model_input_path"]
    data_path = get_data_path(parameters["data_path"])
    model_output_path = parameters["model_output_path"]

    logger.info(f"Loading input model from {model_input_path}")
    with open(model_input_path, "rb",) as f:
        model = pkl.load(f)

    logger.info("Loading data for calibration")

    _, _, X_valid, y_valid, _, _ = load_data(data_path)

    logger.info("Calibrating model")
    calibrated = CalibratedClassifierCV(
        model, cv="prefit", method=parameters["calibration_method"]
    )
    calibrated.fit(X_valid, y_valid)

    logger.info(f"Saving calibrated model to {model_output_path}")
    with open(model_output_path, "wb",) as f:
        model = pkl.dump(calibrated, f)


if __name__ == "__main__":
    main()
