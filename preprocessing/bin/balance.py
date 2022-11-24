import logging

from utils.parameters import get_data_path, get_parameters
from utils.io import load_data, save_data
from preprocessing.lib.factory.balance_data import OversampleFactory

logger = logging.getLogger(__name__)


def main():
    logger.info("Started balancing data pipeline")
    input_path = get_data_path("transformed_numerical_columns")
    output_path = get_data_path("balanced")
    parameters = get_parameters("balance_data")

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        path=input_path
    )
    logger.info("Balancing train data")
    method = parameters["method"]
    method_parameters = get_parameters(method)
    oversample = OversampleFactory().create_oversampler(
        method, method_parameters
    )
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    save_data(
        path=output_path,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
