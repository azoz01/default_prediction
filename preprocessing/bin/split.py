import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_parameters, get_data_path, save_data, constants

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Started splitting data pipeline")
    parameters = get_parameters("split_data")
    input_path = get_data_path("initially_preprocessed")
    output_path = get_data_path("splitted")

    logger.info(f"Loading data from {input_path}")
    raw_data = pd.read_parquet(input_path / "loan.parquet")

    y = raw_data[[constants.TARGET_COLUMN]]
    X = raw_data.drop(columns=constants.TARGET_COLUMN)

    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"] + parameters["valid_size"],
        stratify=y,
        random_state=parameters["random_state"],
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test,
        y_test,
        test_size=parameters["test_size"]
        / (parameters["test_size"] + parameters["valid_size"]),
        stratify=y_test,
        random_state=42,
    )

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

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
