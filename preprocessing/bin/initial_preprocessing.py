import logging
import pandas as pd
from utils.parameters import get_data_path, get_preprocessing_path
from preprocessing.lib.cleaning import InitialPreprocessor
from utils.io import save_pickle
from utils import constants

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Started initial preprocessing pipeline")
    input_path = get_data_path("raw")
    output_path = get_data_path("initially_preprocessed")
    serialized_transformer_output_path = (
        get_preprocessing_path("serialized") / "initial_preprocessor.pkl"
    )

    logger.info("Reading input data")
    raw_data = pd.read_csv(input_path / "loan.csv")

    logger.info("Processing data")
    transformer: InitialPreprocessor = InitialPreprocessor()
    processed_data = transformer.transform(raw_data)

    logger.info("Validating output data")
    validate_data(processed_data)

    if not output_path.exists():
        output_path.mkdir()

    processed_data.to_parquet(output_path / "loan.parquet")
    save_pickle(transformer, serialized_transformer_output_path)

    logger.info("Initial preprocessing pipeline completed")


def validate_data(X: pd.DataFrame) -> None:
    for column in constants.DATA_TO_DROP:
        assert (
            column not in X.columns
        ), f"{column} shouldn't be present in dataset"


if __name__ == "__main__":
    main()
