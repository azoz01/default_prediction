import logging
import pandas as pd

from utils.parameters import get_data_path, get_preprocessing_path
from preprocessing.lib.cleaning import Imputer
from preprocessing.utils.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Started cleaning data pipeline")
    input_path = get_data_path("splitted")
    output_path = get_data_path("clean")
    serialized_transformer_output_path = (
        get_preprocessing_path("serialized") / "imputer.pkl"
    )
    imputer = Imputer()
    pipeline: GenericPipeline = GenericPipeline(
        transformer=imputer,
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
        validate_data=validate_data,
    )
    pipeline.run_pipeline()
    logger.info("Cleaning data pipeline completed")


def validate_data(X: pd.DataFrame, y: pd.DataFrame = None) -> None:
    assert (~X.isna()).all(axis=None), "Data contains nulls"


if __name__ == "__main__":
    main()
