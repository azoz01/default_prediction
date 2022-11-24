import logging
import pandas as pd
from utils.parameters import (
    get_data_path,
    get_preprocessing_path,
    get_parameters,
)
from preprocessing.lib.numerical import NumericalColumnsPreprocessor
from preprocessing.utils.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


def main():
    logger.info("Started transformation of numerical columns pipeline")
    input_path = get_data_path("clean")
    output_path = get_data_path("transformed_numerical_columns")
    serialized_transformer_output_path = (
        get_preprocessing_path("serialized")
        / "numerical_columns_transformer.pkl"
    )
    parameters = get_parameters("transform_numerical_columns")

    pipeline: GenericPipeline = GenericPipeline(
        transformer=NumericalColumnsPreprocessor(**parameters),
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
        validate_data=validate_data,
    )
    pipeline.run_pipeline()
    logger.info("Transformations of numerical columns pipeline completed")


def validate_data(X: pd.DataFrame, y: pd.DataFrame):
    assert (~X.isna()).all(axis=None)


if __name__ == "__main__":
    main()
