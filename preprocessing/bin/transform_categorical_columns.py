import logging
from typing import Dict
import pandas as pd
from pandas.api.types import is_numeric_dtype
from utils.parameters import (
    get_data_path,
    get_parameters,
    get_preprocessing_path,
)
from preprocessing.lib.factory.categorical import CategoryEncodingFactory
from preprocessing.utils.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


def main():
    logger.info("Started categorical columns transformations pipeline")
    input_path = get_data_path("transformed_numerical_columns")
    output_path = get_data_path("transformed_categorical_columns")
    serialized_transformer_output_path = (
        get_preprocessing_path("serialized")
        / "categorical_columns_transformer.pkl"
    )
    parameters: Dict[str, str] = get_parameters(
        "transform_categorical_columns"
    )
    method = parameters["method"]
    method_parameters = get_parameters(method)
    transformer = CategoryEncodingFactory().create_oversampler(
        method, method_parameters
    )
    pipeline: GenericPipeline = GenericPipeline(
        transformer=transformer,
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
        validate_data=validate_data,
    )
    pipeline.run_pipeline()
    logger.info("Categorical columns transformations pipeline completed")


def validate_data(X: pd.DataFrame, y: pd.DataFrame = None) -> None:
    assert X.apply(is_numeric_dtype).all()


if __name__ == "__main__":
    main()
