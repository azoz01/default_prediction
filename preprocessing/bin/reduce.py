from typing import Dict
import logging
from utils.parameters import (
    get_data_path,
    get_parameters,
    get_preprocessing_path,
)
from preprocessing.utils.generic_pipeline import GenericPipeline
from preprocessing.lib.factory.feature_selection import FeatureSelectionFactory

logger = logging.getLogger(__name__)


def main():
    logger.info("Started data reduction pipeline")
    input_path: str = get_data_path("transformed_categorical_columns")
    output_path: str = get_data_path("reduced")
    serialized_transformer_output_path: str = (
        get_preprocessing_path("serialized") / "data_reductor.pkl"
    )
    parameters: Dict[str, str] = get_parameters("reduce_data")
    method = parameters["method"]
    method_parameters = get_parameters(method)
    transformer = FeatureSelectionFactory().create_oversampler(
        method, method_parameters
    )
    pipeline: GenericPipeline = GenericPipeline(
        transformer=transformer,
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
    )
    pipeline.run_pipeline()
    logger.info("Data reduction pipeline completed")


if __name__ == "__main__":
    main()
