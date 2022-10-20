import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
import pandas as pd
import logging

from utils.parameters import get_data_path, get_pipelines_path
from pipelines.lib.built import NumericalTransformations
from pipelines.utils.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


def main():
    logger.info("Started transformation of numerical columns pipeline")
    input_path: str = get_data_path("clean")
    output_path: str = get_data_path("transformed_numerical_columns")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "numerical_columns_transformer.pkl"
    )
    pipeline: GenericPipeline = GenericPipeline(
        transformer=NumericalTransformations(),
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
    )
    pipeline.run_pipeline()
    logger.info("Transformations of numerical columns pipeline completed")


if __name__ == "__main__":
    main()
