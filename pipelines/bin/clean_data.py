import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
import pandas as pd
import logging

from utils.parameters import get_data_path, get_pipelines_path
from pipelines.lib.built import MissingDataRemovalPipeline
from pipelines.utils.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


def main():
    logger.info("Started cleaning data pipeline")
    input_path: str = get_data_path("splitted")
    output_path: str = get_data_path("clean")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "data_cleaner.pkl"
    )
    pipeline: GenericPipeline = GenericPipeline(
        transformer=MissingDataRemovalPipeline(),
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
    )
    pipeline.run_pipeline()
    logger.info("Cleaning data pipeline completed")


if __name__ == "__main__":
    main()
