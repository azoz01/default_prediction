import os
import sys


sys.path.append(os.path.abspath(os.getcwd()))
import logging
from utils.parameters import get_data_path, get_pipelines_path
from pipelines.utils.generic_pipeline import GenericPipeline
from pipelines.lib.other import InitialPreprocessor

logger = logging.getLogger(__name__)


def main():
    logger.info("Started initial preprocessing pipeline")
    input_path: str = get_data_path("splitted")
    output_path: str = get_data_path("initially_preprocessed")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "initial_preprocessor.pkl"
    )
    transformer: InitialPreprocessor = InitialPreprocessor()
    pipeline: GenericPipeline = GenericPipeline(
        transformer=transformer,
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
    )
    pipeline.run_pipeline()
    logger.info("Initial preprocessing pipeline completed")


if __name__ == "__main__":
    main()
