import os
import sys


sys.path.append(os.path.abspath(os.getcwd()))
from typing import Dict, Any
import logging
from pipelines.lib.feature_selection.rfe import RfeAdapter
from pipelines.lib.feature_selection import FRUFSAdapter
from pipelines.lib.dummy import DummyPipeline
from utils.parameters import get_data_path, get_parameters, get_pipelines_path
from pipelines.utils.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)

frufs_params: dict = get_parameters("frufs")
rfe_params: dict = get_parameters("rfe")

DATA_REDUCTION_METHODS: Dict[str, Any] = {
    "frufs": FRUFSAdapter(n_jobs=-1, k=frufs_params["output_n_cols"]),
    "rfe": RfeAdapter(output_n_cols=rfe_params["output_n_cols"]),
    "passthrough": DummyPipeline(),
}


def main():
    logger.info("Started data reduction pipeline")
    input_path: str = get_data_path("transformed_categorical_columns")
    output_path: str = get_data_path("reduced")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "data_reductor.pkl"
    )
    parameters: Dict[str, str] = get_parameters("reduce_data")
    transformer = DATA_REDUCTION_METHODS[parameters["method"]]
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
