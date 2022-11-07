import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
import logging
from typing import Dict, Any

from utils.parameters import get_data_path, get_pipelines_path, get_parameters
from pipelines.lib.null_removal import ModelBasedImputer
from pipelines.lib.dummy import DummyPipeline
from pipelines.lib.built import FixedValueImputationPipeline
from pipelines.utils.generic_pipeline import GenericPipeline
from utils.io import update_data_state

logger = logging.getLogger(__name__)

model_based_parameters = get_parameters("model_based_imputation")

DATA_IMPUTATION_METHODS: Dict[str, Any] = {
    "fixed_value": FixedValueImputationPipeline(),
    "model_based": ModelBasedImputer.create_from_files(
        hyperparams_cat_dict_path=model_based_parameters[
            "hyperparams_cat_dict_path"
        ],
        hyperparams_reg_dict_path=model_based_parameters[
            "hyperparams_reg_dict_path"
        ],
    ),
    "passthrough": DummyPipeline(),
}


def main():
    logger.info("Started cleaning data pipeline")
    input_path: str = get_data_path("initially_preprocessed")
    output_path: str = get_data_path("clean")
    parameters: Dict[str, Any] = get_parameters("clean_data")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "data_cleaner.pkl"
    )
    transformer = DATA_IMPUTATION_METHODS[parameters["method"]]
    pipeline: GenericPipeline = GenericPipeline(
        transformer=transformer,
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
    )
    pipeline.run_pipeline()
    logger.info("Cleaning data pipeline completed")
    update_data_state("clean", parameters["method"])


if __name__ == "__main__":
    main()
