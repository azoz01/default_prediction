import os
import sys


sys.path.append(os.path.abspath(os.getcwd()))
import logging
from typing import Dict, Any
from utils.parameters import get_data_path, get_parameters, get_pipelines_path
from utils.io import update_data_state
from pipelines.lib import constants
from pipelines.lib.dummy import DummyPipeline
from pipelines.lib.categorical import OneHotColumnEncoder, CategoryEmbedder
from pipelines.utils.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)

category_embedding_params: Dict[str, str] = get_parameters(
    "category_embedding"
)

CATEGORICAL_COLUMNS_TRANSFORMATION_METHODS: Dict[str, Any] = {
    "one_hot": OneHotColumnEncoder(
        columns_to_encode=constants.CATEGORICAL_COLUMNS
        + constants.ONE_HOT_ENCODED
    ),
    "category_embedding": CategoryEmbedder(
        embedder_n_epochs=category_embedding_params["embedder_n_epochs"],
        embedder_num_layers=category_embedding_params["embedder_num_layers"],
    ),
    "passthrough": DummyPipeline(),
}


def main():
    logger.info("Started categorical columns transformations pipeline")
    input_path: str = get_data_path("transformed_numerical_columns")
    output_path: str = get_data_path("transformed_categorical_columns")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "categorical_columns_transformer.pkl"
    )
    parameters: Dict[str, str] = get_parameters(
        "transform_categorical_columns"
    )
    transformer = CATEGORICAL_COLUMNS_TRANSFORMATION_METHODS[
        parameters["method"]
    ]
    pipeline: GenericPipeline = GenericPipeline(
        transformer=transformer,
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
    )
    pipeline.run_pipeline()
    update_data_state("transform_categorical_columns", parameters["method"])
    logger.info("Categorical columns transformations pipeline completed")


if __name__ == "__main__":
    main()
