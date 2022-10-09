import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
import pandas as pd
import logging

from utils.parameters import get_data_path, get_parameters, get_pipelines_path
from pipelines.lib.categorical import CategoryEmbedder
from pipelines.utils.generic_pipeline import (
    GenericPipeline,
    DataValidator,
    InvalidDataError,
)

logger = logging.getLogger(__name__)


class CategoryEmbeddingDataValidator(DataValidator):
    def validate_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        self._validate_sample(X_train, y_train)
        self._validate_sample(X_valid, y_valid)
        self._validate_sample(X_test, y_test)

    def _validate_sample(self, X: pd.DataFrame, y: pd.DataFrame):
        if X.isna().sum().sum() != 0:
            raise InvalidDataError("Train sample contains nulls")
        if len(X.select_dtypes(exclude=["float32"]).columns) != 0:
            raise InvalidDataError(
                "There are non-numerical columns in train sample"
            )


def main():
    logger.info("Started category embedding pipeline")
    input_path: str = get_data_path("numerical_transformed")
    output_path: str = get_data_path("categorical_embedded")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "category_embedding.pkl"
    )
    params: dict = get_parameters("category_embedding")
    pipeline: GenericPipeline = GenericPipeline(
        transformer=CategoryEmbedder(
            embedder_n_epochs=params["embedder_n_epochs"],
            embedder_num_layers=params["embedder_num_layers"],
        ),
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
        data_validator=CategoryEmbeddingDataValidator(),
    )
    pipeline.run_pipeline()
    logger.info("Categorty embedding pipeline completed.")


if __name__ == "__main__":
    main()
