import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
import pandas as pd
import logging

from utils.parameters import get_data_path, get_pipelines_path
from pipelines.lib.built import MissingDataRemovalPipeline
from pipelines.utils.generic_pipeline import (
    GenericPipeline,
    DataValidator,
    InvalidDataError,
)

logger = logging.getLogger(__name__)


class MissingDataValidator(DataValidator):
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

    def _validate_sample(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise InvalidDataError("Different lengths of X and y")

        if X.isna().sum().sum() != 0:
            raise InvalidDataError("Null inside data")


def main():
    logger.info("Started missing data removal pipeline")
    input_path: str = get_data_path("splitted")
    output_path: str = get_data_path("clean")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "clean_data.pkl"
    )
    pipeline: GenericPipeline = GenericPipeline(
        transformer=MissingDataRemovalPipeline(),
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
        data_validator=MissingDataValidator(),
    )
    pipeline.run_pipeline()
    logger.info("Missing data removal pipeline completed")


if __name__ == "__main__":
    main()
