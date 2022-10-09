import os
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.getcwd()))
import logging
import pandas as pd
from pipelines.lib.feature_selection import FRUFSAdapter
from utils.parameters import get_data_path, get_parameters, get_pipelines_path
from pipelines.utils.generic_pipeline import (
    GenericPipeline,
    DataValidator,
    InvalidDataError,
)

logger = logging.getLogger(__name__)


class FrufsDataValidator(DataValidator):
    def validate_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:

        params: dict = get_parameters("frufs")

        self._validate_sample(X_train, y_train, params)
        self._validate_sample(X_valid, y_valid, params)
        self._validate_sample(X_test, y_test, params)

    def _validate_sample(self, X: pd.DataFrame, y: pd.DataFrame, params):
        if X.isna().sum().sum() != 0:
            raise InvalidDataError("Sample contains nulls")
        if len(X.select_dtypes(exclude=["float32", "float64"]).columns) != 0:
            raise InvalidDataError("There are non-numerical columns in sample")
        if X.shape[1] != params["output_n_cols"]:
            raise InvalidDataError("Shape of sample is improper")


def main():
    logger.info("Started FRUFS pipeline")
    input_paths_dict: Dict[str, str] = {
        "categorical_transformed": get_data_path("categorical_transformed"),
        "categorical_embedded": get_data_path("categorical_embedded"),
    }
    params: dict = get_parameters("frufs")
    input_path: str = input_paths_dict[params["input"]]
    output_path: str = get_data_path("reduced")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "frufs.pkl"
    )
    pipeline: GenericPipeline = GenericPipeline(
        transformer=FRUFSAdapter(n_jobs=-1, k=params["output_n_cols"]),
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
        data_validator=FrufsDataValidator(),
    )
    pipeline.run_pipeline()
    logger.info("Categorty embedding pipeline completed")


if __name__ == "__main__":
    main()
