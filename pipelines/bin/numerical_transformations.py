import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
import pandas as pd
import logging

from utils.parameters import get_data_path, get_pipelines_path
from pipelines.lib import constants
from pipelines.lib.built import NumericalTransformations
from pipelines.utils.generic_pipeline import (
    GenericPipeline,
    DataValidator,
    InvalidDataError,
)

logger = logging.getLogger(__name__)


class NumericalDataValidator(DataValidator):
    def validate_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:

        if X_train.isna().sum().sum() != 0:
            raise InvalidDataError("Train sample contains nulls")
        if X_valid.isna().sum().sum() != 0:
            raise InvalidDataError("Validation sample contains nulls")
        if X_test.isna().sum().sum() != 0:
            raise InvalidDataError("Test sample contains nulls")

        if not (
            X_train[constants.NUMERICAL_COLUMNS].mean(axis=0).abs() <= 1e-5
        ).all():
            raise InvalidDataError("Train contains columns with non-zero mean")

        if not (
            (X_train[constants.NUMERICAL_COLUMNS].std(axis=0) - 1).abs()
            <= 1e-5
        ).all():
            raise InvalidDataError("Train contains columns with non-zero std")

        frac_skew_greater_1 = (
            X_train[constants.NUMERICAL_COLUMNS].skew().abs() >= 1
        ).mean()

        frac_skew_greater_2 = (
            X_train[constants.NUMERICAL_COLUMNS].skew().abs() >= 2
        ).mean()

        # TODO: Target is to have zeros here
        logger.info(
            f"Fraction of features with skewness >= 1: {frac_skew_greater_1}"
        )
        logger.info(
            f"Fraction of features with skewness >= 2: {frac_skew_greater_2}"
        )


def main():
    logger.info("Started numerical transformations pipeline")
    input_path: str = get_data_path("clean")
    output_path: str = get_data_path("numerical_transformed")
    serialized_transformer_output_path: str = os.path.join(
        get_pipelines_path("serialized"), "numerical_transformations.pkl"
    )
    pipeline: GenericPipeline = GenericPipeline(
        transformer=NumericalTransformations(),
        input_path=input_path,
        output_path=output_path,
        serialized_transformer_output_path=serialized_transformer_output_path,
        data_validator=NumericalDataValidator(),
    )
    pipeline.run_pipeline()
    logger.info("Numerical transformations pipeline completed")


if __name__ == "__main__":
    main()
