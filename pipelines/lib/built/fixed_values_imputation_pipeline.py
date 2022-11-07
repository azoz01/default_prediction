from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from ..null_removal import *
from .. import constants


class FixedValueImputationPipeline(TransformerMixin):
    """
    Pipeline responsible for removal missing data. It applies:
        * median imputation
        * imputation of 0 and -1
        * mode imputation
    """

    def __init__(self):
        self.fixed_value_imputation_pipeline: Pipeline = Pipeline(
            [
                (
                    "median_imputer",
                    MedianImputer(
                        columns_to_impute=constants.COLUMNS_TO_IMPUTE_MEDIAN
                    ),
                ),
                (
                    "fixed_value_imputer_0",
                    FixedValueImputer(
                        columns_to_impute=constants.COLUMNS_TO_IMPUTE_0,
                        fill_value=0,
                    ),
                ),
                (
                    "fixed_value_imputer_-1",
                    FixedValueImputer(
                        columns_to_impute=constants.COLUMNS_TO_IMPUTE_NEG_1,
                        fill_value=-1,
                    ),
                ),
                (
                    "missing_category_imputer",
                    FixedValueImputer(
                        columns_to_impute=constants.COLUMNS_TO_IMPUTE_MISSING_CATEGORY,
                        fill_value="Missing",
                        missing_values=None,
                    ),
                ),
                (
                    "mode_imputer",
                    ModeImputer(
                        columns_to_impute=constants.COLUMNS_TO_IMPUTE_MODE,
                        missing_values=None,
                    ),
                ),
            ]
        )

        self.rows_with_missing_data_dropper: RowsWithMissingDataDropper = (
            RowsWithMissingDataDropper(
                columns_to_filter_by=constants.COLUMNS_TO_FILTER_ROWS_WITH_NULLS
            )
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        """
        Fits pipeline

        Args:
            X (pd.DataFrame): X to fit
            y (pd.DataFrame, optional): y to fit. Defaults to None.
        """
        self.fixed_value_imputation_pipeline.fit(X)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms data according to pipeline logic.

        Args:
            X (pd.DataFrame): X to transform
            y (pd.DataFrame): y to transform

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Transformed X and y
        """
        X = self.fixed_value_imputation_pipeline.transform(X)
        if y is None:
            return X
        X, y = self.rows_with_missing_data_dropper.transform(X, y)
        return X, y

