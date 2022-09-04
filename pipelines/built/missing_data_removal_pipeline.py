from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from pipelines.null_removal import *
import pipelines.constants as constants


class MissingDataRemovalPipeline(TransformerMixin):
    def __init__(self):
        self.missing_data_removal_pipeline: Pipeline = Pipeline(
            [
                (
                    "irrelevant_columns_dropper",
                    IrrelevantColumnsDropper(
                        columns_to_drop=constants.COLUMNS_TO_DROP
                    ),
                ),
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

        self.rows_with_missing_data_dropper = RowsWithMissingDataDropper(
            columns_to_filter_by=constants.COLUMNS_TO_FILTER_ROWS_WITH_NULLS
        )

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        self.missing_data_removal_pipeline.fit(X)
        return self

    def transform(self, X: DataFrame, y: DataFrame = None, **kwargs):
        # TODO: refactor somehow
        X = self.missing_data_removal_pipeline.transform(X)
        if y is not None:
            X, y = self.rows_with_missing_data_dropper.transform(X, y)
            return X, y
        X = self.rows_with_missing_data_dropper.transform(X)
        return X

