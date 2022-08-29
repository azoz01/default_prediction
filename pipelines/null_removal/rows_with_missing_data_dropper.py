from typing import Tuple, Union
from sklearn.base import TransformerMixin
from pandas import DataFrame


class RowsWithMissingDataDropper(TransformerMixin):
    def __init__(self, columns_to_filter_by: list[str]):
        self.columns_to_filter_by = columns_to_filter_by

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        return self

    def transform(
        self, X: DataFrame, y: DataFrame = None, **kwargs
    ) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        none_filter_mask = ~(X[self.columns_to_filter_by].isnull().any(axis=1))
        if y is not None:
            return (
                X.loc[none_filter_mask].reset_index(drop=True),
                y.loc[none_filter_mask].reset_index(drop=True),
            )
        return X.loc[none_filter_mask].reset_index(drop=True)
