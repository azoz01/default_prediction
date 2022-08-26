from typing import Tuple, Union
from sklearn.base import TransformerMixin
from pandas import DataFrame


class RowsWithMissingDataDropper(TransformerMixin):

    COLUMNS_TO_FILTER = [
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_FAM_MEMBERS",
    ]

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        return self

    def transform(
        self, X: DataFrame, y: DataFrame = None, **kwargs
    ) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        none_filter_mask = ~(X[self.COLUMNS_TO_FILTER].isna().any(axis=1))
        if y is not None:
            return (
                X[none_filter_mask].reset_index(drop=True),
                y[none_filter_mask].reset_index(drop=True),
            )
        return X.loc[~none_filter_mask].reset_index(drop=True)
