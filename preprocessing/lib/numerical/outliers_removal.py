from __future__ import annotations
from typing import Union, Tuple
from scipy import stats
import pandas as pd
from sklearn.base import TransformerMixin


class OutliersCutter(TransformerMixin):
    """
    Removes outliers from specified numerical columns.
    As outliers we consider median +/- 1.5*IQR
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns_to_cut_outliers = columns

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> OutliersCutter:
        self.outliers_thresholds = {}
        for column in self.columns_to_cut_outliers:
            self.outliers_thresholds[column] = self._get_outliers_thresholds(
                X[column]
            )
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        X = X.copy()
        for column in self.columns_to_cut_outliers:
            lower_bound, upper_bound = self.outliers_thresholds[column]
            X[column] = self._cut_outliers(X[column], lower_bound, upper_bound)
        if y is not None:
            return X, y
        return X

    def _get_outliers_thresholds(
        self, column: pd.Series
    ) -> Tuple[float, float]:
        column = column.copy()
        column = column.dropna()
        iqr = stats.iqr(column, nan_policy="raise")
        median = column.median()
        lower_bound = median - 1.5 * iqr
        lower_bound = max(lower_bound, 0)
        upper_bound = median + 1.5 * iqr
        upper_bound = max(upper_bound, 0)
        return lower_bound, upper_bound

    def _cut_outliers(
        self, column: pd.Series, lower_bound: float, upper_bound: float
    ) -> pd.Series:
        column = column.copy()
        column.loc[column < lower_bound] = lower_bound
        column.loc[column > upper_bound] = upper_bound
        return column

    def get_params(*args, **kwargs) -> dict:
        return {}
