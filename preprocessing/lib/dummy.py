from __future__ import annotations
from typing import Tuple, Union
from sklearn.base import TransformerMixin
import pandas as pd


class DummyPipeline(TransformerMixin):
    """
    Placeholder for pipeline
    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> DummyPipeline:
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        if y is not None:
            return X, y
        return X

    def fit_resample(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return X, y
