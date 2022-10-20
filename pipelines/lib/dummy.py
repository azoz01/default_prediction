from sklearn.base import TransformerMixin
from FRUFS import FRUFS
import pandas as pd


class DummyPipeline(TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        if y is not None:
            return X, y
        return X
