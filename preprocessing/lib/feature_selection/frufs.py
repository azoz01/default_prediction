from sklearn.base import TransformerMixin
from FRUFS import FRUFS
import pandas as pd


class FRUFSAdapter(TransformerMixin):
    def __init__(self, **kwargs):
        kwargs["k"] = kwargs.pop("output_n_cols")
        self.frufs: FRUFS = FRUFS(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        self.frufs.fit(X, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        X = self.frufs.transform(X, **kwargs)
        if y is not None:
            return X, y
        return X
