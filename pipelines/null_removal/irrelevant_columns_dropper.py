from typing import Tuple, Union
from typing_extensions import Self
from sklearn.base import TransformerMixin
from pandas import DataFrame


class IrrelevantColumnsDropper(TransformerMixin):
    def __init__(self, columns_to_drop: list[str]):
        self.columns_to_drop = columns_to_drop

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        return self

    def transform(
        self, X: DataFrame, y: DataFrame = None, **kwargs
    ) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        X = X.drop(columns=self.columns_to_drop)
        if y:
            return X, y
        else:
            return X
