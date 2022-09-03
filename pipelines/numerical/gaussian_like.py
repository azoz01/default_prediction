from typing import Tuple, Union
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer


class ToGaussianTransformer(TransformerMixin):
    def __init__(
        self, columns_to_transform: list[str],
    ):
        self.columns_to_transform = columns_to_transform
        self.column_transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                (
                    "gaussian_like_transformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                    self.columns_to_transform,
                )
            ],
            remainder="drop",
        )

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        self.column_transformer.fit(X)
        return self

    def transform(
        self, X: DataFrame, y: DataFrame = None, **kwargs
    ) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        X[self.columns_to_transform] = self.column_transformer.transform(X)
        if y is not None:
            return X, y
        return X

