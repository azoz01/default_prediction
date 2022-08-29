import numpy as np
from typing import Any, Tuple, Union
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class ModeImputer(TransformerMixin):
    def __init__(
        self, columns_to_impute: list[str], missing_values: Any = np.nan
    ):
        self.columns_to_impute = columns_to_impute
        self.column_transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                (
                    "mode_imputer",
                    SimpleImputer(
                        strategy="most_frequent", missing_values=missing_values
                    ),
                    self.columns_to_impute,
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
        X[self.columns_to_impute] = self.column_transformer.transform(X)
        if y is not None:
            return X, y
        return X
