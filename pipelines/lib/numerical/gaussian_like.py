from typing import List, Tuple, Union
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer


class ToGaussianTransformer(TransformerMixin):
    """
    Transforms numerical columns to be more Gaussian-like
    in terms of distribution.
    """

    def __init__(self, columns_to_transform: List[str],) -> None:
        """
        Args:
            columns_to_transform (List[str]): columns to process
        """
        self.columns_to_transform: List[str] = columns_to_transform
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

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        """
        Fits pipeline

        Args:
            X (pd.DataFrame): X to fit
            y (pd.DataFrame, optional): y to fit. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.column_transformer.fit(X)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Transforms numerical columns to be more Gaussian-like
        in terms of distribution.

        Args:
            X (pd.DataFrame): X to transform
            y (pd.DataFrame, optional): DataFrame which is passed through.
                Defaults to None.


        Returns:
            Union[DataFrame, Tuple[DataFrame, DataFrame]]: DataFrame with 
                processed numerical columns. If y is not None, then passed through.
        """
        X[self.columns_to_transform] = self.column_transformer.transform(X)
        if y is not None:
            return X, y
        return X

