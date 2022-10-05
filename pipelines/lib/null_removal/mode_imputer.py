from typing import Any, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class ModeImputer(TransformerMixin):
    """
    Imputes missing columns with specified columns with modes
    """

    def __init__(
        self, columns_to_impute: List[str], missing_values: Any = np.nan
    ) -> None:
        """
        Args:
            columns_to_impute (List[str]): columns to impute data
            missing_values (Any, optional): Values considered as missing. 
                Defaults to np.nan.
        """
        self.columns_to_impute: List[str] = columns_to_impute
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

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        """
        Fits pipeline

        Args:
            X (DataFrame): X to fit
            y (DataFrame, optional): y to train. Defaults to None.

        """
        self.column_transformer.fit(X)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Replaces missing data in specified columns with modes

        Args:
            X (DataFrame): X to transform
            y (DataFrame, optional): y to transform, if not None, then
                passed through. Defaults to None.

        Returns:
            Union[DataFrame, Tuple[DataFrame, DataFrame]]: Data with 
                imputed values. If y is not None, then passed through.
        """
        X = X.copy()
        X[self.columns_to_impute] = self.column_transformer.transform(X)
        if y is not None:
            return X, y
        return X
