from typing import Any, List, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class FixedValueImputer(TransformerMixin):
    """
    Replaces nas inside specified column with fixed value
    """

    def __init__(
        self,
        columns_to_impute: List[str],
        fill_value: Any,
        missing_values: Any = np.nan,
    ) -> None:
        """
        Args:
            columns_to_impute (List[str]): columns to impute missing values
            fill_value (Any): Value which replaces nas
            missing_values (Any, optional): Values considered as missing. 
                Defaults to np.nan.
        """
        self.columns_to_impute: List[str] = columns_to_impute
        self.column_transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                (
                    "fixed_value_imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=fill_value,
                        missing_values=missing_values,
                    ),
                    self.columns_to_impute,
                )
            ],
            remainder="drop",
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        """
        Placeholder method for fit

        Args:
            X (pd.DataFrame)
            y (pd.DataFrame, optional)
        """
        self.column_transformer.fit(X)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Replaces missing values from specified columns with self.fill_value

        Args:
            X (pd.DataFrame): DataFrame to process
            y (pd.DataFrame, optional): DataFrame which is passed through.
                Defaults to None.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
                X with imputed columns. If y is not none, then passed through
        """
        X = X.copy()
        X[self.columns_to_impute] = self.column_transformer.transform(X)
        if y is not None:
            return X, y
        return X

