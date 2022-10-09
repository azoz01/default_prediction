from typing import List, Tuple, Union
import pandas as pd
from sklearn.base import TransformerMixin


class IrrelevantColumnsDropper(TransformerMixin):
    """
    Drops missing data from DataFrame
    """

    def __init__(self, columns_to_drop: List[str]) -> None:
        """
        Args:
            columns_to_drop (List[str]): columns to drop
        """
        self.columns_to_drop: List[str] = columns_to_drop

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        """
        Placeholder method for fit

        Args:
            X (pd.DataFrame)
            y (pd.DataFrame, optional)
        """
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Drops specified columns from X. If y is None, only X is returned

        Args:
            X (pd.DataFrame): DataFrame to process
            y (pd.DataFrame, optional): DataFrame which is passed through.
                Defaults to None.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
                X without specified columns. If y is not none, then passed through
        """
        X = X.drop(columns=self.columns_to_drop)
        if y is not None:
            return X, y
        return X
