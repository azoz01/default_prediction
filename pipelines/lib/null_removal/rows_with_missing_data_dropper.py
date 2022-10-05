from typing import List, Tuple, Union
import pandas as pd
from sklearn.base import TransformerMixin


class RowsWithMissingDataDropper(TransformerMixin):
    """
    Removes rows with missing values in specified columns
    """

    def __init__(self, columns_to_filter_by: List[str]) -> None:
        """
        Args:
            columns_to_filter_by (List[str]): columns for which
                rows with missing data will be dropped
        """
        self.columns_to_filter_by: List[str] = columns_to_filter_by

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
        Removes rows with missing values in specified columns. If y is None, only X is returned

        Args:
            X (pd.DataFrame): DataFrame to process
            y (pd.DataFrame, optional): DataFrame which is passed through.
                Defaults to None.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
                X without rows with missing data. 
                If y is not none, then passed through
        """
        none_filter_mask: pd.Series = ~(
            X[self.columns_to_filter_by].isnull().any(axis=1)
        )
        if y is not None:
            return (
                X.loc[none_filter_mask].reset_index(drop=True),
                y.loc[none_filter_mask].reset_index(drop=True),
            )
        return X.loc[none_filter_mask].reset_index(drop=True)
