from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class OneHotColumnEncoder(TransformerMixin):
    """
    Encodes specified categorical columns using one-hot encoding
    """

    def __init__(self, columns_to_encode: List[str]) -> None:
        """
        Args:
            columns_to_encode (List[str]): categorical columns 
                to one-hot encode
        """
        self.columns_to_encode: List[str] = columns_to_encode
        self.column_transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                (
                    "one_hot_encoder",
                    OneHotEncoder(sparse=False),
                    self.columns_to_encode,
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
        """
        self.column_transformer.fit(X)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Removes specified categorical columns from DataFrame. Afterwards
        adds to end of X one-hot encoded values in same order, as they were
        specified. Name of columns is like {column_name}_{column_value}

        Args:
            X (pd.DataFrame): X to transform
            y (pd.DataFrame, optional): DataFrame which is passed through.
                Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with removed specified 
                columns and their one-hot representations
        """
        X = X.copy()
        encoded_values: np.ndarray = self.column_transformer.transform(X)
        columns: List[str] = self._get_output_columns(self.column_transformer)
        one_hot_dataframe: pd.DataFrame = pd.DataFrame(
            data=encoded_values, columns=columns
        )
        X = X.drop(columns=self.columns_to_encode)
        X = pd.concat([X, one_hot_dataframe], axis="columns")
        if y is not None:
            return X, y
        return X

    def _get_output_columns(
        self, column_transformer: ColumnTransformer
    ) -> List[str]:
        """
        Returns columns' names of one-hot encoded part of input DataFrame
        using pattern {source_column_name}_{column_value}

        Args:
            column_transformer (ColumnTransformer): column transformer
                which wraps OneHotEncoder

        Returns:
            List[str]: List of columns' names
        """
        columns: List[str] = column_transformer.transformers_[0][2]
        all_categories: List[List[str]] = column_transformer.transformers_[0][
            1
        ].categories_
        output_columns: List[str] = []
        for column, categories in zip(columns, all_categories):
            categories_for_column: List[str] = [
                f"{column}_{category}" for category in categories
            ]
            output_columns.extend(categories_for_column)
        return output_columns
