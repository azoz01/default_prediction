from __future__ import annotations
from typing import List, Tuple, Union
import re
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from utils import constants


class OneHotColumnEncoder(TransformerMixin):
    """
    Encodes specified categorical columns using one-hot encoding
    """

    def __init__(self) -> None:
        self.columns_to_encode = constants.STANDARD_CAT_COLUMNS
        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "one_hot_encoder",
                    OneHotEncoder(sparse=False),
                    self.columns_to_encode,
                )
            ],
            remainder="drop",
        )

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> OneHotColumnEncoder:
        self.column_transformer.fit(X)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        X = X.copy()
        encoded_values = self.column_transformer.transform(X)
        columns = self._get_output_columns(self.column_transformer)
        one_hot_dataframe = pd.DataFrame(data=encoded_values, columns=columns)
        X = X.drop(columns=self.columns_to_encode)
        X = pd.concat([X, one_hot_dataframe], axis="columns")
        print(X.shape)
        if y is not None:
            return X, y
        return X

    def _get_output_columns(
        self, column_transformer: ColumnTransformer
    ) -> List[str]:
        columns = column_transformer.transformers_[0][2]
        all_categories = column_transformer.transformers_[0][1].categories_
        output_columns = []
        for column, categories in zip(columns, all_categories):
            categories_for_column = [
                f"{column}_{category}" for category in categories
            ]
            output_columns.extend(categories_for_column)
        output_columns = list(
            map(self._remove_forbidden_characters, output_columns)
        )
        return output_columns

    def _remove_forbidden_characters(self, string: str) -> str:
        string = re.sub(",", "(COMMA)", string)
        string = re.sub("<", "(LEQ)", string)
        return string
