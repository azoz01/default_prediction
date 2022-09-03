import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class OneHotColumnEncoder(TransformerMixin):
    def __init__(self, columns_to_encode: list[str]):
        self.columns_to_encode = columns_to_encode
        self.column_transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("one_hot_encoder", OneHotEncoder(), self.columns_to_encode)
            ],
            remainder="drop",
        )

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        self.column_transformer.fit(X)
        return self

    def transform(self, X: DataFrame, y: DataFrame = None, **kwargs):
        encoded_values: np.ndarray = self.column_transformer.transform(X)
        columns: list[str] = self._get_output_columns(self.column_transformer)
        one_hot_dataframe: DataFrame = DataFrame(
            data=encoded_values, columns=columns
        )
        X = X.drop(columns=self.columns_to_encode)
        X = pd.concat([X, one_hot_dataframe], axis="columns")
        if y is not None:
            return X, y
        return X

    def _get_output_columns(
        self, column_transformer: ColumnTransformer
    ) -> list[str]:
        columns: list[str] = column_transformer.transformers_[0][2]
        all_categories: list[list[str]] = column_transformer.transformers_[0][
            1
        ].categories_
        print(all_categories)
        output_columns: list[str] = []
        for column, categories in zip(columns, all_categories):
            categories_for_column: list[str] = [
                f"{column}_{category}" for category in categories
            ]
            output_columns.extend(categories_for_column)
        return output_columns
