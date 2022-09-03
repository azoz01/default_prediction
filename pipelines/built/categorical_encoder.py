from pandas import DataFrame
from pipelines.categorical import OneHotColumnEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from pipelines.null_removal import *
from pipelines.numerical import *
import pipelines.constants as constants


class CategoricalEncoder(TransformerMixin):
    def __init__(self):
        self.categorical_encoder = Pipeline(
            [
                (
                    "one_hot_encoder",
                    OneHotColumnEncoder(
                        columns_to_encode=constants.CATEGORICAL_COLUMNS
                        + constants.ONE_HOT_ENCODED
                    ),
                ),
            ]
        )

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        self.categorical_encoder.fit(X)
        return self

    def transform(self, X: DataFrame, y: DataFrame = None, **kwargs):
        X = self.categorical_encoder.transform(X)
        if y is not None:
            return X, y
        return X
