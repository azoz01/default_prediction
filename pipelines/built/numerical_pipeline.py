from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from pipelines.null_removal import *
from pipelines.numerical import *
import pipelines.constants as constants


class NumericalPipeline(TransformerMixin):
    def __init__(self):
        self.numerical_pipeline = Pipeline(
            [
                ("age_data_preprocessor", AgeDataPreprocessor()),
                (
                    "to_gaussian_transformer",
                    ToGaussianTransformer(constants.NUMERICAL_COLUMNS),
                ),
            ]
        )

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        self.numerical_pipeline.fit(X)
        return self

    def transform(self, X: DataFrame, y: DataFrame = None, **kwargs):
        X = self.numerical_pipeline.transform(X)
        if y is not None:
            return X, y
        return X

