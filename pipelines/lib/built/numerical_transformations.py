from typing import Tuple, Union
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from ..null_removal import *
from ..numerical import *
from .. import constants


class NumericalTransformations(TransformerMixin):
    """
    Pipeline applies numerical transformations to 
    proper columns of DataFrame
    """

    def __init__(self):
        self.numerical_pipeline: Pipeline = Pipeline(
            [
                ("age_data_preprocessor", AgeDataPreprocessor()),
                (
                    "to_gaussian_transformer",
                    ToGaussianTransformer(constants.NUMERICAL_COLUMNS),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, **kwargs):
        """
        Fits pipeline

        Args:
            X (pd.DataFrame): X to fit
        """
        self.numerical_pipeline.fit(X)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Transforms data according to pipeline's logic

        Args:
            X (pd.DataFrame): X to transform
            y (pd.DataFrame, optional): If not none, then passed through. 
                Defaults to None.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]: 
                Transformed X and y if passed
        """
        X = self.numerical_pipeline.transform(X)
        if y is not None:
            return X, y
        return X

