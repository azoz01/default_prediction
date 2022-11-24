from __future__ import annotations
from typing import Union, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (
    PowerTransformer,
    MinMaxScaler,
    StandardScaler,
)
from preprocessing.lib.numerical import OutliersCutter
from utils import constants

logger = logging.getLogger(__name__)


class NumericalColumnsPreprocessor(TransformerMixin):
    """
    1. Encodes cyclical columns to sin/cos columns
    2. Cuts outliers
    3. Reduces skewness using yeo-johnson transformation
    4. Scales selected columns to [0-1] range
    5. Standardize selected columns using z-score scaling
    """

    def __init__(
        self, remove_skewness: bool = True, scale: bool = True
    ) -> None:
        self.columns_to_sine_encode = constants.COLUMNS_TO_CYCLICAL_ENCODING
        self.columns_to_cut_outliers = constants.COLUMNS_TO_DROP_OUTLIERS
        self.columns_to_gaussian_transform = constants.SKEWED_COLUMNS
        self.columns_to_min_max_scale = constants.COLUMNS_TO_SCALE
        self.columns_to_standardize = constants.COLUMNS_TO_STANDARDIZE

        self.standard_transformers = []
        self.standard_transformers.append(
            (
                self.columns_to_cut_outliers,
                OutliersCutter(columns=self.columns_to_cut_outliers),
            )
        )
        if remove_skewness:
            self.standard_transformers.append(
                (
                    self.columns_to_gaussian_transform,
                    PowerTransformer(method="yeo-johnson"),
                )
            )
        if scale:
            if len(self.columns_to_min_max_scale) != 0:
                self.standard_transformers.append(
                    (self.columns_to_min_max_scale, MinMaxScaler())
                )
            if len(self.columns_to_standardize) != 0:
                self.standard_transformers.append(
                    (self.columns_to_standardize, StandardScaler())
                )

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> NumericalColumnsPreprocessor:
        logger.info(
            f"Encoding cyclical columns : {self.columns_to_sine_encode}"
        )
        X = self._encode_cyclical_columns(X)
        logger.info(
            "Fitting standard transformations: "
            "outliers_cutter, power_transform, min_max_scaler, standard_scaler"
        )
        for columns, transformer in self.standard_transformers:
            X[columns] = transformer.fit_transform(X[columns])
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        logger.info(
            f"Encoding cyclical columns : {self.columns_to_sine_encode}"
        )
        X = self._encode_cyclical_columns(X)
        logger.info(
            "Transforming using standard transformations: "
            "outliers_cutter, power_transform, min_max_scaler, standard_scaler"
        )
        for columns, transformer in self.standard_transformers:
            X[columns] = transformer.transform(X[columns])

        if y is not None:
            return X, y
        return X

    def _encode_cyclical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns_to_sine_encode:
            scaled_column = (X[column] - 1) / 12 * 2 * np.pi
            sine_scaled_column = np.sin(scaled_column)
            cosine_scaled_column = np.cos(scaled_column)
            data = np.stack([sine_scaled_column, cosine_scaled_column]).T
            encoded_column = pd.DataFrame(
                data=data, columns=[f"{column}_sine", f"{column}_cosine"]
            )
            X = pd.concat([X, encoded_column], axis=1).drop(columns=[column])
        return X
