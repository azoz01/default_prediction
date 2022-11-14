from __future__ import annotations
from typing import Tuple, Union
import logging
import pandas as pd
from sklearn.base import TransformerMixin
from utils import constants

logger = logging.getLogger(__name__)


class InitialPreprocessor(TransformerMixin):
    """
    Drops unwanted columns from dataset and selects
    only relevant target values.
    """

    def __init__(self) -> None:
        self.columns_to_drop = constants.DATA_TO_DROP
        self.target_column = constants.TARGET_COLUMN
        self.allowed_targets = [
            constants.FULLY_PAID_STATUS,
            constants.CHARGED_OFF_STATUS,
        ]
        self.date_columns_to_split = constants.DATE_COLUMNS_TO_SPLIT
        self.months_to_nums_mapping = constants.MONTHS_TO_NUMS_MAPPING
        self.grade_to_num_mapping = {
            "G": 7,
            "F": 6,
            "E": 5,
            "D": 4,
            "C": 3,
            "B": 2,
            "A": 1,
        }

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> InitialPreprocessor:
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        logger.info("Dropping unwanted columns")
        X = X.drop(columns=self.columns_to_drop)

        logger.info("Selecting relevant feature values")
        allowed_target_mask = X[self.target_column].apply(
            lambda target: target in self.allowed_targets
        )
        X = X.loc[allowed_target_mask]

        logger.info("Converting features to numerical values")
        X[constants.TARGET_COLUMN] = X[constants.TARGET_COLUMN].map(
            {constants.FULLY_PAID_STATUS: 0, constants.CHARGED_OFF_STATUS: 1}
        )

        logger.info("Splitting date columns")
        for column in self.date_columns_to_split:
            column_splitted = (
                X[column]
                .str.split("-", 1, expand=True)
                .rename(columns={0: f"{column}_month", 1: f"{column}_year"})
            )
            column_splitted[f"{column}_year"] = column_splitted[
                f"{column}_year"
            ].astype("float")
            column_splitted[f"{column}_month"] = (
                column_splitted[f"{column}_month"]
                .map(self.months_to_nums_mapping)
                .astype("float")
            )
            X = X.drop(columns=[column])
            X = pd.concat([X, column_splitted], axis=1)

        logger.info("Trimming emp_title column")
        X["emp_title"] = X["emp_title"].str.strip()

        logger.info("Ecoding grade to numerical values using ordinal encoding")
        X["grade"] = X["grade"].map(self.grade_to_num_mapping)

        if y is not None:
            return X, y
        return X
