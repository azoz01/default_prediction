from __future__ import annotations
from typing import Tuple, Union
import logging
import pandas as pd
from sklearn.base import TransformerMixin
from utils import constants

logger = logging.getLogger(__name__)


class InitialPreprocessor(TransformerMixin):
    """
    1. Drops unwanted columns
    2. Selects only relevant target values
    3. Converts target to numerical variable
    4. Splits date columns to month and year
    5. Trims 'emp_title' column
    6. Encodes 'grade' using ordinal encoding


    Drops unwanted columns from dataset and selects
    only relevant target values.
    """

    def __init__(self) -> None:
        self.columns_to_drop = constants.COLUMNS_TO_DROP
        self.target_column = constants.TARGET_COLUMN
        self.allowed_targets = [
            constants.FULLY_PAID_STATUS,
            constants.CHARGED_OFF_STATUS,
        ]
        self.date_columns_to_split = constants.DATE_COLUMNS_TO_SPLIT
        self.months_to_nums_mapping = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }
        self.grade_to_num_mapping = {
            "G": 7,
            "F": 6,
            "E": 5,
            "D": 4,
            "C": 3,
            "B": 2,
            "A": 1,
        }
        self.term_to_num_mapping = {" 36 months": 0, " 60 months": 1}
        self.emp_len_to_num_mapping = {
            "10+ years": 11,
            "4 years": 10,
            "< 1 year": 9,
            "6 years": 8,
            "9 years": 7,
            "2 years": 6,
            "3 years": 5,
            "8 years": 4,
            "7 years": 3,
            "5 years": 2,
            "1 year": 1,
        }
        self.home_ownership_mapping = {
            "RENT": "RENT",
            "MORTGAGE": "MORTGAGE",
            "OWN": "OTHER",
            "OTHER": "OTHER",
            "NONE": "OTHER",
            "ANY": "OTHER",
        }
        self.purpose_mapping = {
            "vacation": "vacation",
            "debt_consolidation": "debt_consolidation",
            "credit_card": "credit_card",
            "home_improvement": "home_improvement",
            "small_business": "small_business",
            "major_purchase": "major_purchase",
            "other": "other",
            "medical": "other",
            "wedding": "other",
            "car": "other",
            "moving": "other",
            "house": "other",
            "educational": "other",
            "renewable_energy": "other",
        }
        self.columns_to_binarize = constants.COLUMNS_TO_BINARIZE

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

        logger.info("Converting target to numerical values")
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

        logger.info(
            "Ecoding specified categorical "
            "to numerical values using ordinal encoding"
        )
        X["term"] = X["term"].map(self.term_to_num_mapping)
        X["grade"] = X["grade"].map(self.grade_to_num_mapping)
        X["emp_length"] = X["emp_length"].map(self.emp_len_to_num_mapping)

        logger.info("Merging categories in home_ownerhip and purpose column")
        X["home_ownership"] = X["home_ownership"].map(
            self.home_ownership_mapping
        )
        X["purpose"] = X["purpose"].map(self.purpose_mapping)

        logger.info(f"Binarizing columns: {self.columns_to_binarize}")
        for column in self.columns_to_binarize:
            X[column] = (X[column] > 0).astype(int)

        if y is not None:
            return X, y
        return X
