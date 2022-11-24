from __future__ import annotations
from typing import Union, Tuple
import logging
import pandas as pd
import miceforest as mf
from sklearn.base import TransformerMixin
from feature_engine.encoding import CountFrequencyEncoder
from utils import constants

logger = logging.getLogger(__name__)


class Imputer(TransformerMixin):
    """
    1. Imputes dates with oldest ones present in columns.
    2. Imputes selected categorical variables using 'missing' category.
    3. Imputes selected columns with 0.
    4. Categorizes chosen columns based on quartiles
        and imputes missing category.
    5. Imputes selected columns with means.
    6. Encodes high cardinality columns using frequency encoding.
    7. Imputes selected columns using MICE imputation method.

    """

    def __init__(self) -> None:
        self.columns_to_impute_missing_category = (
            constants.COLUMNS_TO_IMPUTE_MISSING_CATEGORY
        )
        self.columns_to_impute_0 = constants.COLUMNS_TO_IMPUTE_0
        self.columns_to_frequency_encode = (
            constants.COLUMNS_TO_FREQUENCY_ENCODE
        )
        self.frequency_encoder = CountFrequencyEncoder(
            variables=self.columns_to_frequency_encode
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> Imputer:

        logger.info("Fitting Frequency Encoder")
        self.frequency_encoder.fit(
            X[self.columns_to_frequency_encode].fillna("missing"), y
        )

        logger.info("Preparing data to fit MICE imputer")
        X = self._transform_pre_mice(X)

        logger.info("Converting categorical types")
        object_columns = X.select_dtypes("object").columns
        X[object_columns] = X[object_columns].astype("category")

        logger.info("Converting numerical columns to proper types")
        num_cols = X.select_dtypes(exclude=["category"]).columns
        X[num_cols] = X[num_cols].astype("float")

        logger.info("Saving schema of training data")
        self.cat_cols_dict = {
            col: X[col].dtype for col in X.select_dtypes("category").columns
        }

        logger.info("Fitting MICE imputer")
        self.kernel = mf.ImputationKernel(data=X, datasets=5, random_state=42)
        self.kernel.mice(verbose=True)

        logger.info("Fitting Imputer completed")
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        logger.info("Preparing data to be imputed by MICE imputer")
        X = self._transform_pre_mice(X)

        logger.info("Converting categorical columns to proper types")
        for col, dtype in self.cat_cols_dict.items():
            X[col] = X[col].astype(dtype)

        logger.info("Converting numerical columns to proper types")
        num_cols = X.select_dtypes(exclude=["category"]).columns
        X[num_cols] = X[num_cols].astype("float")

        logger.info("Imputing data using MICE imputer")
        X = self.kernel.transform(X)

        logger.info("Data imputation completed")
        if y is None:
            return X
        return X, y

    def _transform_pre_mice(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        1. Imputes selected categorical variables using 'missing' category
        2. Imputes selected columns with 0
        3. Encodes high cardinality columns using Frequency encoding

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        X = X.copy()

        logger.info(
            f"Imputing {self.columns_to_impute_missing_category} "
            "using 'missing' category"
        )
        X[self.columns_to_impute_missing_category] = X[
            self.columns_to_impute_missing_category
        ]

        logger.info(f"Imputing {self.columns_to_impute_0} with 0")
        X[self.columns_to_impute_0] = X[self.columns_to_impute_0].fillna(0)

        logger.info(
            "Encoding emp_title using Frequency encoding "
            "(if missing then default frequency is imputed)"
        )
        X["emp_title"] = self.frequency_encoder.transform(
            X[["emp_title"]].fillna("missing")
        ).fillna(0.19)

        return X
