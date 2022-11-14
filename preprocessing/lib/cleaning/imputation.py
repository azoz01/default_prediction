from __future__ import annotations
from typing import Union, Tuple, Dict
import logging
import pandas as pd
import miceforest as mf
from sklearn.base import TransformerMixin
from utils import constants

logger = logging.getLogger(__name__)


class Imputer(TransformerMixin):
    def __init__(self) -> None:

        self.columns_to_categorize = constants.COLUMNS_TO_CATEGORIZE
        self.missing_data_indicator_columns_names = {
            "last_pymnt_d_year": "first_inst",
            "last_credit_pull_d_year": "no_credit_pull",
        }
        self.missing_categories_names = {
            "mths_since_last_delinq": "no_delinq",
            "mths_since_last_record": "no_record",
            "mths_since_last_major_derog": "no_major_derog",
        }
        self.columns_to_impute_missing_category = (
            constants.COLUMNS_TO_IMPUTE_MISSING_CATEGORY
        )
        self.columns_to_impute_zero = constants.COLUMNS_TO_IMPUTE_0
        self.columns_to_impute_mean = constants.COLUMNS_TO_IMPUTE_MEAN

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> Imputer:
        logger.info(
            "Finding oldest years for last_pymnt_d and last_credit_pull_d"
        )
        self.min_years = self._get_min_years(X)
        logger.info(
            "Finding quantiles for categorization "
            f"for columns: {self.columns_to_categorize}"
        )
        self.quantiles_dict = self._get_quantiles_for_columns(X)

        logger.info("Computing means to impute")
        self.means_for_impute = self._get_means_for_impute(X)
        (
            self.default_frequency,
            self.frequencies_to_impute,
        ) = self._get_target_frequencies_for_emp_title(X, y)

        logger.info("Preparing data to fit MICE imputer")
        X = self._transform_pre_mice(X)

        logger.info("Converting categorical types")
        object_columns = X.select_dtypes("object").columns
        X[object_columns] = X[object_columns].astype("category")

        logger.info("Saving schema of training data")
        self.cat_cols_dict = {
            col: X[col].dtype for col in X.select_dtypes("category").columns
        }

        logger.info("Fitting MICE imputer")
        self.kernel = mf.ImputationKernel(data=X, datasets=2, random_state=42)
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

        logger.info("Imputing data using MICE imputer")
        X = self.kernel.transform(X)

        logger.info("Data imputation completed")
        if y is None:
            return X
        return X, y

    def _transform_pre_mice(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        logger.info(
            "Imputing first_inst, no_credit_pull "
            "month and year using oldest year and January"
        )
        for column in ["last_pymnt_d", "last_credit_pull_d"]:
            X[f"{column}_month"] = X[f"{column}_month"].fillna(1)
            X[f"{column}_year"] = X[f"{column}_year"].fillna(
                self.min_years[f"{column}_year"]
            )

        logger.info(
            f"Imputing {self.columns_to_impute_missing_category} "
            "using 'missing' category"
        )
        X[self.columns_to_impute_missing_category] = X[
            self.columns_to_impute_missing_category
        ].fillna("missing")

        logger.info(f"Imputing {self.columns_to_impute_zero} with 0")
        X[self.columns_to_impute_zero] = X[self.columns_to_impute_zero].fillna(
            0
        )

        logger.info(
            "Categorizing and adding special category "
            f"for {self.columns_to_categorize}"
        )
        for column in self.columns_to_categorize:
            X[column] = pd.cut(
                X[column],
                self.quantiles_dict[column],
                labels=["1_quant", "2_quant", "3_quant", "4_quant"],
            )
            X[column] = X[column].cat.add_categories(
                self.missing_categories_names[column]
            )
            X[column] = X[column].fillna(self.missing_categories_names[column])

        logger.info(f"Imputing with means: {self.columns_to_impute_mean}")
        for column, mean in self.means_for_impute.items():
            X[column] = X[column].fillna(mean)

        logger.info(
            "Encoding emp_title using frequency encoding "
            "(if missing then mean of target is imputed)"
        )
        X["emp_title"] = (
            X["emp_title"]
            .map(self.frequencies_to_impute)
            .fillna(self.default_frequency)
        )
        return X

    def _get_min_years(self, X) -> Dict[str, float]:
        min_years = {}
        for column in ["last_pymnt_d", "last_credit_pull_d"]:
            year_col = X[f"{column}_year"].astype("float")
            min_years[f"{column}_year"] = year_col.min()
        return min_years

    def _get_quantiles_for_columns(
        self, X: pd.DataFrame
    ) -> Dict[str, list[float]]:
        quantiles_dict = {}
        for column in self.columns_to_categorize:
            quantiles_dict[column] = self._get_quantiles(X[column])
        return quantiles_dict

    def _get_quantiles(self, column: pd.Series) -> list[float]:
        quantiles = [column.quantile(q) for q in [0, 0.25, 0.5, 0.75, 1]]
        return quantiles

    def _get_oldest_date(self, date_col: pd.Series) -> str:
        splitted_col = date_col.loc[~date_col.isna().values].str.split(
            "-", expand=True
        )
        min_yr = splitted_col[1].min()
        min_mon = "Jan"
        min_date = f"{min_mon}-{min_yr}"
        return min_date

    def _get_means_for_impute(self, X: pd.DataFrame) -> Dict[str, float]:
        return {
            column: X[column].mean() for column in self.columns_to_impute_mean
        }

    def _get_target_frequencies_for_emp_title(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[float, Dict[str, float]]:
        default_frequency = y[constants.TARGET_COLUMN].mean()
        X = pd.concat([X, y], axis=1)
        means = X.groupby("emp_title").agg({"loan_status": "mean"})
        frequencies_to_impute = {
            title: mean.values[0] for title, mean in means.iterrows()
        }
        return default_frequency, frequencies_to_impute
