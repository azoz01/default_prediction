import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import TransformerMixin


class AgeDataPreprocessor(TransformerMixin):
    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        return self

    def transform(self, X: DataFrame, y: DataFrame = None, **kwargs):
        X = self._process_age_columns(X)
        if y is not None:
            return X, y
        return X

    def _process_age_columns(self, X: DataFrame) -> DataFrame:
        X = self._process_days_birth(X)
        X = self._process_days_employed(X)
        X = self._process_days_registration(X)
        return X

    def _process_days_birth(self, X: DataFrame) -> DataFrame:
        age_col: Series = -X["DAYS_BIRTH"] // 365
        age_col_segmented: Series = pd.cut(
            age_col,
            bins=[float("-inf"), 0, 24, 34, 44, 54, 64, float("inf")],
            labels=[
                "Non-positive",
                "<=25",
                "25< & <=35",
                "35< & <=45",
                "45< & <=55",
                "55< & <=65",
                "<65",
            ],
        )
        X["DAYS_BIRTH"] = age_col_segmented
        X = X.rename(columns={"DAYS_BIRTH": "AGE"})
        # X = X.drop(columns=["DAYS_BIRTH"])
        # X = X.assign(AGE=age_col_segmented)
        return X

    def _process_days_employed(self, X: DataFrame) -> DataFrame:
        months_employed_col: Series = -X["DAYS_EMPLOYED"] // 30
        months_employed_col_segmented: Series = pd.cut(
            months_employed_col,
            bins=[float("-inf"), 0, 6, 24, 60, float("inf")],
            labels=["Non-positive", "<=6", "<6 & <=24", "<24 & <=60", "<60"],
        )
        X["DAYS_EMPLOYED"] = months_employed_col_segmented
        X = X.rename(columns={"DAYS_EMPLOYED": "MONTHS_EMPLOYED"})
        # X = X.drop(columns=["DAYS_EMPLOYED"])
        # X = X.assign(MONTHS_EMPLOYED=months_employed_col_segmented)
        return X

    def _process_days_registration(self, X: DataFrame) -> DataFrame:
        years_registration_col: Series = -X["DAYS_REGISTRATION"] // 365
        years_registration_col_segmented: Series = pd.cut(
            years_registration_col,
            bins=[float("-inf"), -1, 5, 10, 20, float("inf")],
            labels=["Negative", "<=5", "<5 & <=10", "<10 & <=20", "<20"],
        )
        X["DAYS_REGISTRATION"] = years_registration_col_segmented
        X = X.rename(columns={"DAYS_REGISTRATION": "YEARS_REGISTRATION"})
        # X = X.drop(columns=["DAYS_REGISTRATION"])
        # X = X.assign(YEARS_REGISTRATION=years_registration_col_segmented)
        return X
