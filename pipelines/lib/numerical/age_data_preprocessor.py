from typing import Tuple
import pandas as pd
from sklearn.base import TransformerMixin


class AgeDataPreprocessor(TransformerMixin):
    """
    Applies proper preprocessing to age columns
    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        """
        Placeholder method for fit

        Args:
            X (pd.DataFrame)
            y (pd.DataFrame, optional)
        """
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes age columns inside DataFrame

        Args:
            X (DataFrame): X to transform
            y (DataFrame, optional): y to transform, if not None, then
                passed through. Defaults to None.

        Returns:
            Union[DataFrame, Tuple[DataFrame, DataFrame]]: DataFrame with 
                processed age columns. If y is not None, then passed through.
        """
        X = self._process_age_columns(X)
        if y is not None:
            return X, y
        return X

    def _process_age_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Method responsible for business logic. Method processes:
            * DAYS_BIRTH
            * DAYS_EMPLOYEED
            * DAYS_REGISTRATION
            * OWN_CAR_AGE

        Args:
            X (DataFrame): X to process. Contains proper columns


        Returns:
            DataFrame: X with processed age columns
        """
        X = X.copy()
        X = self._process_days_birth(X)
        X = self._process_days_employed(X)
        X = self._process_days_registration(X)
        X = self._process_own_car_age(X)
        return X

    def _process_days_birth(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Processes DAYS_BIRTH column. Originally columns contains
        days count from birth to data collection. Value is stored
        in negative value. Pipeline makes value positive, calculates
        years completed and categorizes obtained value

        Args:
            X (DataFrame): X to process with DAYS_BIRTH

        Returns:
            pd.DataFrame: X with processed DAYS_BIRTH
        """
        age_col: pd.Series = -X["DAYS_BIRTH"] // 365
        age_col_segmented: pd.Series = pd.cut(
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
        return X

    def _process_days_employed(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Processes DAYS_EMPLOYED column. Originally columns contains
        days count from hiring to data collection. Value is stored
        in negative value. Pipeline makes value positive, calculates
        months completed and categorizes obtained value

        Args:
            X (DataFrame): X to process with DAYS_EMPLOYED

        Returns:
            pd.DataFrame: X with processed DAYS_EMPLOYED
        """
        months_employed_col: pd.Series = -X["DAYS_EMPLOYED"] // 30
        months_employed_col_segmented: pd.Series = pd.cut(
            months_employed_col,
            bins=[float("-inf"), 0, 6, 24, 60, float("inf")],
            labels=["Non-positive", "<=6", "<6 & <=24", "<24 & <=60", "<60"],
        )
        X["DAYS_EMPLOYED"] = months_employed_col_segmented
        X = X.rename(columns={"DAYS_EMPLOYED": "MONTHS_EMPLOYED"})
        return X

    def _process_days_registration(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Processes DAYS_REGISTRATION column. Originally columns contains
        days count from registration of residence to data collection. 
        Value is stored in negative value. Pipeline makes value positive, 
        calculates years completed and categorizes obtained value

        Args:
            X (DataFrame): X to process with DAYS_REGISTRATION

        Returns:
            pd.DataFrame: X with processed DAYS_REGISTRATION
        """
        years_registration_col: pd.Series = -X["DAYS_REGISTRATION"] // 365
        years_registration_col_segmented: pd.Series = pd.cut(
            years_registration_col,
            bins=[float("-inf"), -1, 5, 10, 20, float("inf")],
            labels=["Negative", "<=5", "<5 & <=10", "<10 & <=20", "<20"],
        )
        X["DAYS_REGISTRATION"] = years_registration_col_segmented
        X = X.rename(columns={"DAYS_REGISTRATION": "YEARS_REGISTRATION"})
        return X

    def _process_own_car_age(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Processes OWN_CAR_AGE column. Originally columns contains
        years count of current car ownership relative to data collection. 
        Value is stored in negative value. Pipeline makes value positive 
        and categorizes obtained value

        Args:
            X (DataFrame): X to process with OWN_CAR_AGE

        Returns:
            pd.DataFrame: X with processed OWN_CAR_AGE
        """
        own_car_age_col: pd.Series = X["OWN_CAR_AGE"]
        own_car_age_col_segmented: pd.Series = pd.cut(
            own_car_age_col,
            bins=[float("-inf"), 0, 3, 6, 9, float("inf")],
            labels=["Undefined", "3<=", "3<&<=6", "6<&<=9", "9<"],
        )
        X["OWN_CAR_AGE"] = own_car_age_col_segmented
        return X
