from typing import Tuple, Union
from typing_extensions import Self
from sklearn.base import TransformerMixin
from pandas import DataFrame


class IrrelevantColumnsDropper(TransformerMixin):
    COLUMNS_TO_DROP: list[str] = [
        "CODE_GENDER",
        "NAME_TYPE_SUITE",
        "DAYS_LAST_PHONE_CHANGE",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "SK_ID_CURR",
        "DAYS_ID_PUBLISH",
        "WEEKDAY_APPR_PROCESS_START",
        "HOUR_APPR_PROCESS_START",
    ]

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        return self

    def transform(
        self, X: DataFrame, y: DataFrame = None, **kwargs
    ) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        X = X.drop(columns=self.COLUMNS_TO_DROP)
        if y:
            return X, y
        else:
            return X

