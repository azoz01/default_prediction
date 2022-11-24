from typing import Union, Tuple
import pandas as pd
from imblearn.over_sampling import SMOTENC


class SmotencWrapper:
    """
    Performs data augmentation using SMOTENC method.
    """

    def fit_resample(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        cat_cols = X.select_dtypes("category").columns.values
        mask = list(map(lambda col: col in cat_cols, X.columns))
        oversample = oversample = SMOTENC(
            categorical_features=mask, n_jobs=-1, random_state=42
        )
        X, y = oversample.fit_resample(X, y)
        return X, y
