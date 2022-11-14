from imblearn.over_sampling import SMOTENC
from utils import constants


class SmotencWrapper:
    def fit_resample(self, X, y):
        cat_cols = set(X.columns) - set(constants.NUMERICAL_COLUMNS)
        mask = list(map(lambda col: col in cat_cols, X.columns))
        oversample = oversample = SMOTENC(
            categorical_features=mask, n_jobs=-1, random_state=42
        )
        X, y = oversample.fit_resample(X, y)
        return X, y
