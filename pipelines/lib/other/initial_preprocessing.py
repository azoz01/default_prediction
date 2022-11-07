from sklearn.base import TransformerMixin
import pipelines.lib.constants as constants


class InitialPreprocessor(TransformerMixin):
    def __init__(self):
        self.columns_to_drop = constants.COLUMNS_TO_DROP
        self.categorical_columns = constants.CATEGORICAL_COLUMNS_RAW
        self.numerical_columns = constants.NUMERICAL_COLUMNS_RAW

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        X = X.replace("Y", 1).replace("N", 0)
        X = X.drop(columns=self.columns_to_drop)
        if y is not None:
            return X, y
        return X
