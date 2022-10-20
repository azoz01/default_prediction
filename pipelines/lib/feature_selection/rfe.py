import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE


class RfeAdapter(TransformerMixin):
    def __init__(self, output_n_cols: int):
        self.rfe: RFE = RFE(
            estimator=DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=28,
                min_samples_leaf=10,
                class_weight="balanced",
                max_features="sqrt",
                random_state=42,
            ),
            n_features_to_select=output_n_cols,
            step=10,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        self.rfe.fit(X, y, **kwargs)
        self.out_columns: np.ndarray[str] = self.rfe.feature_names_in_[
            self.rfe.ranking_ == 1
        ]
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        transformed_data = self.rfe.transform(X, **kwargs)
        X = pd.DataFrame(data=transformed_data, columns=self.out_columns)
        if y is not None:
            return X, y
        return X
