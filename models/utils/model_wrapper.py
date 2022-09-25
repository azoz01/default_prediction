import pandas as pd


class ModelWrapper:
    def __init__(self, prerdict_proba_function, threshold=0.5):
        self._predict_proba_function = prerdict_proba_function
        self._threshold = threshold

    def predict(self, X: pd.DataFrame):
        preds = self._predict_proba_function(X)
        return preds[:, 1] > self._threshold
