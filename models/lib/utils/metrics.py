from typing import Dict, Callable
from pandas import DataFrame
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(
    model, X: DataFrame, y_true: DataFrame, predict_fun: Callable = None
) -> Dict[str, float]:
    if not predict_fun:
        predict_fun = model.predict
    y_pred = predict_fun(X)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_pred),
    }
