from pandas import DataFrame

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(model, X: DataFrame, y_true: DataFrame):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_pred),
    }
