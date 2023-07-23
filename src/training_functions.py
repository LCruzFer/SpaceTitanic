from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def evaluate(true, preds) -> dict:
    """
    Calculate various evaluation metrics.

    *true: array-like
        Array containing true values.
    *preds: array-like
        Array containing predicted values.
    """
    auc = roc_auc_score(true, preds)
    precision = precision_score(true, preds)
    recall = recall_score(true, preds)
    f1 = f1_score(true, preds)

    score_dict = {"auc": auc, "precision": precision, "recall": recall, "f1": f1}

    return score_dict
