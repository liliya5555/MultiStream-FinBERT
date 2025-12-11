# multistream_finbert_evaluation_metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_predictions(y_true, y_pred, y_prob):
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob)
    }
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results
