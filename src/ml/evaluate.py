from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def calculate_metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    y_pred = [int(p >= 0.50) for p in y_prob]
    report = classification_report(y_true, y_pred)
    
    return {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "classification_report": report
    }
