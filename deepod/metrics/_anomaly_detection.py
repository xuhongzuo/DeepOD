from sklearn import metrics
import numpy as np
from deepod.metrics.affiliation.generics import convert_vector_to_events
from deepod.metrics.vus.metrics import get_range_vus_roc
from deepod.metrics.affiliation.metrics import pr_from_events


def auc_roc(y_true, y_score):
    return metrics.roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):
    return metrics.average_precision_score(y_true, y_score)


def tabular_metrics(y_true, y_score):
    """calculate evaluation metrics"""

    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), f1


def ts_metrics(y_true, y_score):
    """calculate evaluation metrics for time series anomaly detection"""
    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), best_f1, best_p, best_r


def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def ts_metrics_enhanced(y_true, y_score, y_test):
    """
    Compared with ts_metrics, this function can return more metrics
    with one more input y_test (predictions of events)
    revised by @Yiyuan Yang 2023/11/08

    Args:
        y_true:
        y_score:
        y_test

    Returns:
        auroc:
        aupr:
        best_f1:
        best_p:
        best_r:
        affiliation_precision:
        affiliation_recall:
        vus_r_auroc:
        vus_r_aupr:
        vus_roc:
        vus_pr:

    Example:
        from deepod.models.time_series import DCdetector
        clf = DCdetector()
        clf.fit(X_train)
        pred, scores = clf.decision_function(X_test)

        from deepod.metrics import point_adjustment
        from deepod.metrics import ts_metrics_enhanced
        adj_eval_metrics = ts_metrics_enhanced(labels, point_adjustment(labels, scores), pred)
        print('adj_eval_metrics',adj_eval_metrics)
    """

    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    
    events_pred = convert_vector_to_events(y_test) 
    events_gt = convert_vector_to_events(y_true)
    Trange = (0, len(y_test))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    vus_results = get_range_vus_roc(y_score, y_true, 100) # default slidingWindow = 100

    auroc = auc_roc(y_true, y_score)
    aupr = auc_pr(y_true, y_score)

    affiliation_precision = affiliation['Affiliation_Precision']
    affiliation_recall = affiliation['Affiliation_Recall']
    vus_r_auroc = vus_results["R_AUC_ROC"]
    vus_r_aupr = vus_results["R_AUC_PR"]
    vus_roc = vus_results["VUS_ROC"]
    vus_pr = vus_results["VUS_PR"]

    return auroc, aupr, best_f1, best_p, best_r, \
           affiliation_precision, affiliation_recall, \
           vus_r_auroc, vus_r_aupr, \
           vus_roc, vus_pr


