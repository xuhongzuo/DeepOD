from sklearn import metrics
import numpy as np
from deepod.metrics.affiliation.generics import convert_vector_to_events
from deepod.metrics.vus.metrics import get_range_vus_roc
from deepod.metrics.affiliation.metrics import pr_from_events


def auc_roc(y_true, y_score):
    """
    Calculates the area under the Receiver Operating Characteristic (ROC) curve.

    Args:
    
        y_true (np.array, required): 
            True binary labels. 0 indicates a normal timestamp, and 1 indicates an anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores. A higher score indicates a higher likelihood of being an anomaly.

    Returns:
    
        float: 
            The score of the area under the ROC curve.
    """
    
    return metrics.roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):
    """
    Calculates the area under the Precision-Recall (PR) curve.

    Args:
    
        y_true (np.array, required): 
            True binary labels. 0 indicates a normal timestamp, and 1 indicates an anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores. A higher score indicates a higher likelihood of being an anomaly.

    Returns:
    
        float: 
            The score of the area under the PR curve.
    """
    
    return metrics.average_precision_score(y_true, y_score)


def tabular_metrics(y_true, y_score):
    """
    Calculates evaluation metrics for tabular anomaly detection.

    Args:
    
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
        tuple: A tuple containing:
        
        - auc_roc (float):
            The score of area under the ROC curve.
            
        - auc_pr (float):
            The score of area under the precision-recall curve.
            
        - f1 (float): 
            The score of F1-score.
    """

    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), f1


def ts_metrics(y_true, y_score):
    """
    Calculates evaluation metrics for time series anomaly detection.

    Args:
    
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
        tuple: A tuple containing:
        
        - roc_auc_score (float):
            The score of area under the ROC curve.
            
        - average_precision_score (float):
            The score of area under the precision-recall curve.
            
        - best_f1 (float): 
            The best score of F1-score.
            
        - best_p (float): 
            The best score of precision.
            
        - best_r (float): 
            The best score of recall.
    """
    
    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), best_f1, best_p, best_r


def get_best_f1(label, score):
    """
    Return the best F1-score, precision and recall

    Args:
        label (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
        tuple: A tuple containing:
        
        - best_f1 (float):
            The best score of F1-score.
        - best_p (float):
            The best score of precision.
        - best_r (float):
            The best score of recall.
    """
    
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def ts_metrics_enhanced(y_true, y_score, y_test):
    """
    This function calculates additional evaluation metrics for time series anomaly detection. It returns a variety of metrics, including those sourced from the code in [A Huet et al. KDD22] and [J Paparrizos et al. VLDB22]. The function requires three inputs: y_true (data label), y_score (predicted anomaly scores), and y_test (predictions of events). 
    
    Args:
        y_true (np.array): 
            Data label, where 0 indicates a normal timestamp and 1 indicates an anomaly.
            
        y_score (np.array): 
            Predicted anomaly scores, where a higher score indicates a higher likelihood of being an anomaly.
        
        y_test (np.array): 
            Predictions of events, where 0 indicates a normal timestamp and 1 indicates an anomaly.

    Returns:
        tuple: A tuple containing:
        
        - auroc (float):
            The score of the area under the ROC curve.
            
        - aupr (float):
            The score of the area under the precision-recall curve.
            
        - best_f1 (float): 
            The best score of F1-score.
            
        - best_p (float): 
            The best score of precision.
            
        - best_r (float): 
            The best score of recall.
            
        - affiliation_precision (float):
            The score of affiliation precision.
            
        - affiliation_recall (float):
            The score of affiliation recall.
            
        - vus_r_auroc (float):
            The score of range VUS ROC.
            
        - vus_r_aupr (float):
            The score of range VUS PR.
            
        - vus_roc (float):
            The score of VUS ROC.
            
        - vus_pr (float):
            The score of VUS PR.
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


