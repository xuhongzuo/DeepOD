from deepod.metrics.vus.utils.metrics import metricor


def get_range_vus_roc(score, labels, slidingWindow):
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor().RangeAUC(labels=labels, score=score,
                                                       window=slidingWindow, plot_ROC=True)
    _, _, _, _, VUS_ROC, VUS_PR = metricor().RangeAUC_volume(labels_original=labels,
                                                 score=score,
                                                 windowSize=2*slidingWindow)

    metrics = {'R_AUC_ROC': R_AUC_ROC, 'R_AUC_PR': R_AUC_PR, 'VUS_ROC': VUS_ROC, 'VUS_PR': VUS_PR}

    return metrics
