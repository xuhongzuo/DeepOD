import numpy as np


def point_adjustment(y_true, y_score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    *This function is copied/modified from the source code in [Zhihan Li et al. KDD21]* 

    Args:
    
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
    
        np.array: 
            Adjusted anomaly scores.

    """
    score = y_score.copy()
    assert len(score) == len(y_true)
    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = y_true[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(y_true)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score
