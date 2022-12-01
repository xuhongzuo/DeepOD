from deepod.models import *
from sklearn.metrics import roc_auc_score
import numpy as np

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    file = '../../data/38_thyroid.npz'
    data = np.load(file, allow_pickle=True)
    x, y = data['X'], data['y']
    y = np.array(y, dtype=int)

    anom_id = np.where(y==1)[0]
    known_anom_id = np.random.choice(anom_id, 30)
    y_semi = np.zeros_like(y)
    y_semi[known_anom_id] = 1

    clf = DevNet(device='cpu')
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_score=scores, y_true=y)
    print(auc)

    clf = DeepSVDD(device='cpu')
    clf.fit(x, y_semi)
    scores = clf.decision_function(x)
    auc = roc_auc_score(y_score=scores, y_true=y)
    print(auc)