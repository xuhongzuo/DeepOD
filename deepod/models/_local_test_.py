from deepod.models import *
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from deepod.utils.utility import cal_metrics


if __name__ == '__main__':
    # # thyroid data
    # file = '../../data/38_thyroid.npz'
    # data = np.load(file, allow_pickle=True)
    # x, y = data['X'], data['y']
    # y = np.array(y, dtype=int)

    # ts data
    train_file = '../../data/omi-1/omi-1_train.csv'
    test_file = '../../data/omi-1/omi-1_test.csv'
    train_df = pd.read_csv(train_file, sep=',', index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)
    y = test_df['label'].values
    train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)
    x = train_df.values
    x_test = test_df.values

    # anom_id = np.where(y==1)[0]
    # known_anom_id = np.random.choice(anom_id, 30)
    # y_semi = np.zeros_like(y)
    # y_semi[known_anom_id] = 1

    # clf = DevNet(device='cpu')
    # clf.fit(x, y_semi)
    #
    # scores = clf.decision_function(x)
    #
    # from sklearn.metrics import roc_auc_score
    # auc = roc_auc_score(y_score=scores, y_true=y)
    # print(auc)

    clf = DeepSVDD(data_type='ts', stride=50, seq_len=100, epochs=20,
                   device='cpu', network='TCN')
    clf.fit(x)
    scores = clf.decision_function(x_test)

    adj_eval_info = cal_metrics(y, scores, pa=True)
    print(adj_eval_info)