from deepod.models import *
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from deepod.utils.utility import cal_metrics


if __name__ == '__main__':
    # # # thyroid data
    # file = '../../data/38_thyroid.npz'
    # data = np.load(file, allow_pickle=True)
    # x, y = data['X'], data['y']
    # y = np.array(y, dtype=int)
    #
    # anom_id = np.where(y == 1)[0]
    # semi_y = np.zeros_like(y)
    # semi_y[np.random.choice(anom_id, 30, replace=False)] = 1

    # # # ts data
    train_file = '../../data/omi-1/omi-1_train.csv'
    test_file = '../../data/omi-1/omi-1_test.csv'
    train_df = pd.read_csv(train_file, sep=',', index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)
    y_test = test_df['label'].values
    y_train = train_df['label'].values
    train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)
    x_train = train_df.values
    x_test = test_df.values

    n = len(x_test)
    x_dsad_train = np.vstack([x_train, x_test[:int(n*0.5)]])
    y_dsad_train = np.hstack([y_train, y_test[:int(n*0.5)]])

    x_val = x_test[int(n*0.5):]
    y_val = y_test[int(n*0.5):]

    print(x_dsad_train.shape, x_val.shape)
    print(np.sum(y_dsad_train), np.sum(y_val))

    # @TODO:
    """
    semi setting: use all training data and partial 
        testing set (before the end point of the first anomaly segment)
        only the first anomaly segment is known anomalies, and the rest of data is unlabeled.
    """

    # ---------------------------------------- #
    # clf = DevNet(device='cpu')
    # clf.fit(x, semi_y)
    # scores = clf.decision_function(x)
    # auc = roc_auc_score(y_score=scores, y_true=y)
    # print(auc)

    # clf = DeepSAD(data_type='tabular', epochs=20,
    #               device='cpu', network='MLP')
    # clf.fit(x, semi_y)
    # scores = clf.decision_function(x)
    # auc = roc_auc_score(y, scores)
    # print(auc)

    # -------------------- ts data --------------------- #

    # clf = DeepSVDD(data_type='ts', stride=50, seq_len=100, epochs=20,
    #                device='cpu', network='TCN')
    # clf.fit(x_dsad_train)
    # scores = clf.decision_function(x_val)
    #
    # adj_eval_info = cal_metrics(y_val, scores, pa=True)
    # print(adj_eval_info)

    clf = DeepSAD(data_type='ts', stride=50, seq_len=100, epochs=20,
                  device='cpu', network='TCN')
    clf.fit(x_dsad_train, y_dsad_train)
    scores = clf.decision_function(x_val)
    adj_eval_info = cal_metrics(y_val, scores, pa=True)
    print(adj_eval_info)

    pred, conf = clf.predict(x_val, return_confidence=True)
    print(conf.shape)
