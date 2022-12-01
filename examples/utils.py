import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.io import arff
from glob import glob
import datetime


def read_data(file, split='50%-normal', normalization='z-score', seed=42):
    """
    read data from files, normalization, and perform train-test splitting

    Parameters
    ----------
    file: str
        file path of dataset

    split: str (default='50%-normal', choice=['50%-normal', '60%', 'none'])
        training-testing set splitting methods:
            - if '50%-normal': use half of the normal data as training set,
                and the other half attached with anomalies as testing set,
                this splitting method is used in self-supervised studies GOAD [ICLR'20], NeuTraL [ICML'21]
            - if 'none': use the whole set as both the training and testing set
                This is commonly used in traditional methods.
            - if '60%': use 60% data during training and the rest 40% data in testing,
                while keeping the original anomaly ratio.

    normalization: str (default='z-score', choice=['z-score', 'min-max'])

    seed: int (default=42)
        random seed
    """

    if file.endswith('.npz'):
        data = np.load(file, allow_pickle=True)
        x, y = data['X'], data['y']
        y = np.array(y, dtype=int)
    else:
        if file.endswith('pkl'):
            func = pd.read_pickle
        elif file.endswith('csv'):
            func = pd.read_csv
        elif file.endswith('arff'):
            def func(f):
                df_ = pd.DataFrame(arff.loadarff(f)[0])
                df_ = df_.replace({b'no': 0, b'yes': 1})
                df_ = df_.drop('id', axis=1)
                return df_
        else:
            raise NotImplementedError('')

        df = func(file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        x = df.values[:, :-1]
        y = np.array(df.values[:, -1], dtype=int)

    # train-test splitting
    if split == '50%-normal':
        rng = np.random.RandomState(seed)
        idx = rng.permutation(np.arange(len(x)))
        x, y = x[idx], y[idx]

        norm_idx = np.where(y==0)[0]
        anom_idx = np.where(y==1)[0]
        split = int(0.5 * len(norm_idx))
        train_norm_idx, test_norm_idx = norm_idx[:split], norm_idx[split:]

        x_train = x[train_norm_idx]
        y_train = y[train_norm_idx]

        x_test = x[np.hstack([test_norm_idx, anom_idx])]
        y_test = y[np.hstack([test_norm_idx, anom_idx])]

        print(f'Original size: [{x.shape}], Normal/Anomaly: [{len(norm_idx)}/{len(anom_idx)}] \n'
              f'After splitting: training/testing [{len(x_train)}/{len(x_test)}]')

    elif split == '60%':
        x_train, y_train, x_test, y_test = train_test_split(x, y, shuffle=True, random_state=seed,
                                                            test_size=0.4, stratify=y)

    else:
        x_train, x_test = x.copy(), x.copy()
        y_train, y_test = y.copy(), y.copy()

    # normalization
    if normalization == 'min-max':
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(x_train)
        x_train = minmax_scaler.transform(x_train)
        x_test = minmax_scaler.transform(x_test)

    elif normalization == 'z-score':
        mus = np.mean(x_train, axis=0)
        sds = np.std(x_train, axis=0)
        sds[sds == 0] = 1
        x_train = np.array([(xx - mus) / sds for xx in x_train])
        x_test = np.array([(xx - mus) / sds for xx in x_test])

    elif normalization == 'scale':
        x_train = x_train / 255
        x_test = x_test / 255

    return x_train, y_train, x_test, y_test


def min_max_normalize(x):
    filter_lst = []
    for k in range(x.shape[1]):
        s = np.unique(x[:, k])
        if len(s) <= 1:
            filter_lst.append(k)
    if len(filter_lst) > 0:
        print('remove features', filter_lst)
        x = np.delete(x, filter_lst, 1)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    return x


def evaluate(y_true, scores):
    """calculate evaluation metrics"""
    roc_auc = metrics.roc_auc_score(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)

    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true==0)[0]) / len(y_true)
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = y_true.astype(int)
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f_score


def get_data_lst(dataset_dir, dataset):
    if dataset == 'FULL':
        print(os.path.join(dataset_dir, '*.*'))
        data_lst = glob(os.path.join(dataset_dir, '*.*'))
    else:
        name_lst = dataset.split(',')
        data_lst = []
        for d in name_lst:
            data_lst.extend(glob(os.path.join(dataset_dir, d + '.*')))
    data_lst = sorted(data_lst)
    return data_lst


def add_irrelevant_features(x, ratio, seed=None):
    n_samples, n_f = x.shape
    size = int(ratio * n_f)

    irr_new = np.zeros([n_samples, size])
    np.random.seed(seed)
    for i in tqdm(range(size)):
        irr_new[:, i] = np.random.rand(n_samples)

    # irr_new = np.zeros([n_samples, size])
    # np.random.seed(seed)
    # for i in range(size):
    #     array = x[:, np.random.choice(x.shape[1], 1)]
    #     new_array = array[np.random.permutation(n_samples)].flatten()
    #     irr_new[:, i] = new_array

    x_new = np.hstack([x, irr_new])

    return x_new


def adjust_contamination(x, y, contamination_r, swap_ratio=0.05, random_state=42):
    """
    add/remove anomalies in training data to replicate anomaly contaminated data sets.
    randomly swap 5% features of two anomalies to avoid duplicate contaminated anomalies.
    """
    rng = np.random.RandomState(random_state)

    anom_idx = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    n_cur_anom = len(anom_idx)
    n_adj_anom = int(len(norm_idx) * contamination_r / (1. - contamination_r))

    # x_train = np.delete(x_train, unknown_anom_idx, axis=0)
    # y_train = np.delete(y_train, unknown_anom_idx, axis=0)
    # noises = inject_noise(true_anoms, n_adj_noise, 42)
    # x_train = np.append(x_train, noises, axis=0)
    # y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    # inject noise
    if n_cur_anom < n_adj_anom:
        n_inj_noise = n_adj_anom - n_cur_anom
        print(f'Control Contamination Rate: injecting [{n_inj_noise}] Noisy samples')

        seed_anomalies = x[anom_idx]

        n_sample, dim = seed_anomalies.shape
        n_swap_feat = int(swap_ratio * dim)
        inj_noise = np.empty((n_inj_noise, dim))
        for i in np.arange(n_inj_noise):
            idx = rng.choice(n_sample, 2, replace=False)
            o1 = seed_anomalies[idx[0]]
            o2 = seed_anomalies[idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            inj_noise[i] = o1.copy()
            inj_noise[i, swap_feats] = o2[swap_feats]

        x = np.append(x, inj_noise, axis=0)
        y = np.append(y, np.ones(n_inj_noise))

    # remove noise
    elif n_cur_anom > n_adj_anom:
        n_remove = n_cur_anom - n_adj_anom
        print(f'Control Contamination Rate: Removing [{n_remove}] Noise')

        remove_id = anom_idx[rng.choice(n_cur_anom, n_remove, replace=False)]
        print(x.shape)

        x = np.delete(x, remove_id, 0)
        y = np.delete(y, remove_id, 0)
        print(x.shape)

    return x, y


def make_print_to_file(path='./'):

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))


