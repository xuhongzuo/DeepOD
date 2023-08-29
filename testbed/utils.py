import os
import glob
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


intermediate_dir = './z-intermediate_model_files/'

# --------------------------- evaluation --------------------------- #


def get_best_f1(label, score):
    precision, recall, ths = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    best_th = ths[np.argmax(f1)]
    return best_f1, best_p, best_r, best_th


def get_metrics(label, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    best_f1, best_p, best_r, _ = get_best_f1(label, score)

    return auroc, ap, best_f1, best_p, best_r


def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]
    :param score - anomaly score, higher score indicates higher likelihoods to be anomaly
    :param label - ground-truth label
    """
    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score


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


# --------------------------- data preprocessing --------------------------- #


def data_standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                    X_train[col].unique(),
                    X_test[col].unique()))
    X_train = X_train.values
    X_test = X_test.values

    return X_train, X_test


def data_normalization(x_train, x_test):
    x_train = x_train.values
    x_test = x_test.values

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


# --------------------------- data loading --------------------------- #


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


def get_data_lst(dataset_dir, dataset):
    if dataset == 'FULL':
        print(os.path.join(dataset_dir, '*.*'))
        data_lst = glob.glob(os.path.join(dataset_dir, '*.*'))
    else:
        name_lst = dataset.split(',')
        data_lst = []
        for d in name_lst:
            data_lst.extend(glob.glob(os.path.join(dataset_dir, d + '.*')))
    data_lst = sorted(data_lst)
    return data_lst


def import_ts_data_unsupervised(data_root, data, entities=None, combine=False):
    if type(entities) == str:
        entities_lst = entities.split(',')
    elif type(entities) == list:
        entities_lst = entities
    else:
        raise ValueError('wrong entities')

    name_lst = []
    train_lst = []
    test_lst = []
    label_lst = []

    if len(glob.glob(os.path.join(data_root, data) + '/*.csv')) == 0:
        machine_lst = os.listdir(data_root + data + '/')
        for m in sorted(machine_lst):
            if entities != 'FULL' and m not in entities_lst:
                continue
            train_path = glob.glob(os.path.join(data_root, data, m, '*train*.csv'))
            test_path = glob.glob(os.path.join(data_root, data, m, '*test*.csv'))

            assert len(train_path) == 1 and len(test_path) == 1, f'{m}'
            train_path, test_path = train_path[0], test_path[0]

            train_df = pd.read_csv(train_path, sep=',', index_col=0)
            test_df = pd.read_csv(test_path, sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

            # normalization
            train, test = data_standardize(train_df, test_df)

            train_lst.append(train)
            test_lst.append(test)
            label_lst.append(labels)
            name_lst.append(m)

        if combine:
            train_lst = [np.concatenate(train_lst)]
            test_lst = [np.concatenate(test_lst)]
            label_lst = [np.concatenate(label_lst)]
            name_lst = [data + '_combined']

    else:
        train_df = pd.read_csv(f'{data_root}{data}/{data}_train.csv', sep=',', index_col=0)
        test_df = pd.read_csv(f'{data_root}{data}/{data}_test.csv', sep=',', index_col=0)
        labels = test_df['label'].values
        train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)
        train, test = data_standardize(train_df, test_df)

        train_lst.append(train)
        test_lst.append(test)
        label_lst.append(labels)
        name_lst.append(data)

    return train_lst, test_lst, label_lst, name_lst


def get_anom_pairs(y):
    anom_pairs = []
    anom_index = np.where(y == 1)[0]
    tmp_seg = []
    for i in anom_index:
        tmp_seg.append(i)
        if i + 1 not in anom_index:
            anom_pairs.append((tmp_seg[0], tmp_seg[-1]))
            tmp_seg = []
    return anom_pairs


def set_semi_supervised(y_label, n_known=1, seed=42):
    semi_y = np.zeros_like(y_label)

    anom_pairs = get_anom_pairs(y_label)
    print()
    print(f'This Dataset has [{len(anom_pairs)}] anom pairs in total')
    if n_known == -1:
        n_known = len(anom_pairs)
    n_known = min(n_known, len(anom_pairs))

    rng = np.random.RandomState(seed)
    pair_ids = rng.choice(len(anom_pairs), n_known)
    known_segs = []
    for p in pair_ids:
        segment = anom_pairs[p]
        semi_y[segment[0]: segment[1]+1] = 1
        known_segs.append(segment)

    print(f'known segment: {known_segs}, {int(np.sum(semi_y))} / {len(semi_y)}')
    return semi_y
