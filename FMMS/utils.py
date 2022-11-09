import numpy as np
import pandas as pd
import config
import torch
import sklearn
import sklearn.impute
from sklearn.model_selection import train_test_split


def train_test_val_split(x, y, ratio_test):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio_test, random_state=config.random_state)
    return X_train, y_train, X_test, y_test


def get_data2(rate=0.1):
    FEATURE_FILE, TARGET_FILE, TRAIN_IDS, TEST_IDS = config.get_path()
    # 提取target
    df = pd.read_csv(TARGET_FILE)
    Y = df.values[:, 1:].astype(np.float64)
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    Y = imp.fit(Y).transform(Y).T         # 替换为对应列的均值，全nan则删除

    # 提取feature
    df = pd.read_csv(FEATURE_FILE)
    df = df.drop(['Data'], axis=1)
    df = df.fillna(0)
    df = MinMaxNorm(df)     # 归一化
    df = df.fillna(0)
    X = df.values.astype(np.float32)
    Xtrain, ytrain, Xtest, ytest = train_test_val_split(X, Y, rate)

    return ytrain, ytest, Xtrain, Xtest


def get_data():
    FEATURE_FILE, TARGET_FILE, TRAIN_IDS, TEST_IDS = config.get_path()
    # 提取target
    df = pd.read_csv(TARGET_FILE)
    dataset_ids = df.columns.tolist()[1:]
    dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
    # df = df.fillna(0)
    Y = df.values[:, 1:].astype(np.float64)

    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    Y = imp.fit(Y).transform(Y)         # 替换为对应列的均值，全nan则删除

    ids_train = np.loadtxt(TRAIN_IDS).astype(int).tolist()
    ids_test = np.loadtxt(TEST_IDS).astype(int).tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ytrain = Y[:, ix_train]
    Ytest = Y[:, ix_test]

    # 提取feature
    df = pd.read_csv(FEATURE_FILE)
    dataset_ids = df[df.columns[0]].tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    df = df.fillna(0)
    df = MinMaxNorm(df)     # 归一化
    df = df.fillna(0)
    X = df.values.astype(np.float32)

    # impx = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    # X = impx.fit(X).transform(X)         # 替换为对应列的均值

    Ftrain = X[ix_train, 1:]
    Ftest = X[ix_test, 1:]

    return Ytrain.T, Ytest.T, Ftrain, Ftest


def MinMaxNorm(df):     # 最大最小归一
    return (df - df.min()) / (df.max() - df.min())


def scaleY(Y):
    return -1/Y


def topn_loss(ypred, yreal, n=10):
    data_size, model_size = yreal.shape

    ypred_idx = np.array([np.argmax(ypred[i]) for i in range(data_size)])    # 每个data选出最大model对应的idx
    ypred_max = np.array([yreal[i][idx] for i, idx in enumerate(ypred_idx)])

    # 选出实际上的topn
    topn = np.ones((data_size, n))          # data*n
    for ii in range(data_size):             # 对每个数据
        best_value = list(np.sort(yreal[ii])[::-1])         # 第ii列，即第ii个数据（nan已经被填充为0，因此不会被排为最大，当所有数都是nan时，也不会报错）
        topn[ii] = best_value[:n]                           # 第ii个数据的前n个最大值

    correct = 0
    for ii, pred in enumerate(ypred_max):  # 对每个数据
        if pred in topn[ii]:
            correct += 1

    return -correct/data_size


def rmse_loss(pred, real):
    # 使用均方根误差作为评价指标
    # sum(x2-y2)
    loss_func = torch.nn.MSELoss(reduction='mean')
    mse_loss = loss_func(pred, torch.tensor(real).float())
    loss = torch.sqrt(mse_loss)
    return loss


def mse_loss(pred, real):
    loss_func = torch.nn.MSELoss(reduction='mean')
    loss = loss_func(pred, torch.tensor(real).float())
    return loss


def cos_loss(pred, real):
    pred = pred.type(torch.DoubleTensor)
    real = torch.tensor(real).double()
    mul = pred * real
    mul = torch.sum(mul, dim=1)
    length = torch.norm(pred, p=2, dim=1) * torch.norm(real, p=2, dim=1)
    loss = 1 - mul/length
    loss = sum(loss)/len(pred)
    return loss


def l1_loss(pred, real):
    L1 = torch.nn.L1Loss()
    loss = L1(pred, torch.tensor(real).float())
    return loss


def SmoothL1_loss(pred, real):
    sl1 = torch.nn.SmoothL1Loss()
    loss = sl1(pred, torch.tensor(real).float())
    return loss


def KLDiv_loss(pred, real):
    kd = torch.nn.KLDivLoss()
    loss = kd(pred, torch.tensor(real).float())
    return loss


def DCG_loss(pred, real):
    data_size, model_size = real.shape
    treal = torch.tensor(real).double()
    DCG = torch.zeros(1).type(torch.DoubleTensor)
    for ii in range(data_size):
        for jj in range(model_size):
            part = sum(torch.sigmoid(pred[ii] - pred[ii][jj])).type(torch.DoubleTensor)        # 统计比jj小的数量
            DCG = DCG + (torch.pow(10, treal[ii][jj])-1)/torch.log2(1+part)
    return -DCG/data_size


if __name__ == '__main__':
    get_data()