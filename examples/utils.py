import numpy as np
import pandas as pd
import sklearn
import sklearn.impute
from sklearn.model_selection import train_test_split


def train_test_val_split(x, y, ratio_test, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio_test, random_state=random_state)
    return X_train, y_train, X_test, y_test


def get_data(rate, FEATURE_FILE, TARGET_FILE, TRAIN_IDS, TEST_IDS):
    # 提取target
    df = pd.read_csv(TARGET_FILE)
    Y = df.values[:, 1:].astype(np.float64)
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    Y = imp.fit(Y).transform(Y).T

    # 提取feature
    df = pd.read_csv(FEATURE_FILE)
    df = df.drop(['Data'], axis=1)
    df = df.fillna(0)
    df = min_max_norm(df)
    df = df.fillna(0)
    X = df.values.astype(np.float32)
    Xtrain, ytrain, Xtest, ytest = train_test_val_split(X, Y, rate)

    return ytrain, ytest, Xtrain, Xtest


def min_max_norm(df):
    return (df - df.min()) / (df.max() - df.min())


if __name__ == '__main__':
    get_data()