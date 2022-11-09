import os
import torch
import utils

dataset = 'pmf'
random_state = 0
modelnum = 200


def get_rate():
    if dataset == 'pmf':
        rate = 0.1
    else:
        rate = 0.2
    return rate


def get_path():
    FEATURE_FILE = os.path.join("data", dataset, "feature.csv")
    TARGET_FILE = os.path.join("data", dataset, "target.csv")
    TRAIN_IDS = os.path.join("data", dataset, "ids_train.csv")
    TEST_IDS = os.path.join("data", dataset, "ids_test.csv")
    return FEATURE_FILE, TARGET_FILE, TRAIN_IDS, TEST_IDS


def get_para(train_params, params):
    return '%s_%s_%s_%s_%s_%s' % (train_params['optname'],
                                  train_params['lossname'],
                                  str(params['embedding_size']),
                                  str(train_params['batch']),
                                  str(train_params['epoch']),
                                  str(train_params['lr']))