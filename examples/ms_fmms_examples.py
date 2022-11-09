import examples.utils as utils
from deepod.model_selection.fmms import FMMS
import os

dataset = 'pmf'
random_state = 0
modelnum = 200


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


if __name__ == '__main__':
    FEATURE_FILE, TARGET_FILE, TRAIN_IDS, TEST_IDS = get_path()
    ptrain, ptest, ftrain, ftest = utils.get_data(0.1, FEATURE_FILE, TARGET_FILE, TRAIN_IDS, TEST_IDS)
    ptrain = ptrain[:, :modelnum]
    ptest = ptest[:, :modelnum]
    ftrain, ptrain, fvalid, pvalid = utils.train_test_val_split(ftrain, ptrain, 0.1)

    fmms = FMMS(ftrain, ptrain, fvalid, pvalid)
    fmms.fit(save=True)
    fmms.predict(f=ftest[0], load=True)

