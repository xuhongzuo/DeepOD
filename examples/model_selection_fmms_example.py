import examples.utils as utils
from deepod.model_selection.run_FMMS import run_FMMS
import os

random_state = 0
modelnum = 200


def get_path():
    FEATURE_FILE = os.path.join("data/feature.csv")
    TARGET_FILE = os.path.join("data/target.csv")
    return FEATURE_FILE, TARGET_FILE


if __name__ == '__main__':
    FEATURE_FILE, TARGET_FILE = get_path()
    ptrain, ptest, ftrain, ftest = utils.get_data(0.1, FEATURE_FILE, TARGET_FILE)
    ptrain = ptrain[:, :modelnum]
    ptest = ptest[:, :modelnum]
    ftrain, ptrain, fvalid, pvalid = utils.train_test_val_split(ftrain, ptrain, 0.1)

    rfmms = run_FMMS(ftrain, ptrain, fvalid, pvalid)    # 有早停
    rfmms = run_FMMS(ftrain, ptrain)                    # 无早停
    rfmms.fit(save=True)
    rfmms.predict(f=ftest[0], load=True)
