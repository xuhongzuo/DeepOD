from deepod.models.time_series import *
from deepod.models.time_series.seq_reg_ad import SeqRegAD


def get_model_class(name):
    if name == 'dif':
        return DeepIsolationForestTS
    elif name == 'dsvdd':
        return DeepSVDDTS
    elif name == 'tranad':
        return TranAD
    elif name == 'usad':
        return USAD
    elif name == 'couta':
        return COUTA
    elif name == 'seqregad':
        return SeqRegAD


def get_additional_configs(name):
    config_dict = {
        'seqregad': {
            'seq_len_lst': [10, 30, 50],
            'epochs': 10,
            'batch_size': 64,
            'lr': 1e-4,
            'rep_dim': 128,
        },

        'dif': {
            'rep_dim': 20,
            'hidden_dims': 32,
            'n_ensemble': 50,
            'n_estimators': 6,
        },

        'dsvdd': {
            'network': 'Transformer',
            'rep_dim': 64,
            'hidden_dims': '512',
            'act': 'GELU',
            'lr': 1e-5,
            'epochs': 20,
            'batch_size': 128,
            'epoch_steps': -1,
        },

        'tranad': {
            'lr': 1e-3,
            'epochs': 10,
            'batch_size': 128,
            'epoch_steps': -1,
        },

        'usad': {
            'hidden_dims': 100,
            'lr': 1e-3,
            'epochs': 10,
            'batch_size': 128,
        },

        'couta': {
            'neg_batch_ratio': 0.2,
            'alpha': 0.1,
            'rep_dim': 16,
            'hidden_dims': 16,
            'lr': 1e-4,
            'epochs': 20,
            'batch_size': 64,
        }
    }

    try:
        return config_dict[name]
    except KeyError:
        return {}
