# -*- coding: utf-8 -*-
"""
Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection
PyTorch's implementation
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import get_network
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np


class FeaWAD(BaseDeepAD):
    """
    Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection
    (TNNLS'21)

    Parameters
    ----------
    epochs: int, optional (default=100)
        Number of training epochs

    batch_size: int, optional (default=64)
        Number of samples in a mini-batch

    lr: float, optional (default=1e-3)
        Learning rate

    rep_dim: int, optional (default=128)
        Dimensionality of the representation space

    hidden_dims: list, str or int, optional (default='100,50')
        Number of neural units in hidden layers
            - If list, each item is a layer
            - If str, neural units of hidden layers are split by comma
            - If int, number of neural units of single hidden layer

    act: str, optional (default='ReLU')
        activation layer name
        choice = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']

    bias: bool, optional (default=False)
        Additive bias in linear layer

    margin: float, optional (default=5.)
        margin value used in the deviation loss function

    epoch_steps: int, optional (default=-1)
        Maximum steps in an epoch
            - If -1, all the batches will be processed

    prt_steps: int, optional (default=10)
        Number of epoch intervals per printing

    device: str, optional (default='cuda')
        torch device,

    verbose: int, optional (default=1)
        Verbosity mode

    random_state： int, optional (default=42)
        the seed used by the random
    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 margin=5.,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(FeaWAD, self).__init__(
            data_type='tabular', model_name='FeaWAD', epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.margin = margin

        self.rep_dim = rep_dim
        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias

        return

    def training_prepare(self, X, y):
        # loader: balanced loader, a mini-batch contains a half of normal data and a half of anomalies
        n_anom = np.where(y == 1)[0].shape[0]
        n_norm = self.n_samples - n_anom
        weight_map = {0: 1. / n_norm, 1: 1. / n_anom}

        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data, label in dataset],
                                        num_samples=self.batch_size, replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        network_params = {
            'n_features': self.n_features,
            'network': self.network,
            'n_emb': self.rep_dim,
            'n_hidden': self.hidden_dims,
            'n_hidden2': '256,32',
            'activation': self.act,
            'bias': self.bias
        }
        net = FeaWadNet(**network_params).to(self.device)
        criterion = FeaWADLoss(margin=self.margin)
        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x, batch_y = batch_x
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.to(self.device)
        pred, sub_result = net(batch_x)
        loss = criterion(batch_y, pred, sub_result)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        s, _ = net(batch_x)
        s = s.view(-1)
        batch_z = batch_x
        return batch_z, s


class FeaWadNet(torch.nn.Module):
    def __init__(self, n_features, network, n_hidden='500,100', n_hidden2='256,32', n_emb=20,
                 activation='ReLU', bias=False):
        super(FeaWadNet, self).__init__()

        AEmodel_class = get_network('MlpAE')
        FWmodel = get_network('MLP')
        self.AEmodel = AEmodel_class(n_features, n_hidden=n_hidden, n_emb=n_emb,
                                     activation=activation, bias=bias)
        self.LinearModel = FWmodel(n_features+n_emb, n_hidden=n_hidden2, n_output=1,
                                   activation=activation, bias=bias)

    def forward(self, x):
        x2, enc = self.AEmodel(x)
        sub = x2 - x
        sub_norm = torch.norm(sub, p=2, dim=-1)
        sub_norm = torch.unsqueeze(sub_norm, -1)
        sub_result = sub / sub_norm

        concat = torch.concat([sub_result, enc], dim=-1)
        if len(concat.shape) == 3:
            concat = concat[:, -1]
        out = self.LinearModel(concat)

        return out, sub_result


class FeaWADLoss(torch.nn.Module):
    """
    Deviation Loss

    Parameters
    ----------
    margin: float, optional (default=5.)
        Center of the pre-defined hyper-sphere in the representation space

    l: int, optional (default=5000.)
        the size of samples of the Gaussian distribution used in the deviation loss function

    reduction: str, optional (default='mean')
        choice = [``'none'`` | ``'mean'`` | ``'sum'``]
            - If ``'none'``: no reduction will be applied;
            - If ``'mean'``: the sum of the output will be divided by the number of
            elements in the output;
            - If ``'sum'``: the output will be summed

    """
    def __init__(self, margin=5., reduction='mean'):
        super(FeaWADLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        return

    def forward(self, y_true, y_pred, sub_result):
        dev = y_pred
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(torch.maximum(self.margin - dev, torch.tensor(0.)))

        sub_nor = torch.norm(sub_result, p=2, dim=1 if len(sub_result.shape)==2 else [1,2])
        outlier_sub_loss = torch.abs(torch.maximum(self.margin-sub_nor, torch.tensor(0.)))
        loss = (1 - y_true) * (inlier_loss + sub_nor) + y_true * (outlier_loss + outlier_sub_loss)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss

        return loss
