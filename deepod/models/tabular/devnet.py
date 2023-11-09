# -*- coding: utf-8 -*-
"""
Deep anomaly detection with deviation networks.
PyTorch's implementation
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np


class DevNet(BaseDeepAD):
    """
    Deviation Networks for Weakly-supervised Anomaly Detection (KDD'19)
    :cite:`pang2019deep`

    Args:
        epochs (int, optional):
            number of training epochs (default: 100).
        batch_size (int, optional):
            number of samples in a mini-batch (default: 64)
        lr (float, optional):
            learning rate (default: 1e-3)
        rep_dim (int, optional):
            it is for consistency, unused in this model.
        hidden_dims (list, str or int, optional):
            number of neural units in hidden layers,
            If list, each item is a layer;
            If str, neural units of hidden layers are split by comma;
            If int, number of neural units of single hidden layer
            (default: '100,50')
        act (str, optional):
            activation layer name,
            choice = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']
            (default='ReLU')
        bias (bool, optional):
            Additive bias in linear layer (default=False)
        margin (float, optional):
            margin value used in the deviation loss function (default=5.)
        l (int, optional):
            the size of samples of the Gaussian distribution
            used in the deviation loss function (default=5000.)
        epoch_steps (int, optional):
            Maximum steps in an epoch.
            If -1, all the batches will be processed
            (default=-1)
        prt_steps (int, optional):
            Number of epoch intervals per printing (default=10)
        device (str, optional):
            torch device (default='cuda').
        verbose (int, optional):
            Verbosity mode (default=1)
        random_state (int, optional):
            the seed used by the random  (default=42)
    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 network='MLP',
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 margin=5., l=5000,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DevNet, self).__init__(
            data_type='tabular', model_name='DevNet', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.margin = margin
        self.l = l

        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias

        return

    def training_prepare(self, X, y):
        """

        Args:
            X (np.array): input data array
            y (np.array): input data label

        Returns:
            train_loader (torch.DataLoader): data loader of training data
            net (torch.nn.Module): neural network
            criterion (torch.nn.Module): loss function

        """
        # loader: balanced loader, a mini-batch contains a half of normal data and a half of anomalies
        n_anom = np.where(y == 1)[0].shape[0]
        n_norm = self.n_samples - n_anom
        weight_map = {0: 1. / n_norm, 1: 1. / n_anom}

        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data, label in dataset],
                                        num_samples=len(dataset), replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': 1,
            'activation': self.act,
            'bias': self.bias
        }
        net = MLPnet(**network_params).to(self.device)

        criterion = DevLoss(margin=self.margin, l=self.l)

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
        pred = net(batch_x)
        loss = criterion(batch_y, pred)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        s = net(batch_x)
        s = s.view(-1)
        batch_z = batch_x
        return batch_z, s


class DevLoss(torch.nn.Module):
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
    def __init__(self, margin=5., l=5000, reduction='mean'):
        super(DevLoss, self).__init__()
        self.margin = margin
        self.loss_l = l
        self.reduction = reduction
        return

    def forward(self, y_true, y_pred):
        ref = torch.randn(self.loss_l)  # from the normal dataset
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(torch.max(self.margin - dev, torch.zeros_like(dev)))
        loss = (1 - y_true) * inlier_loss + y_true * outlier_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss

        return loss
