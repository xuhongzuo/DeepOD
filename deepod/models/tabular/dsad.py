# -*- coding: utf-8 -*-
"""
One-class classification
this is partially adapted from https://github.com/lukasruff/Deep-SAD-PyTorch (MIT license)
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np
from collections import Counter


class DeepSAD(BaseDeepAD):
    """ Deep Semi-supervised Anomaly Detection (ICLR'20)

    Parameters
    ----------
    data_type: str, optional (default='tabular')
        Data type

    epochs: int, optional (default=100)
        Number of training epochs

    batch_size: int, optional (default=64)
        Number of samples in a mini-batch

    lr: float, optional (default=1e-3)
        Learning rate

    network: str, optional (default='MLP')
        network structure for different data structures

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

    n_heads: int, optional(default=8):
        number of head in multi-head attention
        used when network='transformer', deprecated in other networks

    d_model: int, optional (default=64)
        number of dimensions in Transformer
        used when network='transformer', deprecated in other networks

    pos_encoding: str, optional (default='fixed')
        manner of positional encoding, deprecated in other networks
        choice = ['fixed', 'learnable']

    norm: str, optional (default='BatchNorm')
        manner of norm in Transformer, deprecated in other networks
        choice = ['LayerNorm', 'BatchNorm']

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
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DeepSAD, self).__init__(
            data_type='tabular', model_name='DeepSAD',
            epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.c = None

        return

    def training_prepare(self, X, y):
        # By following the original paper,
        #   use -1 to denote known anomalies, and 1 to denote known inliers
        known_anom_id = np.where(y == 1)
        y = np.zeros_like(y)
        y[known_anom_id] = -1

        counter = Counter(y)

        if self.verbose >= 2:
            print(f'training data counter: {counter}')

        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(y).long())

        weight_map = {0: 1. / counter[0], -1: 1. / counter[-1]}
        sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data, label in dataset],
                                        num_samples=len(dataset), replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size,
                                  sampler=sampler,
                                  shuffle=True if sampler is None else False)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        net = MLPnet(**network_params).to(self.device)

        self.c = self._set_c(net, train_loader)
        criterion = DSADLoss(c=self.c)

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

        # from collections import Counter
        # tmp = batch_y.data.cpu().numpy()
        # print(Counter(tmp))

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.long().to(self.device)

        z = net(batch_x)
        loss = criterion(z, batch_y)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_z = net(batch_x)
        s = criterion(batch_z)
        return batch_z, s

    def _set_c(self, net, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        net.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)

        # if c is too close to zero, set to +- eps
        # a zero unit can be trivially matched with zero weights
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


class DSADLoss(torch.nn.Module):
    """

    Parameters
    ----------
    c: torch.Tensor
        Center of the pre-defined hyper-sphere in the representation space

    reduction: str, optional (default='mean')
        choice = [``'none'`` | ``'mean'`` | ``'sum'``]
            - If ``'none'``: no reduction will be applied;
            - If ``'mean'``: the sum of the output will be divided by the number of
            elements in the output;
            - If ``'sum'``: the output will be summed

    """

    def __init__(self, c, eta=1.0, eps=1e-6, reduction='mean'):
        super(DSADLoss, self).__init__()
        self.c = c
        self.reduction = reduction
        self.eta = eta
        self.eps = eps

    def forward(self, rep, semi_targets=None, reduction=None):
        dist = torch.sum((rep - self.c) ** 2, dim=1)

        if semi_targets is not None:
            loss = torch.where(semi_targets == 0, dist,
                               self.eta * ((dist+self.eps) ** semi_targets.float()))
        else:
            loss = dist

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
