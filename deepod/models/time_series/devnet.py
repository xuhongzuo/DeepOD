# -*- coding: utf-8 -*-
"""
Deep anomaly detection with deviation networks.
PyTorch's implementation
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import get_network
from deepod.models.tabular.devnet import DevLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np


class DevNetTS(BaseDeepAD):
    """
    Deviation Networks for Weakly-supervised Anomaly Detection (KDD'19)
    :cite:`pang2019deep`

    Parameters
    ----------
    epochs: int, optional (default=100)
        Number of training epochs

    batch_size: int, optional (default=64)
        Number of samples in a mini-batch

    lr: float, optional (default=1e-3)
        Learning rate

    rep_dim: int, optional (default=128)
        it is for consistency, unused in this model

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

    margin: float, optional (default=5.)
        margin value used in the deviation loss function

    l: int, optional (default=5000.)
        the size of samples of the Gaussian distribution used in the deviation loss function

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
                 network='Transformer', seq_len=100, stride=1,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 n_heads=8, d_model=512, attn='self_attn', pos_encoding='fixed', norm='LayerNorm',
                 margin=5., l=5000,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DevNetTS, self).__init__(
            data_type='ts', model_name='DevNet', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.margin = margin
        self.l = l

        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias

        # parameters for Transformer
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = attn
        self.pos_encoding = pos_encoding
        self.norm = norm

        return

    def training_prepare(self, X, y):
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
        if self.network == 'Transformer':
            network_params['n_heads'] = self.n_heads
            network_params['d_model'] = self.d_model
            network_params['pos_encoding'] = self.pos_encoding
            network_params['norm'] = self.norm
            network_params['attn'] = self.attn
            network_params['seq_len'] = self.seq_len
        elif self.network == 'ConvSeq':
            network_params['seq_len'] = self.seq_len

        network_class = get_network(self.network)
        net = network_class(**network_params).to(self.device)

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
