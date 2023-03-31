# -*- coding: utf-8 -*-
"""
Random distance prediction-based anomaly detection
this script is partially adapted from https://github.com/billhhh/RDP
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import copy


class RDP(BaseDeepAD):
    """
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
                 rep_dim=128, hidden_dims='100,50', act='LeakyReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(RDP, self).__init__(
            model_name='RDP', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias
        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = MLPnet(
            n_features=self.n_features,
            n_hidden=self.hidden_dims, n_output=self.rep_dim,
            activation=self.act, bias=self.bias,
            skip_connection=None,
        ).to(self.device)

        rp_net = copy.deepcopy(net)
        criterion = RDPLoss(rp_net)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x1 = batch_x[torch.randperm(batch_x.shape[0])]
        batch_x = batch_x.float().to(self.device)
        batch_x1 = batch_x1.float().to(self.device)
        z, z1 = net(batch_x), net(batch_x1)
        loss = criterion(z, z1, batch_x, batch_x1)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_x1 = batch_x[torch.randperm(batch_x.shape[0])]
        batch_z, batch_z1 = net(batch_x), net(batch_x1)
        s = criterion(batch_z, batch_z1, batch_x, batch_x1)
        return batch_z, s


class RDPLoss(torch.nn.Module):
    def __init__(self, random_projection_net, reduction='mean'):
        super(RDPLoss, self).__init__()
        self.rp_net = random_projection_net
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, rep, rep1, x, x1):
        rep_target = self.rp_net(x)
        rep1_target = self.rp_net(x1)

        d_target = torch.sum(F.normalize(rep_target, p=1, dim=1) *
                             F.normalize(rep1_target, p=1, dim=1), dim=1)
        d_pred = torch.sum(F.normalize(rep, p=1, dim=1) *
                           F.normalize(rep1, p=1, dim=1), dim=1)

        if self.reduction == 'mean' or self.reduction == 'sum':
            gap_loss = self.mse(rep, rep_target)
            rdp_loss = self.mse(d_target, d_pred)

        else:
            gap_loss = torch.mean(F.mse_loss(rep, rep_target, reduction='none'), dim=1)
            rdp_loss = F.mse_loss(d_target, d_pred, reduction='none')

        return gap_loss + rdp_loss
