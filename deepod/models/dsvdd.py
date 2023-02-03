# -*- coding: utf-8 -*-
"""
One-class classification
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import get_network
from torch.utils.data import DataLoader
import torch


class DeepSVDD(BaseDeepAD):
    """ Deep One-class Classification (Deep SVDD) for anomaly detection
    See :cite:`ruff2018deepsvdd` for details

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

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data
        deprecated when handling tabular data (network=='MLP')

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences
        deprecated when handling tabular data (network=='MLP')

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
    def __init__(self, data_type='tabular', epochs=100, batch_size=64, lr=1e-3,
                 network='MLP', seq_len=100, stride=1,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DeepSVDD, self).__init__(
            model_name='DeepSVDD', data_type=data_type, epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
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
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        network_class = get_network(self.network)
        net = network_class(**network_params).to(self.device)

        # self.c = torch.randn(net.n_emb).to(self.device)
        self.c = self._set_c(net, train_loader)
        criterion = DSVDDLoss(c=self.c)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        z = net(batch_x)
        loss = criterion(z)
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
            for x in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


class DSVDDLoss(torch.nn.Module):
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
    def __init__(self, c, reduction='mean'):
        super(DSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, reduction=None):
        loss = torch.sum((rep - self.c) ** 2, dim=1)

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
