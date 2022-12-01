# -*- coding: utf-8 -*-
"""
One-class classification
this is partially adapted from
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


class DeepSAD(BaseDeepAD):
    """ Deep Semi-supervised Anomaly Detection (Deep SAD)
    See :cite:`ruff2020dsad` for details

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
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DeepSAD, self).__init__(
            model_name='DeepSAD', epochs=epochs, batch_size=batch_size, lr=lr,
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

        semi_y2 = y.copy()
        semi_y2[np.where(y == 1)[0]] = -1
        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(semi_y2).long())

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        net = MLPnet(
            n_features=self.n_features,
            n_hidden=self.hidden_dims,
            n_output=self.rep_dim,
            activation=self.act,
            bias=self.bias,
        ).to(self.device)

        # self.c = torch.randn(net.n_emb).to(self.device)
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
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.to(self.device)
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

        # if c i s too close to zero, set to +- eps
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


if __name__ == '__main__':
    import numpy as np

    file = '../../data/38_thyroid.npz'
    data = np.load(file, allow_pickle=True)
    x, y = data['X'], data['y']
    y = np.array(y, dtype=int)

    anom_id = np.where(y == 1)[0]
    known_anom_id = np.random.choice(anom_id, 30)
    y_semi = np.zeros_like(y)
    y_semi[known_anom_id] = 1

    clf = DeepSAD(device='cpu')
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_score=scores, y_true=y)

    print(auc)
