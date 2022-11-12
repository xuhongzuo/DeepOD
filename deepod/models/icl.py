# -*- coding: utf-8 -*-
"""
Anomaly Detection for Tabular Data with Internal Contrastive Learning
this script is partially adapted from the supplementary material in
https://openreview.net/forum?id=_hszZbt46bT
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np


class ICL(BaseDeepAD):
    """ Anomaly Detection for Tabular Data with Internal Contrastive Learning (ICL for short)
    See :cite:`shenkar2022internal` for details

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

    hidden_dims: List, str or int, optional (default='100,50')
        Number of neural units in hidden layers
            - If List, each item is a layer
            - If str, neural units of hidden layers are split by comma
            - If int, number of neural units of single hidden layer

    act: str, optional (default='ReLU')
        activation layer name
        choice = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']

    bias: bool, optional (default=False)
        Additive bias in linear layer

    kernel_size: str or int, optional (default='auto')
        the length of sub-vectors

    temperature: float, optional (default=0.01)
        tau in the cross-entropy function

    max_negatives: int, optional (default=1000)
        Maximum number of negatives (unmatched sub-vectors)

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
                 kernel_size=2, temperature=0.01, max_negatives=1000,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=1, random_state=42):
        super(ICL, self).__init__(
            model_name='DeepSVDD', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.kernel_size = kernel_size
        self.tau = temperature
        self.max_negatives = max_negatives

        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size,
                                  shuffle=True, pin_memory=True)

        net = ICLNet(
            n_features=self.n_features,
            kernel_size=self.kernel_size,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            bias=self.bias
        ).to(self.device)

        criterion = torch.nn.CrossEntropyLoss()

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)

        # positives are sub-vectors, query are their complements
        positives, query = net(batch_x)

        logit = self.cal_logit(query, positives)
        logit = logit.permute(0, 2, 1)

        correct_class = torch.zeros((logit.shape[0], logit.shape[2]),
                                    dtype=torch.long).to(self.device)
        loss = criterion(logit, correct_class)
        return loss

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def inference_forward(self, batch_x, net, criterion):
        loss = self.training_forward(batch_x, net, criterion)
        batch_z = batch_x # for consistency
        s = loss.mean(dim=1)
        return batch_z, s

    def cal_logit(self, query, pos):
        n_pos = query.shape[1]
        batch_size = query.shape[0]

        # get negatives
        negative_index = np.random.choice(np.arange(n_pos), min(self.max_negatives, n_pos), replace=False)
        negative = pos.permute(0, 2, 1)[:, :, negative_index]

        pos_multiplication = (query * pos).sum(dim=2).unsqueeze(2)

        neg_multiplication = torch.matmul(query, negative)  # [batch_size, n_neg, n_neg]

        # Removal of the diagonals
        identity_matrix = torch.eye(n_pos).unsqueeze(0).to(self.device)
        identity_matrix = identity_matrix.repeat(batch_size, 1, 1)
        identity_matrix = identity_matrix[:, :, negative_index]

        neg_multiplication.masked_fill_(identity_matrix==1, -float('inf'))

        logit = torch.cat((pos_multiplication, neg_multiplication), dim=2)
        logit = torch.div(logit, self.tau)
        return logit


class ICLNet(torch.nn.Module):
    def __init__(self, n_features, kernel_size,
                 hidden_dims='100,50', rep_dim=64,
                 activation='ReLU', bias=False):
        super(ICLNet, self).__init__()
        self.n_features = n_features
        self.kernel_size = kernel_size

        # @TODO: dimensionality in batch_norm layer for 3-d vectors
        # @TODO: first layer's activation is tanh, others are leaky_relu

        self.enc_f_net = MLPnet(
            n_features=n_features-kernel_size,
            n_hidden=hidden_dims,
            n_output=rep_dim,
            batch_norm=False,
            activation=activation,
            bias=bias,
        )

        self.enc_g_net = MLPnet(
            n_features=kernel_size,
            n_hidden=hidden_dims,
            n_output=rep_dim,
            batch_norm=False,
            activation=activation,
            bias=bias,
        )

        return

    def forward(self, x):
        x1, x2 = self.positive_matrix_builder(data=x)
        x1 = self.enc_g_net(x1)
        x2 = self.enc_f_net(x2)
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        return x1, x2

    def positive_matrix_builder(self, data):
        """
        Generate matrix of sub-vectors and matrix of complement vectors (positive pairs)

        Parameters
        ----------
        data: torch.Tensor shape (n_samples, n_features), required
            The input data.

        Returns
        -------
        matrix: torch.Tensor of shape [n_samples, number of sub-vectors, kernel_size]
            Derived sub-vectors.

        complement_matrix: torch.Tensor of shape [n_samples, number of sub-vectors, n_features-kernel_size]
            Complement vector of derived sub-vectors.

        """
        kernel_size = self.kernel_size
        dim = self.n_features

        data = torch.unsqueeze(data, 1)  # [size, 1, dim]
        data = data.repeat(1, dim, 1)  # [size, dim, dim]

        # get consecutive subspace indices and the corresponding complement indices
        start_idx = np.arange(dim)[: -kernel_size + 1]  # [0,1,2,...,dim-kernel_size+1]
        all_idx = start_idx[:, None] + np.arange(kernel_size)
        all_idx_complement = np.array([np.setdiff1d(np.arange(dim), row) for row in all_idx])

        matrix = data[:, np.arange(all_idx.shape[0])[:, None], all_idx]
        complement_matrix = data[:, np.arange(all_idx.shape[0])[:, None], all_idx_complement]

        return matrix, complement_matrix


if __name__ == '__main__':
    import numpy as np

    file = '../../data/38_thyroid.npz'
    data_ = np.load(file, allow_pickle=True)
    x, y = data_['X'], data_['y']
    y = np.array(y, dtype=int)

    anom_id = np.where(y==1)[0]
    known_anom_id = np.random.choice(anom_id, 30)
    y_semi = np.zeros_like(y)
    y_semi[known_anom_id] = 1

    clf = ICL(device='cuda', epochs=10, verbose=2)
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_score=scores, y_true=y)

    print(auc)
