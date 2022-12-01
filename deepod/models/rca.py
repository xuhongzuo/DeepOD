"""
RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection
this script is partially adapted from https://hub.nuaa.cf/illidanlab/RCA
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from tqdm import trange
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np


class RCA(BaseDeepAD):
    """
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

    alpha: float, optional (default=0.5)
        decay rate in determining beta

    anom_ratio: float, optional (default=0.5)
        decay rate in determining beta

    dropout: float or None, optional (default=0.5)
        dropout probability, the default setting is 0.5

    inference_ensemble: int, optional(default=10)
        the ensemble size during the inference stage

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
                 alpha=0.5, anom_ratio=0.02, dropout=0.5, inference_ensemble=10,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(RCA, self).__init__(
            model_name='RCA', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.inference_ensemble = inference_ensemble

        self.anom_ratio = anom_ratio
        self.beta = 1.
        self.alpha = alpha
        self.dropout = dropout
        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = RCANet(
            self.n_features,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            bias=False,
            dropout=self.dropout
        ).to(self.device)

        criterion = torch.nn.MSELoss(reduction='mean')

        if self.verbose >=2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        x = batch_x.float().to(self.device)

        n_selected = int(x.shape[0] * self.beta)
        net.eval()
        with torch.no_grad():
            z1, z2, x_hat1, x_hat2 = net(x, x)

            error1 = F.mse_loss(x_hat1, x, reduction='none')
            error2 = F.mse_loss(x_hat2, x, reduction='none')

            error1 = error1.sum(dim=1)
            error2 = error2.sum(dim=1)
            _, index1 = torch.sort(error1)
            _, index2 = torch.sort(error2)

            index1 = index1[:n_selected]
            index2 = index2[:n_selected]

            x1 = x[index2, :]
            x2 = x[index1, :]

        net.train()
        z1, z2, x_hat1, x_hat2 = net(x1.float(), x2.float())
        loss = criterion(x_hat1, x1) + criterion(x_hat2, x2)

        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        z1, z2, x_hat1, x_hat2 = net(batch_x, batch_x)
        batch_z = z1 + z2

        s = criterion(x_hat1, batch_x) + criterion(x_hat2, batch_x)
        s = torch.sum(s, dim=1)

        return batch_z, s

    def _inference(self):
        #   As introduced in the published paper, dropout is used to
        # emulate the ensemble process.
        #   RCA employs dropout during testing by using many networks of
        # perturbed structures to perform multiple forward passes over
        # the data in order to obtain a set of reconstruction errors for
        # each test point.

        repeat_times = self.inference_ensemble if self.dropout is not None else 1

        s_lsts = []
        self.net.train()
        for _ in trange(repeat_times):
            with torch.no_grad():
                z_lst = []
                score_lst = []
                for batch_x in self.test_loader:
                    batch_z, s = self.inference_forward(batch_x, self.net, self.criterion)
                    z_lst.append(batch_z)
                    score_lst.append(s)
            z = torch.cat(z_lst).data.cpu().numpy()
            s = torch.cat(score_lst).data.cpu().numpy()
            s_lsts.append(s)
        scores = np.array(s_lsts).mean(axis=0)
        return z, scores

    def epoch_update(self):
        if self.anom_ratio is not None:
            self.beta = self.beta - self.anom_ratio / (self.alpha * self.epochs)
            self.beta = max(1.-self.anom_ratio, self.beta)
        if self.verbose >=2:
            print(f'beta: {self.beta}')
        return

class RCANet(torch.nn.Module):
    def __init__(self, n_features, hidden_dims='100',
                 rep_dim=24,
                 activation='ReLU',
                 bias=False, dropout=None):
        super(RCANet, self).__init__()

        if type(hidden_dims)==int:
            hidden_dims = [hidden_dims]
        if type(hidden_dims)==str:
            hidden_dims = hidden_dims.split(',')
            hidden_dims = [int(a) for a in hidden_dims]

        self.enc1 = MLPnet(
            n_features=n_features,
            n_hidden=hidden_dims,
            n_output=rep_dim,
            activation=activation,
            bias=bias,
            dropout=dropout
        )

        self.enc2 = MLPnet(
            n_features=n_features,
            n_hidden=hidden_dims,
            n_output=rep_dim,
            activation=activation,
            bias=bias,
            dropout=dropout
        )

        self.dec1 = MLPnet(
            n_features=rep_dim,
            n_hidden=hidden_dims[::-1],
            n_output=n_features,
            activation=activation,
            bias=bias,
            dropout=dropout
        )

        self.dec2 = MLPnet(
            n_features=rep_dim,
            n_hidden=hidden_dims[::-1],
            n_output=n_features,
            activation=activation,
            bias=bias,
            dropout=dropout
        )

        return

    def forward(self, x1, x2):
        z1 = self.enc1(x1)
        x_hat1 = self.dec1(z1)

        # z2 = self.enc1(x2)
        # x_hat2 = self.dec1(z2)

        z2 = self.enc2(x2)
        x_hat2 = self.dec2(z2)
        return z1, z2, x_hat1, x_hat2
