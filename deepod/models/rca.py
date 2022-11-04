"""
RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection
this script is partially adapted from https://hub.nuaa.cf/illidanlab/RCA
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


class RCA(BaseDeepAD):
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 beta=0.,
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

        self.beta = beta
        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = RCANet(
            self.n_features,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation='ReLU',
            bias=False,
            dropout=None
        ).to(self.device)

        criterion = torch.nn.MSELoss(reduction='mean')

        if self.verbose >=2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        x = batch_x.float().to(self.device)

        n_selected = int(x.shape[0] * (1-self.beta))
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

    def epoch_update(self):
        # the anomaly ratio is normally unknown, set 0.05 by default
        anomaly_ratio = 0.05
        alpha = 0.1
        self.beta = self.beta - anomaly_ratio / (alpha * self.epochs)
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


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    file = '../../data/38_thyroid.npz'
    data = np.load(file, allow_pickle=True)
    x, y = data['X'], data['y']
    y = np.array(y, dtype=int)

    anom_id = np.where(y==1)[0]
    known_anom_id = np.random.choice(anom_id, 30)
    y_semi = np.zeros_like(y)
    y_semi[known_anom_id] = 1

    clf = RCA(device='cpu')
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_score=scores, y_true=y)

    print(auc)
