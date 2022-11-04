# -*- coding: utf-8 -*-
"""
Neural Transformation Learning-based Anomaly Detection
this script is partially adapted from https://github.com/boschresearch/NeuTraL-AD (AGPL-3.0 license)
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np


class NeuTraL(BaseDeepAD):
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 n_trans=11, trans_type='residual', temp=0.1,
                 rep_dim=24, hidden_dims='24,24,24,24', trans_hidden_dims=24,
                 act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=1, random_state=42):
        super(NeuTraL, self).__init__(
            model_name='NeuTraL', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.n_trans = n_trans
        self.trans_type = trans_type
        self.temp = temp

        self.trans_hidden_dims = trans_hidden_dims
        self.enc_hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias
        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = TabNeutralADNet(
            n_features=self.n_features,
            n_trans=self.n_trans,
            trans_type=self.trans_type,
            enc_hidden_dims=self.enc_hidden_dims,
            trans_hidden_dims=self.trans_hidden_dims,
            activation=self.act,
            bias=self.bias,
            rep_dim=self.rep_dim,
            device=self.device
        )

        criterion = DCL(temperature=self.temp)

        if self.verbose >=2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, shuffle=False)
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


class TabNeutralADNet(torch.nn.Module):
    """
    network class of NeuTraL for tabular data

    Parameters
    ----------
    n_features: int
        dimensionality of input data

    n_trans: int
        the number of transformation times

    trans_type: str, default='residual'
        transformation type

    enc_hidden_dims: list or str or int
        the number of neural units of hidden layers in encoder net

    trans_hidden_dims: list or str or int
        the number of neural units of hidden layers in transformation net

    rep_dim: int
        representation dimensionality

    activation: str
        activation layer name

    device: str
        device
    """
    def __init__(self, n_features, n_trans=11, trans_type='residual',
                 enc_hidden_dims='24,24,24,24', trans_hidden_dims=24,
                 rep_dim=24,
                 activation='ReLU',
                 bias=False,
                 device='cuda'):
        super(TabNeutralADNet, self).__init__()

        self.enc = MLPnet(
            n_features=n_features,
            n_hidden=enc_hidden_dims,
            n_output=rep_dim,
            activation=activation,
            bias=bias,
            batch_norm=False
        )
        self.trans = torch.nn.ModuleList(
            [MLPnet(n_features=n_features,
                    n_hidden=trans_hidden_dims,
                    n_output=n_features,
                    activation=activation,
                    bias=bias,
                    batch_norm=False) for _ in range(n_trans)]
        )

        self.trans.to(device)
        self.enc.to(device)

        self.n_trans = n_trans
        self.trans_type = trans_type
        self.z_dim = rep_dim

    def forward(self, x):
        x_transform = torch.empty(x.shape[0], self.n_trans, x.shape[-1]).to(x)

        for i in range(self.n_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_transform[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_transform[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_transform[:, i] = mask + x

        x_cat = torch.cat([x.unsqueeze(1), x_transform], 1)
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.n_trans+1, self.z_dim)
        return zs


class DCL(torch.nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super(DCL, self).__init__()
        self.temp = temperature
        self.reduction = reduction

    def forward(self, z):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, n_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(n_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, n_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp) # n,k-1
        K = n_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))

        loss = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        loss = loss.sum(1)

        reduction = self.reduction
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss

        return loss



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

    clf = NeuTraL(device='cpu')
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_score=scores, y_true=y)

    print(auc)