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
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 rep_dim=128, hidden_dims=[100,50], act='ReLU',
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
        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = MLPnet(
            n_features=self.n_features, n_hidden=self.hidden_dims, n_output=self.rep_dim,
            activation=self.act, skip_connection=None,
        ).to(self.device)

        rp_net = copy.deepcopy(net)
        criterion = RDPLoss(rp_net)

        if self.verbose >=2:
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

    clf = RDP(epochs=10, device='cpu')
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_score=scores, y_true=y)

    print(auc)