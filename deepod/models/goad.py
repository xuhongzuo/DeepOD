# -*- coding: utf-8 -*-
"""
Classification-based anomaly detection
this script is partially adapted from https://github.com/lironber/GOAD
License: https://github.com/lironber/GOAD/blob/master/LICENSE
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import ConvNet
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
import numpy as np


class GOAD(BaseDeepAD):
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 n_trans=256, trans_dim=32,
                 alpha=0.1, margin=1., eps=0,
                 kernel_size=1, hidden_dim=8, n_layers=5,
                 act='LeakyReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(GOAD, self).__init__(
            model_name='GOAD', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.n_trans = n_trans
        self.trans_dim = trans_dim

        self.alpha = alpha
        self.margin = margin
        self.eps = eps

        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.act = act
        self.bias = bias

        self.affine_weights = None
        self.rep_means = None
        return

    def training_prepare(self, X, y):
        self.affine_weights = np.random.randn(self.n_trans, self.n_features, self.trans_dim)
        x_trans = np.stack([X.dot(rot) for rot in self.affine_weights], 2) # shape: [n_samples, trans_dim, n_trans]
        x_trans = torch.from_numpy(x_trans).float()
        labels = torch.arange(self.n_trans).unsqueeze(0).expand((X.shape[0], self.n_trans))
        labels = labels.long()

        if self.verbose >= 2:
            print(f'{self.n_trans} transformation done')

        dataset = TensorDataset(x_trans, labels)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        net = GoadNet(
            self.trans_dim,
            kernel_size=self.kernel_size,
            n_hidden=self.hidden_dim,
            n_layers=self.n_layers,
            n_output=self.n_trans,
            activation=self.act, bias=False
        ).to(self.device)
        weights_init(net)

        criterion = GoadLoss(alpha=self.alpha, margin=self.margin, device=self.device)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        x_trans = np.stack([X.dot(rot) for rot in self.affine_weights], 2)
        x_trans = torch.from_numpy(x_trans).float()

        test_loader = DataLoader(x_trans,
                                 batch_size=self.batch_size,
                                 drop_last=False,
                                 shuffle=False)

        # # prepare means:
        self.net.eval()
        sum_reps = torch.zeros((self.hidden_dim, self.n_trans)).to(self.device)
        with torch.no_grad():
            nb = 0
            for batch in self.train_loader:
                batch_data, batch_target = batch
                batch_data = batch_data.float().to(self.device)
                rep, _ = self.net(batch_data)
                sum_reps += rep.mean(0)
                nb += 1

        reps = sum_reps.t() / nb
        reps = reps.unsqueeze(0)
        self.rep_means = reps

        # print(self.rep_means)

        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_data, batch_target = batch_x
        batch_data = batch_data.float().to(self.device)
        batch_target = batch_target.long().to(self.device)

        batch_rep, batch_pred = net(batch_data)
        batch_rep = batch_rep.permute(0, 2, 1)

        loss = criterion(batch_rep, batch_pred, batch_target)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_data = batch_x.float().to(self.device)

        batch_rep, _ = net(batch_data)
        batch_rep = batch_rep.permute(0, 2, 1)

        diffs = ((batch_rep.unsqueeze(2) - self.rep_means) ** 2).sum(-1)
        diffs_eps = self.eps * torch.ones_like(diffs)
        diffs = torch.max(diffs, diffs_eps)

        logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
        s = -torch.diagonal(logp_sz, 0, 1, 2)
        s = s.sum(1)
        return batch_rep, s


class GoadNet(torch.nn.Module):
    def __init__(self, n_input,
                 kernel_size=1, n_hidden=8, n_layers=5, n_output=256,
                 activation='LeakyReLU', bias=False):
        super(GoadNet, self).__init__()

        self.enc = ConvNet(
            n_features=n_input,
            kernel_size=kernel_size,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            bias=bias
        )

        self.head = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv1d(n_hidden, n_output,
                            kernel_size=kernel_size, bias=True)
        )
        return

    def forward(self, x):
        rep = self.enc(x)
        pred = self.head(rep)
        return rep, pred


class GoadLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, margin=1., device='cuda'):
        super(GoadLoss, self).__init__()
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.alpha = alpha
        self.margin = margin
        self.device = device
        return 

    def forward(self, rep, pred, labels):
        loss_ce = self.ce_criterion(pred, labels)

        means = rep.mean(0).unsqueeze(0)
        res = ((rep.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
        pos = torch.diagonal(res, dim1=1, dim2=2)
        offset = torch.diagflat(torch.ones(rep.size(1))).unsqueeze(0).to(self.device) * 1e6
        neg = (res + offset).min(-1)[0]
        loss_tc = torch.clamp(pos + self.margin - neg, min=0).mean()

        loss = self.alpha * loss_tc + loss_ce
        return loss


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        torch.nn.init.eye_(m.weight)
    elif classname.find('Emb') != -1:
        torch.nn.init.normal(m.weight, mean=0, std=0.01)




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

    clf = GOAD(device='cpu', epochs=1)
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_score=scores, y_true=y)

    print(auc)