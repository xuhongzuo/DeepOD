# -*- coding: utf-8 -*-
"""
Deep anomaly detection with deviation networks.
PyTorch's implementation
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch


class DevNet(BaseDeepAD):
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 hidden_dims='100,50', act='ReLU', bias=False,
                 margin=5., l=5000,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DevNet, self).__init__(
            model_name='DevNet', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.margin = margin
        self.l = l

        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias

        return

    def training_prepare(self, X, y):
        # loader: balanced loader, a mini-batch contains a half of normal data and a half of anomalies
        n_anom = np.where(y == 1)[0].shape[0]
        n_norm = self.n_samples - n_anom
        weight_map = {0: 1. / n_norm, 1: 1. / n_anom}

        print(X.shape, y.shape)

        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        print([weight_map[label.item()] for data, label in dataset])
        sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data, label in dataset],
                                        num_samples=self.batch_size, replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        net = MLPnet(
            n_features=self.n_features,
            n_hidden=self.hidden_dims,
            n_output=1,
            activation=self.act,
            bias=self.bias,
        ).to(self.device)

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
        pred = net(batch_x)
        loss = criterion(batch_y, pred)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_z = net(batch_x)
        s = batch_z
        return batch_z, s


class DevLoss(torch.nn.Module):
    def __init__(self, margin=5., l=5000, reduction='mean'):
        super(DevLoss, self).__init__()
        self.margin = margin
        self.loss_l = l
        self.reduction = reduction
        return

    def forward(self, y_true, y_pred):

        ref = torch.randn(self.loss_l)
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(torch.max(self.margin - dev, torch.zeros_like(dev)))
        loss = (1 - y_true) * inlier_loss + y_true * outlier_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
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

    clf = DevNet(device='cpu')
    clf.fit(x, y_semi)

    scores = clf.decision_function(x)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_score=scores, y_true=y)

    print(auc)
