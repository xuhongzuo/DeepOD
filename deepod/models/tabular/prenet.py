# -*- coding: utf-8 -*-
"""
Weakly-supervised anomaly detection by pairwise relation prediction task
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import LinearBlock, MLPnet
import torch
import numpy as np


class PReNet(BaseDeepAD):
    """
    Deep Weakly-supervised Anomaly Detection (KDD‘23)
    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 rep_dim=128, hidden_dims='100,50', act='LeakyReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(PReNet, self).__init__(
            model_name='PReNet', data_type='tabular', epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        return

    def training_prepare(self, X, y):
        train_loader = PReNetLoader(X, y, batch_size=self.batch_size)

        net = DualInputNet(
            self.n_features,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            bias=False,
        ).to(self.device)

        criterion = torch.nn.L1Loss(reduction='mean')

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        # test loader: list of batches
        y = self.train_label
        unlabeled_id = np.where(y == 0)[0]
        known_anom_id = np.where(y == 1)[0]

        if X.shape[0] > 100000:
            a = 10
        elif X.shape[0] > 50000:
            a = 20
        else:
            a = 30

        X = torch.from_numpy(X)
        train_data = torch.from_numpy(self.train_data)

        x2_a_lst = []
        x2_u_lst = []
        for i in range(a):
            a_idx = np.random.choice(known_anom_id, X.shape[0], replace=True)
            u_idx = np.random.choice(unlabeled_id, X.shape[0], replace=True)
            x2_a = train_data[a_idx]
            x2_u = train_data[u_idx]

            x2_a_lst.append(x2_a)
            x2_u_lst.append(x2_u)

        test_loader = []

        n_batches = int(np.ceil(len(X) / self.batch_size))
        for i in range(n_batches):
            left = i * self.batch_size
            right = min((i + 1) * self.batch_size, len(X))
            batch_x1 = X[left: right]
            batch_x_sup1 = [x2[left: right] for x2 in x2_a_lst]
            batch_x_sup2 = [x2[left: right] for x2 in x2_u_lst]
            test_loader.append([batch_x1, batch_x_sup1, batch_x_sup2])
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x1, batch_x2, batch_y = batch_x
        batch_x1 = batch_x1.float().to(self.device)
        batch_x2 = batch_x2.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        pred = net(batch_x1, batch_x2).flatten()

        loss = criterion(pred, batch_y)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x1, batch_x_sup1_lst, batch_x_sup2_lst = batch_x

        batch_x1 = batch_x1.float().to(self.device)
        pred_s = []
        for batch_x2 in batch_x_sup1_lst:
            batch_x2 = batch_x2.float().to(self.device)
            pred = net(batch_x1, batch_x2).flatten()
            pred_s.append(pred)
        for batch_x2 in batch_x_sup2_lst:
            batch_x2 = batch_x2.float().to(self.device)
            pred = net(batch_x1, batch_x2).flatten()
            pred_s.append(pred)

        pred_s = torch.stack(pred_s)
        s = torch.mean(pred_s, dim=0)

        batch_z = batch_x1  # for consistency
        return batch_z, s


class DualInputNet(torch.nn.Module):
    def __init__(self, n_features, hidden_dims='100,50', rep_dim=64,
                 activation='ReLU', bias=False):
        super(DualInputNet, self).__init__()

        network_params = {
            'n_features': n_features,
            'n_hidden': hidden_dims,
            'n_output': rep_dim,
            'activation': activation,
            'bias': bias
        }
        self.enc_net = MLPnet(**network_params)

        self.out_layer = LinearBlock(
            in_channels=2 * rep_dim,
            out_channels=1,
            activation=None,
            bias=False
        )

        return

    def forward(self, x1, x2):
        x1 = self.enc_net(x1)
        x2 = self.enc_net(x2)
        pred = self.out_layer(torch.cat([x1, x2], dim=1))
        return pred


class PReNetLoader:
    def __init__(self, X, y, batch_size, steps_per_epoch=None):
        assert len(X) == len(y)

        self.X = X
        self.y = y
        self.batch_size = min(batch_size, len(X))

        self.unlabeled_id = np.where(y == 0)[0]
        self.known_anom_id = np.where(y == 1)[0]

        self.dim = self.X.shape[1]

        self.counter = 0

        self.steps_per_epoch = steps_per_epoch if steps_per_epoch is not None \
            else int(len(X) / self.batch_size)

        return

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        self.counter += 1
        x1, x2, y = self.batch_generation()
        x1, x2, y = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)

        if self.counter > self.steps_per_epoch:
            raise StopIteration

        return x1, x2, y

    def batch_generation(self):
        batch_x1 = []
        batch_x2 = []
        batch_y = []

        # batch_x1 = np.empty([self.batch_size, self.dim])
        # batch_x2 = np.empty([self.batch_size, self.dim])

        for i in range(self.batch_size):
            if i % 4 == 0 or i % 4 == 1:
                sid = np.random.choice(self.unlabeled_id, 2, replace=False)
                batch_x1.append(self.X[sid[0]])
                batch_x2.append(self.X[sid[1]])
                batch_y.append(0)
            elif i % 4 == 2:
                sid1 = np.random.choice(self.unlabeled_id, 1)
                sid2 = np.random.choice(self.known_anom_id, 1)
                batch_x1.append(self.X[sid1[0]])
                batch_x2.append(self.X[sid2[0]])
                batch_y.append(4)
            else:
                sid = np.random.choice(self.known_anom_id, 2, replace=False)
                batch_x1.append(self.X[sid[0]])
                batch_x2.append(self.X[sid[1]])
                batch_y.append(8)

        batch_x1 = np.array(batch_x1)
        batch_x2 = np.array(batch_x2)
        batch_y = np.array(batch_y)

        return batch_x1, batch_x2, batch_y
