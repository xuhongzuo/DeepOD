# -*- coding: utf-8 -*-
"""
Weakly-supervised anomaly detection by pairwise relation prediction task
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import LinearBlock, get_network
from deepod.models.tabular.prenet import PReNetLoader
import torch
import numpy as np


class PReNetTS(BaseDeepAD):
    """
    Deep Weakly-supervised Anomaly Detection (KDD‘23)
    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 network='Transformer', seq_len=30, stride=1,
                 rep_dim=128, hidden_dims='512', act='GELU', bias=False,
                 n_heads=8, d_model=512, attn='self_attn', pos_encoding='fixed', norm='BatchNorm',
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(PReNetTS, self).__init__(
            model_name='PReNet', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        # parameters for Transformer
        self.n_heads = n_heads
        self.d_model = d_model
        self.pos_encoding = pos_encoding
        self.norm = norm
        self.attn = attn

        return

    def training_prepare(self, X, y):
        train_loader = PReNetLoader(X, y, batch_size=self.batch_size)

        net = DualInputNet(
            self.network,
            self.n_features,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            n_heads=self.n_heads,
            d_model=self.d_model,
            attn=self.attn,
            pos_encoding=self.pos_encoding,
            norm=self.norm,
            seq_len=self.seq_len,
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
    def __init__(self, network_name, n_features, hidden_dims='100,50', rep_dim=64,
                 n_heads=8, d_model=64, attn='self_attn', pos_encoding='fixed', norm='BatchNorm', seq_len=100,
                 activation='ReLU', bias=False):
        super(DualInputNet, self).__init__()

        network_params = {
            'n_features': n_features,
            'n_hidden': hidden_dims,
            'n_output': rep_dim,
            'activation': activation,
            'bias': bias
        }
        if network_name == 'Transformer':
            network_params['n_heads'] = n_heads
            network_params['d_model'] = d_model
            network_params['attn'] = attn
            network_params['pos_encoding'] = pos_encoding
            network_params['norm'] = norm
            network_params['seq_len'] = seq_len
        elif network_name == 'ConvSeq':
            network_params['seq_len'] = self.seq_len

        network_class = get_network(network_name)
        self.enc_net = network_class(**network_params)

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

