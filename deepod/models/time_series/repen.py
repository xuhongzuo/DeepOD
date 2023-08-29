# -*- coding: utf-8 -*-
"""
Representation learning-based unsupervised/weakly-supervised anomaly detection
PyTorch's implementation
this script is partially adapted from the official keras's implementation
https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2FGuansongPang%2Fdeep-outlier-detection&sa=D&sntz=1&usg=AOvVaw2GbqWiY-Y2wkZSjKgU5eQs
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import get_network
from deepod.models.tabular.repen import REPENLoader, REPENLoss, repen_init_score_calculator
from torch.utils.data import DataLoader
import torch


class REPENTS(BaseDeepAD):
    """
    Pang et al.: Learning Representations of Ultrahigh-dimensional Data for Random
    Distance-based Outlier Detection (KDD'18)
    See :cite:`pang2018repen` for details

    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 network='Transformer', seq_len=30, stride=1,
                 init_score_ensemble_size=50, init_score_subsample_size=8,
                 rep_dim=128, hidden_dims='512', act='GELU', bias=False,
                 n_heads=8, d_model=512, attn='self_attn', pos_encoding='fixed', norm='LayerNorm',
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(REPENTS, self).__init__(
            model_name='REPEN', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.init_score_ensemble_size = init_score_ensemble_size
        self.init_score_subsample_size = init_score_subsample_size
        self.rep_dim = rep_dim
        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias

        # parameters for Transformer
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = attn
        self.pos_encoding = pos_encoding
        self.norm = norm

        return

    def training_prepare(self, X, y):

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': 1,
            'activation': self.act,
            'bias': self.bias
        }
        if self.network == 'Transformer':
            network_params['n_heads'] = self.n_heads
            network_params['d_model'] = self.d_model
            network_params['pos_encoding'] = self.pos_encoding
            network_params['norm'] = self.norm
            network_params['attn'] = self.attn
            network_params['seq_len'] = self.seq_len
        elif self.network == 'ConvSeq':
            network_params['seq_len'] = self.seq_len

        network_class = get_network(self.network)
        net = network_class(**network_params).to(self.device)

        init_scores = repen_init_score_calculator(x_train=X,
                                                  ensemble_size=self.init_score_ensemble_size,
                                                  subsample_size=self.init_score_subsample_size)
        train_loader = REPENLoader(X, batch_size=self.batch_size, init_scores=init_scores)
        criterion = REPENLoss()

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        exp, pos, neg = batch_x
        exp, pos, neg = exp.float().to(self.device), \
                        pos.float().to(self.device), \
                        neg.float().to(self.device)
        exp, pos, neg = net(exp), net(pos), net(neg)
        loss = criterion(exp, pos, neg)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_z = net(batch_x)
        s = torch.zeros(batch_z.shape[0])  # for consistency
        return batch_z, s

    def decision_function_update(self, z, scores):
        scores = repen_init_score_calculator(z,
                                             ensemble_size=self.init_score_ensemble_size,
                                             subsample_size=self.init_score_subsample_size).flatten()
        return z, scores

