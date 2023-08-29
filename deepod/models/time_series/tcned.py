"""
TCN is adapted from https://github.com/locuslab/TCN
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_tcn import TcnAE


class TcnED(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, epochs=10, batch_size=32, lr=1e-3,
                 rep_dim=32, hidden_dims=32, kernel_size=3, act='ReLU', bias=False, dropout=0.2,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(TcnED, self).__init__(
            model_name='TcnED', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        self.bias = bias

        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = TcnAE(
            n_features=self.n_features,
            n_hidden=self.hidden_dims,
            n_emb=self.rep_dim,
            activation=self.act,
            bias=self.bias,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

        criterion = torch.nn.MSELoss(reduction="mean")

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        ts_batch = batch_x.float().to(self.device)
        output, _ = net(ts_batch)
        loss = criterion(output[:, -1], ts_batch[:, -1])
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        error = torch.nn.L1Loss(reduction='none')(output[:, -1], batch_x[:, -1])
        error = torch.sum(error, dim=1)
        return output, error
