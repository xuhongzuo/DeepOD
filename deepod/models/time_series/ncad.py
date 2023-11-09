# -*- coding: utf-8 -*-
"""
Neural Contextual Anomaly Detection for Time Series (NCAD)
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_tcn import TCNnet
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn.functional as F


class NCAD(BaseDeepAD):
    """
    Neural Contextual Anomaly Detection for Time Series (IJCAI'22)

    Parameters
    ----------
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

    def __init__(self, epochs=100, batch_size=64, lr=3e-4,
                 seq_len=100, stride=1,
                 suspect_win_len=10, coe_rate=0.5, mixup_rate=2.0,
                 hidden_dims='32,32,32,32', rep_dim=128,
                 act='ReLU', bias=False,
                 kernel_size=5, dropout=0.0,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(NCAD, self).__init__(
            model_name='NCAD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.suspect_win_len = suspect_win_len

        self.coe_rate = coe_rate
        self.mixup_rate = mixup_rate

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias
        self.dropout = dropout

        self.kernel_size = kernel_size

        return

    def training_prepare(self, X, y):
        y_train = np.zeros(len(X))
        train_dataset = TensorDataset(torch.from_numpy(X).float(),
                                      torch.from_numpy(y_train).long())

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  drop_last=True, pin_memory=True, shuffle=True)

        net = NCADNet(
            n_features=self.n_features,
            n_hidden=self.hidden_dims,
            n_output=self.rep_dim,
            kernel_size=self.kernel_size,
            bias=True,
            eps=1e-10,
            dropout=0.2,
            activation=self.act,
        ).to(self.device)

        criterion = torch.nn.BCELoss()

        return train_loader, net, criterion

    def training_forward(self, batch_x, net, criterion):
        x0, y0 = batch_x

        if self.coe_rate > 0:
            x_oe, y_oe = self.coe_batch(
                x=x0.transpose(2, 1),
                y=y0,
                coe_rate=self.coe_rate,
                suspect_window_length=self.suspect_win_len,
                random_start_end=True,
            )
            # Add COE to training batch
            x0 = torch.cat((x0, x_oe.transpose(2, 1)), dim=0)
            y0 = torch.cat((y0, y_oe), dim=0)

        if self.mixup_rate > 0.0:
            x_mixup, y_mixup = self.mixup_batch(
                x=x0.transpose(2, 1),
                y=y0,
                mixup_rate=self.mixup_rate,
            )
            # Add Mixup to training batch
            x0 = torch.cat((x0, x_mixup.transpose(2, 1)), dim=0)
            y0 = torch.cat((y0, y_mixup), dim=0)

        x0 = x0.float().to(self.device)
        y0 = y0.float().to(self.device)

        x_context = x0[:, :-self.suspect_win_len]
        logits_anomaly = net(x0, x_context)
        probs_anomaly = torch.sigmoid(logits_anomaly.squeeze())

        # Calculate Loss
        loss = criterion(probs_anomaly, y0)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        ts = batch_x.float().to(self.device)

        # ts = ts.transpose(2, 1)
        # stride = self.suspect_win_len
        # unfold_layer = torch.nn.Unfold(
        #     kernel_size=(self.n_features, self.win_len),
        #     stride=stride
        # )
        # ts_windows = unfold_layer(ts.unsqueeze(1))
        #
        # num_windows = int(1 + (self.seq_len - self.win_len) / stride)
        # assert ts_windows.shape == (
        #     batch_x.shape[0],
        #     self.n_features * self.win_len,
        #     num_windows,
        # )
        # ts_windows = ts_windows.transpose(1, 2)
        # ts_windows = ts_windows.reshape(
        #     batch_x.shape[0], num_windows,
        #     self.n_features, self.win_len
        # )
        # x0 = ts_windows.flatten(start_dim=0, end_dim=1)
        # x0 = x0.transpose(2, 1)

        x0 = ts

        x_context = x0[:, :-self.suspect_win_len]
        logits_anomaly = net(x0, x_context)
        logits_anomaly = logits_anomaly.squeeze()
        return batch_x, logits_anomaly

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    @staticmethod
    def coe_batch(x: torch.Tensor, y: torch.Tensor, coe_rate: float, suspect_window_length: int,
                  random_start_end: bool = True):
        """Contextual Outlier Exposure.

        Args:
            x : Tensor of shape (batch, ts channels, time)
            y : Tensor of shape (batch, )
            coe_rate : Number of generated anomalies as proportion of the batch size.
            random_start_end : If True, a random subset within the suspect segment is permuted between time series;
                if False, the whole suspect segment is randomly permuted.
        """

        if coe_rate == 0:
            raise ValueError(f"coe_rate must be > 0.")
        batch_size = x.shape[0]
        ts_channels = x.shape[1]
        oe_size = int(batch_size * coe_rate)

        # Select indices
        idx_1 = torch.arange(oe_size)
        idx_2 = torch.arange(oe_size)
        while torch.any(idx_1 == idx_2):
            idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()
            idx_2 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()

        if ts_channels > 3:
            numb_dim_to_swap = np.random.randint(low=3, high=ts_channels, size=(oe_size))
            # print(numb_dim_to_swap)
        else:
            numb_dim_to_swap = np.ones(oe_size) * ts_channels

        x_oe = x[idx_1].clone()  # .detach()
        oe_time_start_end = np.random.randint(
            low=x.shape[-1] - suspect_window_length, high=x.shape[-1] + 1, size=(oe_size, 2)
        )
        oe_time_start_end.sort(axis=1)
        # for start, end in oe_time_start_end:
        for i in range(len(idx_2)):
            # obtain the dimensons to swap
            numb_dim_to_swap_here = int(numb_dim_to_swap[i])
            dims_to_swap_here = np.random.choice(
                range(ts_channels), size=numb_dim_to_swap_here, replace=False
            )

            # obtain start and end of swap
            start, end = oe_time_start_end[i]

            # swap
            x_oe[i, dims_to_swap_here, start:end] = x[idx_2[i], dims_to_swap_here, start:end]

        # Label as positive anomalies
        y_oe = torch.ones(oe_size).type_as(y)

        return x_oe, y_oe

    @staticmethod
    def mixup_batch(x: torch.Tensor, y: torch.Tensor, mixup_rate: float):
        """
        Args:
            x : Tensor of shape (batch, ts channels, time)
            y : Tensor of shape (batch, )
            mixup_rate : Number of generated anomalies as proportion of the batch size.
        """

        if mixup_rate == 0:
            raise ValueError(f"mixup_rate must be > 0.")
        batch_size = x.shape[0]
        mixup_size = int(batch_size * mixup_rate)  #

        # Select indices
        idx_1 = torch.arange(mixup_size)
        idx_2 = torch.arange(mixup_size)
        while torch.any(idx_1 == idx_2):
            idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()
            idx_2 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()

        # sample mixing weights:
        beta_param = float(0.05)
        beta_distr = torch.distributions.beta.Beta(
            torch.tensor([beta_param]), torch.tensor([beta_param])
        )
        weights = torch.from_numpy(np.random.beta(beta_param, beta_param, (mixup_size,))).type_as(x)
        oppose_weights = 1.0 - weights

        # Create contamination
        x_mix_1 = x[idx_1].clone()
        x_mix_2 = x[idx_1].clone()
        x_mixup = (
            x_mix_1 * weights[:, None, None] + x_mix_2 * oppose_weights[:, None, None]
        )  # .detach()

        # Label as positive anomalies
        y_mixup = y[idx_1].clone() * weights + y[idx_2].clone() * oppose_weights

        return x_mixup, y_mixup


class NCADNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=32, n_output=128,
                 kernel_size=2, bias=True,
                 eps=1e-10, dropout=0.2, activation='ReLU',
                 ):
        super(NCADNet, self).__init__()

        self.network = TCNnet(
            n_features=n_features,
            n_hidden=n_hidden,
            n_output=n_output,
            kernel_size=kernel_size,
            bias=bias,
            dropout=dropout,
            activation=activation
        )

        self.distance_metric = CosineDistance()
        self.eps = eps

    def forward(self, x, x_c):
        x_whole_embedding = self.network(x)
        x_context_embedding = self.network(x_c)

        dists = self.distance_metric(x_whole_embedding, x_context_embedding)

        # Probability of the two embeddings being equal: exp(-dist)
        log_prob_equal = -dists

        # Computation of log_prob_different
        prob_different = torch.clamp(1 - torch.exp(log_prob_equal), self.eps, 1)
        log_prob_different = torch.log(prob_different)

        logits_different = log_prob_different - log_prob_equal

        return logits_different


class CosineDistance(torch.nn.Module):
    r"""Returns the cosine distance between :math:`x_1` and :math:`x_2`, computed along dim."""

    def __init__( self, dim=1, keepdim=True):
        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10

    def forward(self, x1, x2):
        # Cosine of angle between x1 and x2
        cos_sim = F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)
        dist = -torch.log((1 + cos_sim) / 2)

        if self.keepdim:
            dist = dist.unsqueeze(dim=self.dim)
        return dist

