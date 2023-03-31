# -*- coding: utf-8 -*-
"""
!! this is in development now. !!
this script is partially adapted from https://github.com/jmjeon94/AnoGAN-pytorch
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch
import time


class AnoGAN(BaseDeepAD):
    """ AnoGAN for anomaly detection
    See : for detail

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
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 z_dim=128,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(AnoGAN, self).__init__(
            model_name='AnoGAN', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        self.z_dim = z_dim

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias
        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        g_net = MLPnet(
            n_features=self.z_dim,
            n_hidden=self.hidden_dims,
            n_output=self.n_features,
            activation=self.act,
            bias=self.bias,
        ).to(self.device)
        d_net = MLPnet(
            n_features=self.n_features,
            n_hidden = self.hidden_dims,
            n_output = 1,
            activation = self.act,
            bias=self.bias
        ).to(self.device)
        net = (g_net, d_net)

        criterion = torch.nn.BCELoss()

        if self.verbose >= 2:
            print(g_net)
            print(d_net)

        return train_loader, net, criterion

    def _training(self):
        optimizer_g = torch.optim.Adam(self.net[0].parameters(),
                                       lr=self.lr,
                                       weight_decay=1e-5)
        optimizer_d = torch.optim.Adam(self.net[1].parameters(),
                                       lr=self.lr,
                                       weight_decay=1e-5)

        for i in range(self.epochs):
            t1 = time.time()
            total_g_loss = 0
            total_d_loss = 0
            cnt = 0
            for batch_x in self.train_loader:
                b_size = batch_x.size(0)

                # update discriminator network
                self.net[1].zero_grad()

                # for real
                batch_x = batch_x.float().to(self.device)
                label = torch.ones(b_size).to(self.device)
                output_real = self.net[1](batch_x).view(-1)
                output_real = torch.sigmoid(output_real)
                err_real = self.criterion(output_real, label)
                err_real.backward()

                # for noise
                fake = self.net[0](torch.randn(b_size, self.z_dim, device=self.device))
                label = torch.zeros(b_size).to(self.device)
                output_fake = self.net[1](fake.detach()).view(-1)
                output_fake = torch.sigmoid(output_fake)
                err_fake = self.criterion(output_fake, label)
                err_fake.backward()

                err_d = err_fake + err_real
                optimizer_d.step()

                # update generative network
                self.net[0].zero_grad()
                label.fill_(1.)
                output = self.net[1](fake).view(-1)
                output = torch.sigmoid(output)
                err_g = self.criterion(output, label)

                err_g.backward()
                optimizer_g.step()

                total_d_loss += err_d.item()
                total_g_loss += err_g.item()
                cnt += 1

                # terminate this epoch when reaching assigned maximum steps per epoch
                if cnt > self.epoch_steps != -1:
                    break

            t = time.time() - t1
            if self.verbose >= 1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch{i+1}, '
                      f'training loss (generative/discriminative): '
                      f'{total_g_loss/cnt:.6f} / {total_d_loss/cnt:.6f}, '
                      f'time: {t:.1f}s')

            if i == 0:
                self.epoch_time = t

            self.epoch_update()

        return

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def _inference(self):
        self.net[1].eval()
        with torch.no_grad():
            z_lst = []
            score_lst = []
            for batch_x in self.test_loader:
                batch_x = batch_x.float().to(self.device)
                s = self.net[1](batch_x)
                s = s.view(-1)

                batch_z = batch_x

                z_lst.append(batch_z)
                score_lst.append(s)

        z = torch.cat(z_lst).data.cpu().numpy()
        scores = torch.cat(score_lst).data.cpu().numpy()

        return z, scores

    def training_forward(self, batch_x, net, criterion):
        # implement in _training
        pass

    def inference_forward(self, batch_x, net, criterion):
        # implement in _inference
        pass
