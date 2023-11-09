"""
TCN is adapted from https://github.com/locuslab/TCN
"""
import numpy as np

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_tcn import TcnAE
from deepod.metrics import ts_metrics, point_adjustment

import time
import torch
from torch.utils.data import DataLoader
from ray import tune
from ray.air import session, Checkpoint


class TcnED(BaseDeepAD):
    """
    An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series (TNNLS'21)
    """
    def __init__(self, seq_len=100, stride=1, epochs=10, batch_size=32, lr=1e-4,
                 rep_dim=32, hidden_dims=32, kernel_size=3, act='ReLU', bias=True, dropout=0.2,
                 epoch_steps=-1, prt_steps=1, device='cuda',
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

    def training_prepare(self, X, y=None):
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
        self.criterion = torch.nn.MSELoss(reduction="none")
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

    def _training_ray(self, config, X_test, y_test):
        train_data = self.train_data[:int(0.8 * len(self.train_data))]
        val_data = self.train_data[int(0.8 * len(self.train_data)):]

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss(reduction="mean")
        self.net = self.set_tuned_net(config)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'], eps=1e-6)

        self.net.train()
        for i in range(config['epochs']):
            t1 = time.time()
            total_loss = 0
            cnt = 0
            for batch_x in train_loader:
                loss = self.training_forward(batch_x, self.net, criterion)
                self.net.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cnt += 1

                # terminate this epoch when reaching assigned maximum steps per epoch
                if cnt > self.epoch_steps != -1:
                    break

            # validation phase
            val_loss = []
            with torch.no_grad():
                for batch_x in val_loader:
                    loss = self.training_forward(batch_x, self.net, criterion)
                    val_loss.append(loss)
            val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

            test_metric = -1
            if X_test is not None and y_test is not None:
                scores = self.decision_function(X_test)
                adj_eval_metrics = ts_metrics(y_test, point_adjustment(y_test, scores))
                test_metric = adj_eval_metrics[2]  # use adjusted Best-F1

            t = time.time() - t1
            if self.verbose >= 1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch{i+1:3d}, '
                      f'training loss: {total_loss/cnt:.6f}, '
                      f'validation loss: {val_loss:.6f}, '
                      f'test F1: {test_metric:.3f},  '
                      f'time: {t:.1f}s')

            checkpoint_data = {
                "epoch": i,
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            session.report(
                {"loss": val_loss, "metric": test_metric},
                checkpoint=checkpoint,
            )

    def load_ray_checkpoint(self, best_config, best_checkpoint):
        self.net = self.set_tuned_net(best_config)
        self.net.load_state_dict(best_checkpoint['net_state_dict'])
        return

    def set_tuned_net(self, config):
        net = TcnAE(
            n_features=self.n_features,
            n_hidden=config['hidden_dims'],
            n_emb=config['rep_dim'],
            activation=self.act,
            bias=self.bias,
            kernel_size=config['kernel_size'],
            dropout=self.dropout
        ).to(self.device)
        return net

    @staticmethod
    def set_tuned_params():
        config = {
            'lr': tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
            'epochs': tune.grid_search([20, 50, 100]),
            'rep_dim': tune.choice([16, 64, 128, 512]),
            'hidden_dims': tune.choice(['100,100', '100']),
            'kernel_size': tune.choice([2, 3, 5])
        }
        return config
