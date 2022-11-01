# -*- coding: utf-8 -*-
"""
Base class for deep Anomaly detection models
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import numpy as np
import torch
import random
import time
from abc import ABCMeta, abstractmethod


class BaseDeepAD(metaclass=ABCMeta):
    def __init__(self, model_name, epochs=100, batch_size=64, lr=1e-3,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=1, random_state=42):
        self.model_name = model_name

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.device = device

        self.epoch_steps = epoch_steps
        self.prt_steps = prt_steps
        self.verbose = verbose

        self.n_features = -1
        self.n_samples = -1
        self.criterion = None
        self.net = None

        self.train_loader = None
        self.test_loader = None

        self.epoch_time = None

        self.random_state=random_state
        self.set_seed(random_state)
        return

    def fit(self, X, y=None):
        """
        Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples, )
            Not used in unsupervised methods, present for API consistency by convention.
            used in (semi-/weakly-) supervised methods

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self.n_samples, self.n_features = X.shape

        self.train_loader, self.net, self.criterion = self.training_prepare(X, y=y)
        self.training()

        return self

    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        self.test_loader = self.inference_prepare(X)
        scores = self.inference()

        return scores

    def training(self):
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.lr,
                                     weight_decay=1e-5)

        self.net.train()
        for i in range(self.epochs):
            t1 = time.time()
            total_loss = 0
            cnt = 0
            for batch_x in self.train_loader:
                loss = self.training_forward(batch_x, self.net, self.criterion)
                self.net.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cnt += 1

                # terminate this epoch when reaching assigned maximum steps per epoch
                if cnt > self.epoch_steps != -1:
                    break

            t = time.time() - t1
            if self.verbose >=1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch{i+1}, '
                      f'training loss: {total_loss/cnt:.6f}, '
                      f'time: {t:.1f}s')

            if i == 0:
                self.epoch_time = t

            self.epoch_update()

        return

    def inference(self):
        self.net.eval()
        with torch.no_grad():
            z_lst = []
            score_lst = []
            for batch_x in self.test_loader:
                batch_z, s = self.inference_forward(batch_x, self.net, self.criterion)

                z_lst.append(batch_z)
                score_lst.append(s)

        z = torch.cat(z_lst).data.cpu().numpy()
        scores = torch.cat(score_lst).data.cpu().numpy()

        return scores

    @abstractmethod
    def training_forward(self, batch_x, net, criterion):
        pass

    @abstractmethod
    def inference_forward(self, batch_x, net, criterion):
        pass

    @abstractmethod
    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        pass

    @abstractmethod
    def inference_prepare(self, X):
        pass

    def epoch_update(self):
        """for any updating operation after each training epoch"""
        return

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
