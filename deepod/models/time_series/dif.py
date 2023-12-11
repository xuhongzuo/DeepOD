# -*- coding: utf-8 -*-
"""
Deep isolation forest for anomaly detection
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from sklearn.utils import check_array
from sklearn.ensemble import IsolationForest
from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_dilated_conv import DilatedConvEncoder
from deepod.utils.utility import get_sub_seqs
from deepod.models.tabular.dif import cal_score
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np


class DeepIsolationForestTS(BaseDeepAD):
    """
    Deep isolation forest for anomaly detection (TKDE'23)


    Implementation of a Deep Isolation Forest model for time-series anomaly detection, as described in TKDE'23.
    This model combines deep learning methods for dimensionality reduction with the traditional Isolation Forest 
    algorithm to detect anomalies in time-series data.

    Args:
    
        epochs (int, optional): 
            Number of training epochs. Default is 100.
        
        batch_size (int, optional):
            Batch size for training. Default is 1000.
        
        lr (float, optional):
            Learning rate for the optimizer. Default is 1e-3.
        
        seq_len (int, optional):
            Length of the input sequences. Default is 100.
        
        stride (int, optional): 
            Stride of the sliding window over the time series. Default is 1.
        
        hidden_dims (str, optional): 
            String representation of the hidden layer dimensions, separated by commas.
        
        bias (bool, optional): 
            If True, adds a bias term to the layers of the neural network. Default is False.
        
        n_ensemble (int, optional): 
            Number of ensemble models to train.
        
        n_estimators (int, optional): 
            Number of base estimators in the Isolation Forest. Default is 6.
        
        max_samples (int, optional): 
            Maximum number of samples to draw to train each base estimator. Default is 256.
        
        n_jobs (int, optional): 
            Number of jobs to run in parallel for Isolation Forest training. Default is 1.
        
        epoch_steps (int, optional): 
            Number of steps per epoch. If -1, all batches will be processed.
        
        prt_steps (int, optional): 
            Interval of epochs at which to print progress updates.
        
        device (str, optional): 
            Device to use for training ('cuda' or 'cpu'). Default is 'cuda'.
        
        verbose (int, optional): 
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        
        random_state (int, optional): 
            Seed for random number generation for reproducibility. Default is 42.
        
    """
    
    def __init__(self,
                 epochs=100, batch_size=1000, lr=1e-3,
                 seq_len=100, stride=1,
                 rep_dim=128, hidden_dims=100, bias=False,
                 n_ensemble=50, n_estimators=6,
                 max_samples=256, n_jobs=1,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        """
        Initializes the Deep Isolation Forest Time-Series model with the specified hyperparameters.
        """
                
        super(DeepIsolationForestTS, self).__init__(
            model_name='DIF', data_type='ts', network='DilatedConv',
            epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.bias = bias

        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs

        self.net_lst = []
        self.iForest_lst = []
        self.x_reduced_lst = []
        self.score_lst = []
        return

    def fit(self, X, y=None):
        """
        Fits the Deep Isolation Forest model on the provided time-series data.

        Args:
        
            X (np.ndarray): 
                The input samples of shape (n_samples, n_features).
                
            y (np.ndarray, optional): 
                Target values of shape (n_samples, ) (ignored in unsupervised training).

        Returns:
        
            self: 
                The fitted estimator.
                
        """
        
        X_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        y_seqs = get_sub_seqs(y, seq_len=self.seq_len, stride=self.stride) if y is not None else None
        self.train_data = X_seqs
        self.train_label = y_seqs
        self.n_samples, self.n_features = X_seqs.shape[0], X_seqs.shape[2]

        if self.verbose >= 1:
            print('Start Training...')

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'bias': self.bias
        }

        if self.verbose >= 2:
            iteration = tqdm(range(self.n_ensemble))
        else:
            iteration = range(self.n_ensemble)

        ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)
        for i in iteration:
            net = DilatedConvEncoder(**network_params)
            net = net.to(self.device)

            torch.manual_seed(ensemble_seeds[i])
            for name, param in net.named_parameters():
                torch.nn.init.normal_(param, mean=0., std=1.)
            x_reduced = self._deep_transfer(self.train_data, net, self.batch_size, self.device)

            self.x_reduced_lst.append(x_reduced)
            self.net_lst.append(net)
            self.iForest_lst.append(IsolationForest(n_estimators=self.n_estimators,
                                                    max_samples=self.max_samples,
                                                    n_jobs=self.n_jobs,
                                                    random_state=ensemble_seeds[i]))
            self.iForest_lst[i].fit(x_reduced)

        if self.verbose >= 1:
            print('Start Inference on the training data...')

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return self

    def decision_function(self, X):
        """
        Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Args:
       
            X (np.ndarray): 
                The input samples of shape (n_samples, n_features). Sparse matrices are accepted only if they are supported by the base estimator.

        Returns:
        
            anomaly_scores (np.ndarray): 
                The anomaly score of the input samples with the shape of (n_samples,).

        """

        if self.verbose >= 1:
            print('Start Inference...')

        testing_n_samples = X.shape[0]
        X = get_sub_seqs(X, seq_len=self.seq_len, stride=1)

        self.score_lst = np.zeros([self.n_ensemble, testing_n_samples])

        if self.verbose >= 2:
            iteration = tqdm(range(self.n_ensemble))
        else:
            iteration = range(self.n_ensemble)

        for i in iteration:
            x_reduced = self._deep_transfer(X, self.net_lst[i], self.batch_size, self.device)
            scores = cal_score(x_reduced, self.iForest_lst[i])

            padding = np.zeros(self.seq_len-1)
            scores = np.hstack((padding, scores))

            self.score_lst[i] = scores

        final_scores = np.average(self.score_lst, axis=0)
        return final_scores

    @staticmethod
    def _deep_transfer(X, net, batch_size, device):
        """
        Transfers the input data through the network to obtain reduced representations.

        Args:
        
            X (np.ndarray): 
                The input samples to be reduced.
                
            net (nn.Module): 
                The neural network model for dimensionality reduction.
            
            batch_size (int): 
                Batch size for processing.
            
            device (str): 
                The device on which to perform computations.

        Returns:
        
            x_reduced (np.ndarray): 
                The reduced representation of the input samples.
            
        """
        
        x_reduced = []
        loader = DataLoader(dataset=X, batch_size=batch_size, drop_last=False, pin_memory=True, shuffle=False)
        for batch_x in loader:
            batch_x = batch_x.float().to(device)
            batch_x_reduced = net(batch_x).data.cpu().numpy()
            x_reduced.extend(batch_x_reduced)
        x_reduced = np.array(x_reduced)
        return x_reduced

    def training_prepare(self, X, y):
        pass

    def training_forward(self, batch_x, net, criterion):
        pass

    def inference_prepare(self, X):
        pass

    def inference_forward(self, batch_x, net, criterion):
        pass
