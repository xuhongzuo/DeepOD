# -*- coding: utf-8 -*-
"""
Representation learning-based unsupervised/weakly-supervised anomaly detection
PyTorch's implementation
this script is partially adapted from the official keras's implementation
https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2FGuansongPang%2Fdeep-outlier-detection&sa=D&sntz=1&usg=AOvVaw2GbqWiY-Y2wkZSjKgU5eQs
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.utils.random import sample_without_replacement
from sklearn.neighbors import KDTree
import numpy as np


class REPEN(BaseDeepAD):
    """
    Learning Representations of Ultrahigh-dimensional Data for Random
    Distance-based Outlier Detection (KDD'18)
    :cite:`pang2018repen`

    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 init_score_ensemble_size=50, init_score_subsample_size=8,
                 rep_dim=128, hidden_dims='100,50', act='LeakyReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(REPEN, self).__init__(
            model_name='REPEN', data_type='tabular', epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.init_score_ensemble_size = init_score_ensemble_size
        self.init_score_subsample_size = init_score_subsample_size
        self.rep_dim = rep_dim
        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias

        return

    def training_prepare(self, X, y):

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        net = MLPnet(**network_params).to(self.device)

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


class REPENLoader:
    """
    Triplets loader
    """
    def __init__(self, X, batch_size, init_scores, steps_per_epoch=None):
        self.X = X
        self.batch_size = min(batch_size, len(X))
        self.init_scores = init_scores
        self.inlier_ids, self.outlier_ids = self.cutoff_unsorted(init_scores)
        self.steps_per_epoch = steps_per_epoch if steps_per_epoch is not None \
            else int(len(X)/self.batch_size)
        self.counter = 0
        return

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        self.counter += 1
        examples, positives, negatives = self.triplet_batch_generation()
        examples, positives, negatives = torch.from_numpy(examples), \
                                         torch.from_numpy(positives), \
                                         torch.from_numpy(negatives)
        if self.counter > self.steps_per_epoch:
            raise StopIteration

        return examples, positives, negatives

    def triplet_batch_generation(self, prior_knowledge=None):
        X = self.X
        outlier_scores = self.init_scores
        inlier_ids = self.inlier_ids
        outlier_ids = self.outlier_ids
        batch_size = self.batch_size

        transforms = np.sum(outlier_scores[inlier_ids]) - outlier_scores[inlier_ids]
        total_weights_p = np.sum(transforms)
        positive_weights = transforms / total_weights_p
        positive_weights = positive_weights.flatten()
        total_weights_n = np.sum(outlier_scores[outlier_ids])
        negative_weights = outlier_scores[outlier_ids] / total_weights_n
        negative_weights = negative_weights.flatten()
        examples_ids = np.zeros([batch_size]).astype('int')
        positives_ids = np.zeros([batch_size]).astype('int')
        negatives_ids = np.zeros([batch_size]).astype('int')
        for i in range(0, batch_size):
            sid = np.random.choice(len(inlier_ids), 1, p=positive_weights)
            examples_ids[i] = inlier_ids[sid]
            sid2 = np.random.choice(len(inlier_ids), 1)

            while sid2 == sid:
                sid2 = np.random.choice(len(inlier_ids), 1)

            positives_ids[i] = inlier_ids[sid2]
            if np.logical_and(prior_knowledge is not None, i % 2 == 0):
                did = np.random.choice(prior_knowledge.shape[0], 1)
                negatives_ids[i] = did
            else:
                sid = np.random.choice(len(outlier_ids), 1, p=negative_weights)
                negatives_ids[i] = outlier_ids[sid]
        examples = X[examples_ids, :]
        positives = X[positives_ids, :]
        negatives = np.zeros([batch_size, X.shape[1]])
        if prior_knowledge is not None:
            negatives[1::2] = X[negatives_ids[1::2], :]
            negatives[::2] = prior_knowledge[negatives_ids[::2], :]
        else:
            negatives = X[negatives_ids, :]
        return examples, positives, negatives

    @staticmethod
    def cutoff_unsorted(values, th=1.7321):
        v_mean = np.mean(values)
        v_std = np.std(values)
        th = v_mean + th * v_std  # 1.7321
        if th >= np.max(values):  # return the top-10 outlier scores
            temp = np.sort(values)
            th = temp[-11]
        outlier_ind = np.where(values > th)[0]
        inlier_ind = np.where(values <= th)[0]
        return inlier_ind, outlier_ind


class REPENLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(REPENLoss, self).__init__()
        self.reduction = reduction
        # self.triplet_loss = torch.nn.TripletMarginLoss(margin=1000., p=2, reduction=reduction)

    def forward(self, example, positive, negative):
        positive_distances = torch.sum(torch.square(example - positive), dim=-1)
        negative_distances = torch.sum(torch.square(example - negative), dim=-1)
        loss = F.relu(1000.-(negative_distances - positive_distances))
        # loss = triplet_loss(example, positive, negative)

        reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss


def repen_init_score_calculator(x_train, ensemble_size=50, subsample_size=8):
    """
    the outlier scoring method, a bagging ensemble of Sp. See the following reference for detail.
    Pang, Guansong, Kai Ming Ting, and David Albrecht.
    "LeSiNN: Detecting anomalies by identifying least similar nearest neighbours."
    In ICDMW15. IEEE.
    """
    # this is for sub-sequences derived from time-series data
    if len(x_train.shape) == 3:
        x_train = x_train[:, -1, :]

    scores = np.zeros([x_train.shape[0], 1])

    ensemble_seeds = np.random.randint(0, np.iinfo(np.int32).max, ensemble_size)

    for i in range(0, ensemble_size):
        rs = np.random.RandomState(ensemble_seeds[i])
        sid = sample_without_replacement(n_population=x_train.shape[0],
                                         n_samples=subsample_size, random_state=rs)
        subsample = x_train[sid]

        kdt = KDTree(subsample, metric='euclidean')
        dists, indices = kdt.query(x_train, k=1)
        scores += dists
    scores = scores / ensemble_size
    return scores
