# -*- coding: utf-8 -*-
"""
Deep isolation forest for anomaly detection
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet
from deepod.utils.utility import get_sub_seqs
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np


class DeepIsolationForest(BaseDeepAD):
    """
    Deep Isolation Forest for Anomaly Detection

    Args:
        epochs (int):
            number of training epochs (Default=100).
        batch_size (int):
            number of samples in a mini-batch (Default=64)
        lr (float):
            it is for consistency, unused in this model
    """
    def __init__(self,
                 epochs=100, batch_size=1000, lr=1e-3,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 n_ensemble=50, n_estimators=6,
                 max_samples=256, n_jobs=1,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DeepIsolationForest, self).__init__(
            model_name='DIF', data_type='tabular',
            epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
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

        self.train_data = X
        self.train_label = y
        self.n_samples, self.n_features = X.shape

        if self.verbose >= 1:
            print('Start Training...')

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }

        if self.verbose >= 2:
            iteration = tqdm(range(self.n_ensemble))
        else:
            iteration = range(self.n_ensemble)

        ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)
        for i in iteration:
            net = MLPnet(**network_params)
            torch.manual_seed(ensemble_seeds[i])
            for m in net.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0., std=1.)

            x_tensor = torch.from_numpy(self.train_data).float()
            x_reduced = net(x_tensor).data.numpy()

            ss = StandardScaler()
            x_reduced = ss.fit_transform(x_reduced)
            x_reduced = np.tanh(x_reduced)

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

        if self.verbose >= 1:
            print('Start Inference...')

        testing_n_samples = X.shape[0]
        self.score_lst = np.zeros([self.n_ensemble, testing_n_samples])

        if self.verbose >= 2:
            iteration = tqdm(range(self.n_ensemble))
        else:
            iteration = range(self.n_ensemble)

        for i in iteration:
            x_reduced = self.net_lst[i](torch.from_numpy(X).float()).data.numpy()
            ss = StandardScaler()
            x_reduced = ss.fit_transform(x_reduced)
            x_reduced = np.tanh(x_reduced)
            scores = cal_score(x_reduced, self.iForest_lst[i])
            self.score_lst[i] = scores

        final_scores = np.average(self.score_lst, axis=0)
        return final_scores

    @staticmethod
    def _deep_transfer(X, net, batch_size, device):
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


def cal_score(xx, clf):
    depths = np.zeros((xx.shape[0], len(clf.estimators_)))
    depth_sum = np.zeros(xx.shape[0])
    deviations = np.zeros((xx.shape[0], len(clf.estimators_)))
    leaf_samples = np.zeros((xx.shape[0], len(clf.estimators_)))

    for ii, estimator_tree in enumerate(clf.estimators_):
        # estimator_population_ind = sample_without_replacement(n_population=xx.shape[0], n_samples=256,
        #                                                       random_state=estimator_tree.random_state)
        # estimator_population = xx[estimator_population_ind]

        tree = estimator_tree.tree_
        n_node = tree.node_count

        if n_node == 1:
            continue

        # get feature and threshold of each node in the iTree
        # in feature_lst, -2 indicates the leaf node
        feature_lst, threshold_lst = tree.feature.copy(), tree.threshold.copy()

        #     feature_lst = np.zeros(n_node, dtype=int)
        #     threshold_lst = np.zeros(n_node)
        #     for j in range(n_node):
        #         feature, threshold = tree.feature[j], tree.threshold[j]
        #         feature_lst[j] = feature
        #         threshold_lst[j] = threshold
        #         # print(j, feature, threshold)
        #         if tree.children_left[j] == -1:
        #             leaf_node_list.append(j)

        # compute depth and score
        leaves_index = estimator_tree.apply(xx)
        node_indicator = estimator_tree.decision_path(xx)

        # The number of training samples in each test sample leaf
        n_node_samples = estimator_tree.tree_.n_node_samples

        # node_indicator is a sparse matrix with shape (n_samples, n_nodes), indicating the path of input data samples
        # each layer would result in a non-zero element in this matrix,
        # and then the row-wise summation is the depth of data sample
        n_samples_leaf = estimator_tree.tree_.n_node_samples[leaves_index]
        d = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
        depths[:, ii] = d
        depth_sum += d

        # decision path of data matrix XX
        node_indicator = np.array(node_indicator.todense())

        # set a matrix with shape [n_sample, n_node], representing the feature value of each sample on each node
        # set the leaf node as -2
        value_mat = np.array([xx[i][feature_lst] for i in range(xx.shape[0])])
        value_mat[:, np.where(feature_lst == -2)[0]] = -2
        th_mat = np.array([threshold_lst for _ in range(xx.shape[0])])

        mat = np.abs(value_mat - th_mat) * node_indicator

        # dev_mat = np.abs(value_mat - th_mat)
        # m = np.mean(dev_mat, axis=0)
        # s = np.std(dev_mat, axis=0)
        # dev_mat_mean = np.array([m for _ in range(xx.shape[0])])
        # dev_mat_std = np.array([s for _ in range(xx.shape[0])])
        # dev_mat_zscore = np.maximum((dev_mat - dev_mat_mean) / (dev_mat_std+1e-6), 0)
        # mat = dev_mat_zscore * node_indicator

        exist = (mat != 0)
        dev = mat.sum(axis=1)/(exist.sum(axis=1)+1e-6)
        deviations[:, ii] = dev

    scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length([clf.max_samples_])))
    deviation = np.mean(deviations, axis=1)
    leaf_sample = (clf.max_samples_ - np.mean(leaf_samples, axis=1)) / clf.max_samples_

    scores = scores * deviation
    # scores = scores * deviation * leaf_sample
    return scores


def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.
    average_path_length[mask_2] = 1.
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)

# def get_depth(x_reduced, clf):
#     n_samples = x_reduced.shape[0]
#
#     depths = np.zeros((n_samples, len(clf.estimators_)))
#     depth_sum = np.zeros(n_samples)
#     for ii, (tree, features) in enumerate(zip(clf.estimators_, clf.estimators_features_)):
#         leaves_index = tree.apply(x_reduced)
#         node_indicator = tree.decision_path(x_reduced)
#         n_samples_leaf = tree.tree_.n_node_samples[leaves_index]
#
#         # node_indicator is a sparse matrix, indicating the path of input data samples
#         # with shape (n_samples, n_nodes)
#         # each layer would result in a non-zero element in this matrix,
#         # and then the row-wise summation is the depth of data sample
#         d = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
#         depths[:, ii] = d
#         depth_sum += d
#
#     scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length([clf.max_samples_])))
#     return depths, scores
