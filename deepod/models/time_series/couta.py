"""
Calibrated One-class classifier for Unsupervised Time series Anomaly detection (COUTA)
@author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import numpy as np
import torch
import time
from torch.utils.data import Dataset
from numpy.random import RandomState
from torch.utils.data import DataLoader
from ray import tune
from ray.air import session, Checkpoint

from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from deepod.core.networks.ts_network_tcn import TcnResidualBlock
from deepod.core.base_model import BaseDeepAD
from deepod.metrics import ts_metrics, point_adjustment


class COUTA(BaseDeepAD):
    """
    Calibrated One-class classifier for Unsupervised Time series
    Anomaly detection (arXiv'22)

    Calibrated One-class classifier for Unsupervised Time Series Anomaly detection (COUTA) is a neural network-based model for anomaly detection in time series.
    It operates unsupervised, meaning it doesn't require labeled data for training.
    The model architecture and training process are designed to learn a representation of normal patterns in order to detect anomalies as deviations from these patterns.

    Args:
    
        seq_len (int, optional): 
            The length of each time series segment that the model will ingest during training and inference. 
            It determines how many time steps back the model looks when considering a point for anomaly detection. 
            Default value is 100, which means the model looks at 100 time steps at a time.

        stride (int, optional): 
            The step size for moving the window of `seq_len` across the time series data. A stride of 1 moves 
            the window one time step at a time, resulting in high overlap between consecutive segments, which 
            can be useful for detecting anomalies that require high temporal resolution. Default is 1.

        epochs (int, optional): 
            The total number of complete passes through the entire training dataset. More epochs can potentially 
            lead to better model performance but also a risk of overfitting. Default is 40.

        batch_size (int, optional): 
            The number of time series segments processed together in one pass of gradient descent during training. 
            Larger batch sizes can lead to faster training but may require more memory. Default is 64.

        lr (float, optional): 
            The learning rate dictates the speed at which the model learns. Specifically, it determines the size 
            of the steps taken during gradient descent optimization. A smaller learning rate can make learning more 
            gradual and precise but may slow down convergence. Default is 1e-4.

        ss_type (str, optional): 
            Specifies the type of subsequence to be used for model training. 'FULL' implies that entire sequences 
            are used without alteration. Other types might involve partial sequences or modified segments to 
            introduce certain types of anomalies during training. Default is 'FULL'.

        hidden_dims (int, optional): 
            Specifies the number of neurons in the hidden layers of the neural network. This parameter is crucial 
            for determining the capacity and complexity of the model. Default is 16.

        rep_dim (int, optional):
            Dimensionality of the representations learned by the neural network. It reflects the size of the 
            feature space in which the time series data is embedded. Default is 16.

        rep_hidden (int, optional): 
            Size of the hidden layers specifically in the representation learning part of the network. 
            This can affect the granularity of the learned representations. Default is 16.

        pretext_hidden (int, optional): 
            Size of the hidden layers for a pretext task, which is a task designed to help the network learn 
            useful representations without requiring labeled data. Default is 16.

        kernel_size (int, optional): 
            Size of the convolutional filters if the model uses convolutional layers. It affects how many 
            time steps are considered together at one time by each filter. Default is 2.

        dropout (float, optional): 
            The dropout rate is a regularization technique where randomly selected neurons are ignored during 
            training. It helps in preventing overfitting. Default is 0.0 (no dropout).

        bias (bool, optional): 
            Indicates whether the neurons in the network layers include a bias term. Bias terms can help 
            the model fit the data better. Default is True.

        alpha (float, optional): 
            A weighting factor that balances different components of the loss function. Tweaking this value 
            can significantly affect the training dynamics and model performance. Default is 0.1.

        neg_batch_ratio (float, optional): 
            The proportion of negative samples in each batch. In the context of anomaly detection, negative 
            samples might refer to examples that are considered normal. Default is 0.2.

        train_val_pc (float, optional): 
            The fraction of the training data set aside for validation purposes. Validation data is used to 
            tune hyperparameters and early stopping to avoid overfitting. Default is 0.25 (25% of the training data).

        epoch_steps (int, optional): 
            Defines the number of mini-batch updates within each epoch. Setting this to -1 uses the full dataset 
            for each epoch, which is the default.

        prt_steps (int, optional): 
            Frequency of epochs at which to print out the training progress and loss metrics to monitor the 
            training process. Default is 1, which means printing after every epoch.

        device (str, optional): 
            The computing device to run the model on. 'cuda' for NVIDIA GPU (if available) or 'cpu' for the computer's 
            central processing unit. Default is 'cuda', which will train the model on GPU if available for faster performance.

        verbose (int, optional):    
            The level of verbosity determines how much information the model outputs during training. 
            Higher verbosity levels result in more detailed messages, which can be useful for debugging or detailed analysis. 
            Default is 2, which typically includes progress bars and loss metrics.

        random_state (int, optional): 
            A seed for the random number generator to ensure reproducibility of the results. Setting this to a fixed 
            number ensures that the random processes within the model are repeatable. Default is 42, a common choice in machine learning.

    """
    
    def __init__(self, seq_len=100, stride=1,
                 epochs=40, batch_size=64, lr=1e-4, ss_type='FULL',
                 hidden_dims=16, rep_dim=16, rep_hidden=16, pretext_hidden=16,
                 kernel_size=2, dropout=0.0, bias=True,
                 alpha=0.1, neg_batch_ratio=0.2, train_val_pc=0.25,
                 epoch_steps=-1, prt_steps=1, device='cuda',
                 verbose=2, random_state=42,
                 ):
        """
        Initialize the COUTA model with the provided parameters
        """
        
        super(COUTA, self).__init__(
            model_name='COUTA', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.ss_type = ss_type

        self.kernel_size = kernel_size
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        self.rep_hidden = rep_hidden
        self.pretext_hidden = pretext_hidden
        self.rep_dim = rep_dim
        self.bias = bias

        self.alpha = alpha
        self.neg_batch_size = int(neg_batch_ratio * self.batch_size)
        self.max_cut_ratio = 0.5

        self.train_val_pc = train_val_pc

        self.net = None
        self.c = None
        self.test_df = None
        self.test_labels = None
        self.n_features = -1

        return

    def fit(self, X, y=None):
        """
        Fit detector. In unsupervised anomaly detection, 'y' is not required and thus ignored, allowing the model to learn the normal patterns of the dataset autonomously. This method initiates the training process for the COUTA model on the given time series dataset.

        Args:
        
            X (numpy.ndarray):
                The input dataset for the model to train on. It is expected to be a 2D numpy array where each row corresponds to a sample in the time series, and each column represents a feature at a given time step. The shape of the array should be (n_samples, n_features), where 'n_samples' is the number of time series segments and 'n_features' is the number of observations at each time step.

            y (numpy.ndarray, optional): 
                Labels for the input data. Although not utilized in the unsupervised version of the COUTA model, it is included in the method signature for compatibility with supervised or semi-supervised extensions of the model. In those cases, 'y' could provide additional context or labeling information that may be used for training purposes. It is a 1D array with a length of 'n_samples', where each entry is the label of the corresponding sample in 'X'.

        Returns:
        
            self: 
                The fitted model instance. By returning 'self', the method allows for a fluent interface, enabling the chaining of method calls. After fitting, the model instance contains the learned parameters, and additional methods can be called on it, such as 'decision_function' to compute anomaly scores, or further training with additional data.
        """

        self.n_features = X.shape[1]
        sequences = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)

        sequences = sequences[RandomState(42).permutation(len(sequences))]
        if self.train_val_pc > 0:
            train_seqs = sequences[: -int(self.train_val_pc * len(sequences))]
            val_seqs = sequences[-int(self.train_val_pc * len(sequences)):]
        else:
            train_seqs = sequences
            val_seqs = None

        self.net = _COUTANet(
            input_dim=self.n_features,
            hidden_dims=self.hidden_dims,
            n_output=self.rep_dim,
            pretext_hidden=self.pretext_hidden,
            rep_hidden=self.rep_hidden,
            out_dim=1,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            bias=self.bias,
            pretext=True,
            dup=True
        )
        self.net.to(self.device)

        self.c = self._set_c(self.net, train_seqs)
        self.net = self.train(self.net, train_seqs, val_seqs)

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return

    def decision_function(self, X, return_rep=False):
        """
        Predict raw anomaly score of X using the fitted detector.
        For consistency, outliers are assigned with larger anomaly scores.

        Args:
        
            X (numpy.ndarray): 
                The input dataset to compute anomaly scores for. numpy array of shape (n_samples, n_features)
                The input samples. Sparse matrices are accepted only
                if they are supported by the base estimator.
                
            return_rep (bool, optional): 
                Flag to return the learned representations. default=False
                                
        Returns:
        
            numpy.ndarray: 
                The computed anomaly scores. numpy array of shape (n_samples,).
        """
        
        test_sub_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        test_dataset = _SubseqData(test_sub_seqs)
        dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        representation_lst = []
        representation_lst2 = []
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                x_output = self.net(x)
                representation_lst.append(x_output[0])
                representation_lst2.append(x_output[1])

        reps = torch.cat(representation_lst)
        dis = torch.sum((reps - self.c) ** 2, dim=1).data.cpu().numpy()

        reps_dup = torch.cat(representation_lst2)
        dis2 = torch.sum((reps_dup - self.c) ** 2, dim=1).data.cpu().numpy()
        dis = dis + dis2

        dis_pad = np.hstack([np.zeros(X.shape[0] - dis.shape[0]), dis])
        return dis_pad

    def train(self, net, train_seqs, val_seqs=None):
        """
        Internal method to train the network.

        Args:
        
            net (nn.Module): 
                The neural network to train.
                
            train_seqs (numpy.ndarray): 
                The training sequences.
                
            val_seqs (numpy.ndarray, optional): 
                The validation sequences. Default is None.

        Returns:
        
            nn.Module: 
                The trained neural network.
        """
        
        val_loader = DataLoader(dataset=_SubseqData(val_seqs),
                                batch_size=self.batch_size,
                                drop_last=False, shuffle=False) if val_seqs is not None else None
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        criterion_oc_umc = _DSVDDUncLoss(c=self.c, reduction='mean')
        criterion_mse = torch.nn.MSELoss(reduction='mean')

        y0 = -1 * torch.ones(self.batch_size).float().to(self.device)

        net.train()
        for i in range(self.epochs):

            copy_times = 1
            while len(train_seqs) * copy_times < self.batch_size:
                copy_times += 1
            train_seqs = np.concatenate([train_seqs for _ in range(copy_times)])

            train_loader = DataLoader(dataset=_SubseqData(train_seqs),
                                      batch_size=self.batch_size,
                                      drop_last=True, pin_memory=True, shuffle=True)

            rng = RandomState(seed=self.random_state+i)
            epoch_seed = rng.randint(0, 1e+6, len(train_loader))
            loss_lst, loss_oc_lst, loss_ssl_lst, = [], [], []
            for ii, x0 in enumerate(train_loader):
                x0 = x0.float().to(self.device)

                x0_output = net(x0)

                rep_x0 = x0_output[0]
                rep_x0_dup = x0_output[1]
                loss_oc = criterion_oc_umc(rep_x0, rep_x0_dup)

                neg_cand_idx = RandomState(epoch_seed[ii]).randint(0, self.batch_size, self.neg_batch_size)
                x1, y1 = self.create_batch_neg(batch_seqs=x0[neg_cand_idx],
                                          max_cut_ratio=self.max_cut_ratio,
                                          seed=epoch_seed[ii],
                                          return_mul_label=False,
                                          ss_type=self.ss_type)
                x1, y1 = x1.to(self.device), y1.to(self.device)
                y = torch.hstack([y0, y1])

                x1_output = net(x1)
                pred_x1 = x1_output[-1]
                pred_x0 = x0_output[-1]

                out = torch.cat([pred_x0, pred_x1]).view(-1)

                loss_ssl = criterion_mse(out, y)

                loss = loss_oc + self.alpha * loss_ssl

                net.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lst.append(loss)
                loss_oc_lst.append(loss_oc)

            epoch_loss = torch.mean(torch.stack(loss_lst)).data.cpu().item()
            epoch_loss_oc = torch.mean(torch.stack(loss_oc_lst)).data.cpu().item()

            # validation phase
            val_loss = np.NAN
            if val_seqs is not None:
                val_loss = []
                with torch.no_grad():
                    for x in val_loader:
                        x = x.float().to(self.device)
                        x_out = net(x)
                        loss = criterion_oc_umc(x_out[0], x_out[1])
                        loss = torch.mean(loss)
                        val_loss.append(loss)
                val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

            if (i+1) % self.prt_steps == 0:
                print(
                    f'|>>> epoch: {i+1:02}  |   loss: {epoch_loss:.6f}, '
                    f'loss_oc: {epoch_loss_oc:.6f}, '
                    f'val_loss: {val_loss:.6f}'
                )

        return net

    def _training_ray(self, config, X_test, y_test):
        """
        Trains the COUTA model with hyperparameters provided in 'config', using Ray Tune for optimization.
        It also validates the performance on a hold-out validation set and can test on a separate test set.

        Args:
        
            config (dict): 
                Configuration dictionary containing hyperparameters for the model.
                
            X_test (numpy.ndarray, optional): 
                Test dataset features for evaluation during training. Default is None.
                
            y_test (numpy.ndarray, optional): 
                Test dataset labels for evaluation during training. Default is None.

        Returns:
        
            None: 
                The function is used to update the model attributes directly.
        """
        
        train_data = self.train_data[:int(0.8 * len(self.train_data))]
        val_data = self.train_data[int(0.8 * len(self.train_data)):]

        train_loader = DataLoader(dataset=_SubseqData(train_data), batch_size=self.batch_size,
                                  drop_last=True, pin_memory=True, shuffle=True)
        val_loader = DataLoader(dataset=_SubseqData(val_data), batch_size=self.batch_size,
                                drop_last=True, pin_memory=True, shuffle=True)

        self.net = self.set_tuned_net(config)
        self.c = self._set_c(self.net, train_data)
        criterion_oc_umc = _DSVDDUncLoss(c=self.c, reduction='mean')
        criterion_mse = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'], eps=1e-6)

        self.net.train()
        for i in range(config['epochs']):
            t1 = time.time()
            rng = RandomState(seed=self.random_state+i)
            epoch_seed = rng.randint(0, 1e+6, len(train_loader))
            loss_lst, loss_oc_lst, loss_ssl_lst, = [], [], []
            for ii, x0 in enumerate(train_loader):
                x0 = x0.float().to(self.device)
                y0 = -1 * torch.ones(self.batch_size).float().to(self.device)

                x0_output = self.net(x0)

                rep_x0 = x0_output[0]
                rep_x0_dup = x0_output[1]
                loss_oc = criterion_oc_umc(rep_x0, rep_x0_dup)

                tmp_rng = RandomState(epoch_seed[ii])
                neg_batch_size = int(config['neg_batch_ratio'] * self.batch_size)
                neg_candidate_idx = tmp_rng.randint(0, self.batch_size, neg_batch_size)

                x1, y1 = self.create_batch_neg(
                    batch_seqs=x0[neg_candidate_idx],
                    max_cut_ratio=self.max_cut_ratio,
                    seed=epoch_seed[ii],
                    return_mul_label=False,
                    ss_type=self.ss_type
                )
                x1, y1 = x1.to(self.device), y1.to(self.device)
                y = torch.hstack([y0, y1])

                x1_output = self.net(x1)
                pred_x1 = x1_output[-1]
                pred_x0 = x0_output[-1]

                out = torch.cat([pred_x0, pred_x1]).view(-1)

                loss_ssl = criterion_mse(out, y)
                loss = loss_oc + config['alpha'] * loss_ssl

                self.net.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lst.append(loss)
                loss_oc_lst.append(loss_oc)

            epoch_loss = torch.mean(torch.stack(loss_lst)).data.cpu().item()
            epoch_loss_oc = torch.mean(torch.stack(loss_oc_lst)).data.cpu().item()

            # validation phase
            val_loss = []
            with torch.no_grad():
                for x in val_loader:
                    x = x.float().to(self.device)
                    x_out = self.net(x)
                    loss = criterion_oc_umc(x_out[0], x_out[1])
                    loss = torch.mean(loss)
                    val_loss.append(loss)
            val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

            test_metric = -1
            if X_test is not None and y_test is not None:
                scores = self.decision_function(X_test)
                adj_eval_metrics = ts_metrics(y_test, point_adjustment(y_test, scores))
                test_metric = adj_eval_metrics[2]  # use adjusted Best-F1

            t = time.time() - t1
            if self.verbose >= 1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(
                    f'epoch: {i+1:3d}, '
                    f'training loss: {epoch_loss:.6f}, '
                    f'training loss_oc: {epoch_loss_oc:.6f}, '
                    f'validation loss: {val_loss:.6f}, '
                    f'test F1: {test_metric:.3f},  '
                    f'time: {t:.1f}s'
                )

            checkpoint_data = {
                "epoch": i+1,
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'c': self.c
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            session.report(
                {"loss": val_loss, "metric": test_metric},
                checkpoint=checkpoint,
            )

    @staticmethod
    def set_tuned_params():
        """
        Defines the hyperparameter space for tuning the model.

        Returns:
        
            dict: 
                A dictionary specifying the hyperparameter search space.
        """
        
        config = {
            'lr': tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
            'epochs': tune.grid_search([20, 50, 100]),
            'rep_dim': tune.choice([16, 64, 128, 512]),
            'hidden_dims': tune.choice(['16', '32,16']),
            'alpha': tune.choice([0.1, 0.2, 0.5, 0.8, 1.0]),
            'neg_batch_ratio': tune.choice([0.2, 0.5]),
        }
        return config

    def set_tuned_net(self, config):
        """
        Initializes the network with the given configuration.

        Args:
        
            config (dict): 
                A dictionary containing the hyperparameters for the model.

        Returns:
        
            _COUTANet: 
                The initialized neural network.
        """
        
        net = _COUTANet(
            input_dim=self.n_features,
            hidden_dims=config['hidden_dims'],
            n_output=config['rep_dim'],
            pretext_hidden=self.pretext_hidden,
            rep_hidden=self.rep_hidden,
            out_dim=1,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            bias=self.bias,
            pretext=True,
            dup=True
        ).to(self.device)
        return net

    def load_ray_checkpoint(self, best_config, best_checkpoint):
        """
        Loads the model and its parameters from the best checkpoint obtained from Ray Tune.

        Args:
        
            best_config (dict): 
                The best hyperparameter configuration from Ray Tune.
                
            best_checkpoint (dict): 
                The best model checkpoint from Ray Tune.

        Returns:
        
            None: 
                The function updates the model's parameters directly.
        """
        
        self.c = best_checkpoint['c']
        self.net = self.set_tuned_net(best_config)
        self.net.load_state_dict(best_checkpoint['net_state_dict'])
        return

    def _set_c(self, net, seqs, eps=0.1):
        """
        Initializes the center 'c' for the hypersphere used in the Deep SVDD loss function.

        Args:
        
            net (_COUTANet): 
                The neural network whose output is used to set the center.
            
            seqs (numpy.ndarray):
                The training sequences used to calculate the center.
            
            eps (float, optional): 
                A small value to ensure the center is not exactly at zero. Default is 0.1.

        Returns:
        
            torch.Tensor: 
                The initialized center 'c'.
        """
        
        dataloader = DataLoader(dataset=_SubseqData(seqs), batch_size=self.batch_size,
                                drop_last=False, pin_memory=True, shuffle=True)
        z_ = []
        net.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                x_output = net(x)
                rep = x_output[0]
                z_.append(rep.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        return

    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        return

    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        return

    def inference_prepare(self, X):
        """define test_loader"""
        return

    @staticmethod
    def create_batch_neg(batch_seqs, max_cut_ratio=0.5, seed=0, return_mul_label=False, ss_type='FULL'):
        """
        Creates a batch of negative samples.

        Args:
        
            batch_seqs (numpy.ndarray): 
                The batch of sequences.
                
            max_cut_ratio (float, optional): 
                The maximum cut ratio. Default is 0.5.
                
            seed (int, optional): 
                The seed for the random number generator. Default is 0.
                
            return_mul_label (bool, optional): 
                Whether to return multiple labels. Default is False.
                
            ss_type (str, optional): 
                The type of the subsequence. Default is 'FULL'.

        Returns:
        
            tuple: 
                The batch of negative samples and their labels.
        """
        
        rng = np.random.RandomState(seed=seed)

        batch_size, l, dim = batch_seqs.shape
        cut_start = l - rng.randint(1, int(max_cut_ratio * l), size=batch_size)
        n_cut_dim = rng.randint(1, dim+1, size=batch_size)
        cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]

        if type(batch_seqs) == np.ndarray:
            batch_neg = batch_seqs.copy()
            neg_labels = np.zeros(batch_size, dtype=int)
        else:
            batch_neg = batch_seqs.clone()
            neg_labels = torch.LongTensor(batch_size)

        if ss_type != 'FULL':
            pool = rng.randint(1e+6, size=int(1e+4))
            if ss_type == 'collective':
                pool = [a % 6 == 0 or a % 6 == 1 for a in pool]
            elif ss_type == 'contextual':
                pool = [a % 6 == 2 or a % 6 == 3 for a in pool]
            elif ss_type == 'point':
                pool = [a % 6 == 4 or a % 6 == 5 for a in pool]
            flags = rng.choice(pool, size=batch_size, replace=False)
        else:
            flags = rng.randint(1e+5, size=batch_size)

        n_types = 6
        for ii in range(batch_size):
            flag = flags[ii]

            # collective anomalies
            if flag % n_types == 0:
                batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 0
                neg_labels[ii] = 1

            elif flag % n_types == 1:
                batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 1
                neg_labels[ii] = 1

            # contextual anomalies
            elif flag % n_types == 2:
                mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
                batch_neg[ii, -1, cut_dim[ii]] = mean + 0.5
                neg_labels[ii] = 2

            elif flag % n_types == 3:
                mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
                batch_neg[ii, -1, cut_dim[ii]] = mean - 0.5
                neg_labels[ii] = 2

            # point anomalies
            elif flag % n_types == 4:
                batch_neg[ii, -1, cut_dim[ii]] = 2
                neg_labels[ii] = 3

            elif flag % n_types == 5:
                batch_neg[ii, -1, cut_dim[ii]] = -2
                neg_labels[ii] = 3

        if return_mul_label:
            return batch_neg, neg_labels
        else:
            neg_labels = torch.ones(batch_size).long()
            return batch_neg, neg_labels


class _COUTANet(torch.nn.Module):
    """
    COUTANet framework.

    Args:
    
        input_dim (int): 
            The dimension of the input.
        
        hidden_dims (int or str, optional): 
            The dimensions of the hidden layers. Default is 32.
        
        rep_hidden (int, optional): 
            The dimension of the hidden representation. Default is 32.
        
        pretext_hidden (int, optional): 
            The dimension of the hidden pretext. Default is 16.
        
        n_output (int, optional): 
            The number of outputs. Default is 10.
        
        kernel_size (int, optional):
            The size of the kernel. Default is 2.
        
        dropout (float, optional):
            The dropout rate. Default is 0.2.
        
        out_dim (int, optional):
            The dimension of the output. Default is 2.
        
        bias (bool, optional): 
            Whether to use bias. Default is True.
        
        dup (bool, optional):
            Whether to duplicate. Default is True.
        
        pretext (bool, optional): 
            Whether to use pretext. Default is True.
    """
    
    def __init__(self, input_dim, hidden_dims=32, rep_hidden=32, pretext_hidden=16,
                 n_output=10, kernel_size=2, dropout=0.2, out_dim=2,
                 bias=True, dup=True, pretext=True):
        """
        Initializes the COUTANet.
        """
        
        super(_COUTANet, self).__init__()

        self.layers = []

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        elif type(hidden_dims) == str:
            hidden_dims = hidden_dims.split(',')
            hidden_dims = [int(a) for a in hidden_dims]

        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            self.layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size,
                                             padding=padding_size, dropout=dropout,
                                             bias=bias)]
        self.network = torch.nn.Sequential(*self.layers)
        self.l1 = torch.nn.Linear(hidden_dims[-1], rep_hidden, bias=bias)
        self.l2 = torch.nn.Linear(rep_hidden, n_output, bias=bias)
        self.act = torch.nn.LeakyReLU()

        self.dup = dup
        self.pretext = pretext

        if dup:
            self.l1_dup = torch.nn.Linear(hidden_dims[-1], rep_hidden, bias=bias)

        if pretext:
            self.pretext_l1 = torch.nn.Linear(hidden_dims[-1], pretext_hidden, bias=bias)
            self.pretext_l2 = torch.nn.Linear(pretext_hidden, out_dim, bias=bias)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): 
                The input tensor.

        Returns:
            torch.Tensor: 
                The output tensor.
        """
        
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = out[:, -1]
        rep = self.l2(self.act(self.l1(out)))

        # pretext head
        if self.pretext:
            score = self.pretext_l2(self.act(self.pretext_l1(out)))

            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup, score
            else:
                return rep, score

        else:
            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup
            else:
                return rep


class _SubseqData(Dataset):
    """
    This class is used to create a dataset object for PyTorch's DataLoader. It takes in sequences, labels, and two sets of sample weights.
    The sequences are the main data points, the labels are the corresponding labels for the sequences, and the sample weights are used to adjust the importance of each data point during training.
    The class provides methods to get the length of the dataset and to get a data point by its index.

    Args:
    
        x (numpy.ndarray): 
            The sub sequences.
        
        y (numpy.ndarray, optional): 
            The labels. Default is None.
        
        w1 (numpy.ndarray, optional): 
            The first set of sample weights. Default is None.
        
        w2 (numpy.ndarray, optional): 
            The second set of sample weights. Default is None.
    """
    
    def __init__(self, x, y=None, w1=None, w2=None):
        """
        Initializes the SubseqData.
        """
        
        self.sub_seqs = x
        self.label = y
        self.sample_weight1 = w1
        self.sample_weight2 = w2

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        
            int: 
                The length of the dataset.
        """
        
        return len(self.sub_seqs)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
        
            idx (int): 
                The index.

        Returns:
        
            tuple: 
                The item at the given index.
        """
        
        if self.label is not None and self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.label[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        if self.label is not None:
            return self.sub_seqs[idx], self.label[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is None:
            return self.sub_seqs[idx], self.sample_weight[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        return self.sub_seqs[idx]


class _DSVDDUncLoss(torch.nn.Module):
    """
    This class defines the Deep Support Vector Data Description (DSVDD) Uncertainty Loss. 
    It is used to calculate the loss between the representation tensors and the center tensor.
    The loss is calculated based on the distance between the representation tensors and the center tensor, 
    and the variance of the distances.
    The reduction method can be either 'mean' or 'sum'.
    
    Args:
    
        c (torch.Tensor): 
            The center tensor.
            
        reduction (str, optional): 
            The reduction method. Default is 'mean'.
    """
    
    def __init__(self, c, reduction='mean'):
        """
        Initializes the DSVDDUncLoss.
        """
        
        super(_DSVDDUncLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, rep2):
        """
        Defines the forward pass of the loss.

        Args:
        
            rep (torch.Tensor): 
                The first representation tensor.
            
            rep2 (torch.Tensor): 
                The second representation tensor.

        Returns:
        
            torch.Tensor: 
                The loss tensor.
        """
        
        dis1 = torch.sum((rep - self.c) ** 2, dim=1)
        dis2 = torch.sum((rep2 - self.c) ** 2, dim=1)
        var = (dis1 - dis2) ** 2

        loss = 0.5*torch.exp(torch.mul(-1, var)) * (dis1+dis2) + 0.5*var

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
