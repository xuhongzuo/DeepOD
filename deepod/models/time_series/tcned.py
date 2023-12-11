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
    
    Temporal Convolutional Network for Anomaly Detection in Multivariate Time Series.

    Args:
    
        seq_len (int): 
            The length of the input sequences for the network. Default is 100.
        
        stride (int): 
            The stride of the convolutional operation. Default is 1.
        
        epochs (int): 
            The number of training epochs. Default is 10.
        
        batch_size (int): 
            The batch size used in training. Default is 32.
        
        lr (float): 
            The learning rate for the optimizer. Default is 1e-4.
        
        rep_dim (int): 
            The dimensionality of the latent representation (embedding) layer. Default is 32.
        
        hidden_dims (int): 
            The number of hidden units in each layer. Default is 32.
        
        kernel_size (int):
            The size of the kernel in the convolutional layers. Default is 3.
        
        act (str): 
            The activation function used in the network. Default is 'ReLU'.
        
        bias (bool): 
            Whether to use bias in the convolutional layers. Default is True.
        
        dropout (float): 
            The dropout rate used in the network. Default is 0.2.
        
        epoch_steps (int): 
            The number of steps per epoch. Default is -1, indicating use of the full dataset.
        
        prt_steps (int): 
            The interval of epochs at which to print training progress. Default is 1.
        
        device (str): 
            The device on which to train the model, 'cuda' or 'cpu'. Default is 'cuda'.
        
        verbose (int): 
            The verbosity level. Default is 2.
        
        random_state (int): 
            The seed for random number generation. Default is 42.
        
    """
    
    def __init__(self, seq_len=100, stride=1, epochs=10, batch_size=32, lr=1e-4,
                 rep_dim=32, hidden_dims=32, kernel_size=3, act='ReLU', bias=True, dropout=0.2,
                 epoch_steps=-1, prt_steps=1, device='cuda',
                 verbose=2, random_state=42):
        """
        Initializes the TcnED model with specified parameters.
        """
        
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
        """
        Sets up the model for training including the data loader, network, and loss criterion.

        Args:
        
            X (numpy.ndarray):
                The input features for training.
            
            y (numpy.ndarray, optional): 
                The target values for training. Defaults to None.

        Returns:
        
            tuple: 
                A tuple containing the training data loader, network, and loss criterion.
                
        """

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
        """
        Prepares the model for inference, including setting up the data loader.

        Args:
        
            X (numpy.ndarray): 
                The input features for inference.

        Returns:
        
            DataLoader: 
                A data loader containing the test dataset.
                
        """
        
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion = torch.nn.MSELoss(reduction="none")
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        """
        Conducts a forward training pass with a batch of data.

        Args:
        
            batch_x (Tensor): 
                The batch of training data.
            
            net (torch.nn.Module): 
                The network model.
            
            criterion (callable): 
                The loss criterion.

        Returns:
        
            Tensor: 
                The loss for the training batch.
            
        """       
         
        ts_batch = batch_x.float().to(self.device)
        output, _ = net(ts_batch)
        loss = criterion(output[:, -1], ts_batch[:, -1])
        return loss

    def inference_forward(self, batch_x, net, criterion):
        """
        Conducts a forward inference pass with a batch of data.

        Args:
        
            batch_x (Tensor): 
                The batch of inference data.
            
            net (torch.nn.Module):
                The network model.
            
            criterion (callable):
                The loss criterion used to compute the error.

        Returns:
        
            tuple: 
                A tuple containing the output and the error for the inference batch.
            
        """
        
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        error = torch.nn.L1Loss(reduction='none')(output[:, -1], batch_x[:, -1])
        error = torch.sum(error, dim=1)
        return output, error

    def _training_ray(self, config, X_test, y_test):
        """
        Internal method for training using Ray Tune for hyperparameter search.

        Args:
        
            config (dict): 
                The configuration dictionary for Ray Tune.
            
            X_test (numpy.ndarray): 
                The test dataset features.
            
            y_test (numpy.ndarray):
                The test dataset labels.

        """
        
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
        """
        Loads the best model checkpoint from Ray Tune.

        Args:
        
            best_config (dict): 
                The best configuration found by Ray Tune.
            
            best_checkpoint (dict): 
                The checkpoint data to load.
            
        """
        
        self.net = self.set_tuned_net(best_config)
        self.net.load_state_dict(best_checkpoint['net_state_dict'])
        return

    def set_tuned_net(self, config):
        """
        Sets up the network model with tuned hyperparameters.

        Args:
        
            config (dict): 
                The configuration dictionary containing the hyperparameters.

        Returns:
        
            TcnAE: 
                The initialized network model with the specified hyperparameters.
                
        """
        
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
        """
        Defines the grid of hyperparameters for tuning.

        Returns:
        
            dict: 
                A configuration dictionary for Ray Tune.
                
        """
        
        config = {
            'lr': tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
            'epochs': tune.grid_search([20, 50, 100]),
            'rep_dim': tune.choice([16, 64, 128, 512]),
            'hidden_dims': tune.choice(['100,100', '100']),
            'kernel_size': tune.choice([2, 3, 5])
        }
        return config
