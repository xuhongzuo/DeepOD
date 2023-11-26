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
    Neural Contextual Anomaly Detection for Time Series. (IJCAI'22)
    
    It extends the BaseDeepAD class to implement anomaly detection specific for time series data.

    Args:
    
        epochs: (int, optional)
            The number of epochs to train the model (default is 100).
            
        batch_size: (int, optional)
            The number of samples per batch to load (default is 64).
            
        lr: (float, optional)
            Learning rate for the optimizer (default is 3e-4).
            
        seq_len: (int, optional)
            Length of the input sequences for the model (default is 100).
            
        stride: (int, optional)
            The stride of the window during training (default is 1).
            
        suspect_win_len: (int, optional)
            The length of the window considered as suspect for anomaly (default is 10).
            
        coe_rate: (float, optional)
            Rate at which contextual outlier exposure is applied (default is 0.5).
            
        mixup_rate: (float, optional)
            Rate at which mixup is applied (default is 2.0).
            
        hidden_dims: (list or str, optional)
            The list or comma-separated string of hidden dimensions for the neural network layers (default is '32,32,32,32').
                - If list, each item is a layer
                - If str, neural units of hidden layers are split by comma
                - If int, number of neural units of single hidden layer
                
        rep_dim: (int, optional)
            The size of the representation layer (default is 128).
            
        act: (str, optional)
            The activation function to use in the neural network (default is 'ReLU'). choice = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']
            
        bias: (bool, optional)
            Whether to use bias in the layers (default is False).
            
        kernel_size: (int, optional)
            The size of the kernel for convolutional layers (default is 5).
            
        dropout: (float, optional)
            The dropout rate (default is 0.0).
            
        epoch_steps: (int, optional)
            The maximum number of steps per epoch (default is -1, which processes all batches).
            
        prt_steps: (int, optional)
            The interval for printing the training progress (default is 10).
            
        device: (str, optional)
            The device to use for training the model ('cuda' or 'cpu') (default is 'cuda').
            
        verbose: (int, optional)
            Verbosity mode (default is 2).
            
        random_state: (int, optional)
            Seed used by the random number generator (default is 42).
                  
    """

    def __init__(self, epochs=100, batch_size=64, lr=3e-4,
                 seq_len=100, stride=1,
                 suspect_win_len=10, coe_rate=0.5, mixup_rate=2.0,
                 hidden_dims='32,32,32,32', rep_dim=128,
                 act='ReLU', bias=False,
                 kernel_size=5, dropout=0.0,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        """
        Initializes NCAD with specified hyperparameters.
        """
        
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
        """
        Prepares the training process by creating data loaders and initializing the network and loss criterion.

        Args:
        
            X (numpy.ndarray): 
                Input data array for training.
            
            y (numpy.ndarray): 
                Target labels for training.

        Returns:
            tuple: 
                A tuple containing the DataLoader for training data, the initialized neural network, and the loss criterion.
            
        """
        
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
        """
        Conducts a forward pass during training, including data augmentation strategies like COE and mixup.

        Args:
        
            batch_x (torch.Tensor): 
                The input batch of data.
            
            net (NCADNet): 
                The neural network for NCAD.
            
            criterion (torch.nn.modules.loss): 
                The loss function used for training.

        Returns:
        
            torch.Tensor: 
                The computed loss for the batch.
            
        """
        
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
        """
        Conducts a forward pass during inference to calculate logits for anomaly scores.

        Args:
        
            batch_x (torch.Tensor):
                The input batch of data.
                
            net (NCADNet):
                The neural network for NCAD.
                
            criterion (torch.nn.modules.loss): 
                The loss function used for inference.

        Returns:
        
            tuple:
                A tuple containing the input batch and the logits representing anomaly scores.
                
        """
        
        ts = batch_x.float().to(self.device)
        x0 = ts
        x_context = x0[:, :-self.suspect_win_len]
        logits_anomaly = net(x0, x_context)
        logits_anomaly = logits_anomaly.squeeze()
        return batch_x, logits_anomaly

    def inference_prepare(self, X):
        """
        Prepares the inference process by creating a DataLoader for the test data.

        Args:
        
            X (numpy.ndarray): 
                Input data array for inference.

        Returns:
        
            DataLoader: 
                The DataLoader containing the test data.
                
        """
        
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    @staticmethod
    def coe_batch(x: torch.Tensor, y: torch.Tensor, coe_rate: float, suspect_window_length: int,
                  random_start_end: bool = True):
        """
        Generates a batch of data with contextual outlier exposure (COE) augmentations.

        Args:
        
            x (torch.Tensor): 
                Input batch of data with dimensions (batch, ts channels, time).
            
            y (torch.Tensor): 
                Target labels for the batch.
            
            coe_rate (float): 
                The proportion of the batch to augment with COE.
            
            suspect_window_length (int): 
                The length of the window considered as suspect for anomaly.
            
            random_start_end (bool, optional): 
                Whether to permute a random subset within the suspect segment. Defaults to True.

        Returns:
        
            tuple: 
                A tuple containing the augmented data and corresponding labels.
                
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
        Generates a batch of data with mixup augmentations.

        Args:
        
            x (torch.Tensor): 
                Input batch of data with dimensions (batch, ts channels, time).
                
            y (torch.Tensor): 
                Target labels for the batch.
            
            mixup_rate (float): 
                The proportion of the batch to augment with mixup.

        Returns:
        
            tuple: 
                A tuple containing the mixup-augmented data and corresponding labels.
                
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
    """
    Neural network module used within NCAD for time series anomaly detection.

    This module is based on a temporal convolutional network architecture.

    Args:
    
        n_features (int): 
            Number of features in the input data.
        
        n_hidden (int): 
            Number of hidden units. Default is 32.
        
        n_output (int): 
            Size of the output layer. Default is 128.
        
        kernel_size (int): 
            Kernel size for the convolutional layers. Default is 2.
        
        bias (bool): 
            Whether to use bias in the layers. Default is True.
        
        eps (float): 
            Small epsilon value for numerical stability. Default is 1e-10.
        
        dropout (float): 
            Dropout rate for the network. Default is 0.2.
        
        activation (str): 
            Activation function to use. Default is 'ReLU'.
        
    """
    
    def __init__(self, n_features, n_hidden=32, n_output=128,
                 kernel_size=2, bias=True,
                 eps=1e-10, dropout=0.2, activation='ReLU',
                 ):
        super(NCADNet, self).__init__()
        """
        Initializes the NCADNet with specified parameters.
        """
        
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
        """
        Performs a forward pass of the NCADNet.

        Args:
        
            x (Tensor): 
                The input tensor containing the whole time series data.
            
            x_c (Tensor): 
                The context input tensor for comparison.

        Returns:
        
            Tensor: 
                Logits representing the probability of differences between embeddings of `x` and `x_c`.
            
        """
        
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
    """
    Module that calculates the cosine distance between two tensors.
    Returns the cosine distance between :math:`x_1` and :math:`x_2`, computed along dim.

    Args:
    
        dim (int): 
            The dimension along which to compute the cosine distance. Default is 1.
            
        keepdim (bool): 
            Whether to keep the dimension for the output. Default is True.
            
    """
    
    def __init__( self, dim=1, keepdim=True):
        """
        Initializes the CosineDistance module with specified parameters.
        """
        
        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10

    def forward(self, x1, x2):
        """
        Calculates the cosine distance between two input tensors.

        Args:
        
            x1 (Tensor): 
                The first input tensor.
                
            x2 (Tensor): 
                The second input tensor to compare against `x1`.

        Returns:
        
            Tensor: 
                The cosine distance between the two input tensors.
            
        """
        
        # Cosine of angle between x1 and x2
        cos_sim = F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)
        dist = -torch.log((1 + cos_sim) / 2)

        if self.keepdim:
            dist = dist.unsqueeze(dim=self.dim)
        return dist
