# -*- coding: utf-8 -*-
"""
One-class classification
this is partially adapted from https://github.com/lukasruff/Deep-SAD-PyTorch (MIT license)
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import get_network
from deepod.models.tabular.dsad import DSADLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np
from collections import Counter


class DeepSADTS(BaseDeepAD):
    """ Deep Semi-supervised Anomaly Detection (ICLR'20)
    :cite:`ruff2020dsad`
    
    This model extends the semi-supervised anomaly detection framework to time-series datasets, aiming
    to detect anomalies by learning a representation of the data in a lower-dimensional hypersphere.

    Args:

        data_type (str, optional): 
            The type of data, here it's defaulted to 'ts' (time-series).
        
        epochs (int, optional): 
            The number of epochs for training, default is 100.
        
        batch_size (int, optional): 
            The size of the mini-batch for training, default is 64.
        
        lr (float, optional): 
            The learning rate for the optimizer, default is 1e-3.
        
        network (str, optional): 
            The type of network architecture to use, default is 'TCN'.
        
        rep_dim (int, optional): 
            The size of the representation dimension, default is 128.
        
        hidden_dims (Union[list, str, int], optional): 
            The dimensions for hidden layers. It can be a list, a comma-separated string, or a single integer. Default is '100,50'.
                - If list, each item is a layer
                - If str, neural units of hidden layers are split by comma
                - If int, number of neural units of single hidden layer
        
        act (str, optional): 
            The activation function to use. Possible values are 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', default is 'ReLU'.
        
        bias (bool, optional): 
            Whether to include a bias term in the layers, default is False.
        
        n_heads (int, optional): 
            The number of heads in a multi-head attention mechanism, default is 8.
        
        d_model (int, optional): 
            The number of dimensions in the transformer model, default is 512.
        
        attn (str, optional): 
            The type of attention mechanism used, default is 'self_attn'.
        
        pos_encoding (str, optional): 
            The type of positional encoding used in the transformer model, default is 'fixed'.
        
        norm (str, optional): 
            The type of normalization used in the transformer model, default is 'LayerNorm'.
        
        epoch_steps (int, optional): 
            The maximum number of steps per epoch, default is -1, indicating that all batches will be processed.
        
        prt_steps (int, optional): 
            The number of epoch intervals for printing progress, default is 10.
        
        device (str, optional): 
            The device to use for training and inference, default is 'cuda'.
        
        verbose (int, optional): 
            The verbosity mode, default is 2.
        
        random_state (int, optional): 
            The seed for the random number generator, default is 42.

    """

    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 network='TCN', seq_len=100, stride=1,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 n_heads=8, d_model=512, attn='self_attn', pos_encoding='fixed', norm='LayerNorm',
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        """
        Initializes the DeepSADTS model with the provided parameters.
        """
        
        super(DeepSADTS, self).__init__(
            data_type='ts', model_name='DeepSAD', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        # parameters for Transformer
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = attn
        self.pos_encoding = pos_encoding
        self.norm = norm

        self.c = None

        return

    def training_prepare(self, X, y):
        """
        Prepares the model for training by setting up data loaders, initializing the network, and defining the loss criterion.

        Args:
        
            X (np.ndarray): 
                The input feature matrix for training.
                
            y (np.ndarray): 
                The target labels where 1 indicates known anomalies.

        Returns:
        
            train_loader (DataLoader): 
                The data loader for training.
            
            net (nn.Module): 
                The neural network for feature extraction.
            
            criterion (Loss): 
                The loss function used for training.
            
        """
        
        # By following the original paper,
        #   use -1 to denote known anomalies, and 1 to denote known inliers
        known_anom_id = np.where(y == 1)
        y = np.zeros_like(y)
        y[known_anom_id] = -1

        counter = Counter(y)

        if self.verbose >= 2:
            print(f'training data counter: {counter}')

        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(y).long())

        weight_map = {0: 1. / counter[0], -1: 1. / counter[-1]}
        sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data, label in dataset],
                                        num_samples=len(dataset), replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size,
                                  sampler=sampler,
                                  shuffle=True if sampler is None else False)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        if self.network == 'Transformer':
            network_params['n_heads'] = self.n_heads
            network_params['d_model'] = self.d_model
            network_params['pos_encoding'] = self.pos_encoding
            network_params['norm'] = self.norm
            network_params['attn'] = self.attn
            network_params['seq_len'] = self.seq_len
        elif self.network == 'ConvSeq':
            network_params['seq_len'] = self.seq_len

        network_class = get_network(self.network)
        net = network_class(**network_params).to(self.device)

        self.c = self._set_c(net, train_loader)
        criterion = DSADLoss(c=self.c)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        """
        Prepares the model for inference by setting up data loaders.

        Args:
        
            X (np.ndarray): 
                The input feature matrix for inference.

        Returns:
        
            test_loader (DataLoader): 
                The data loader for inference.
            
        """
        
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        """
        Performs a forward training pass.

        Args:
        
            batch_x (tuple): 
                A batch of input data and labels.
            
            net (nn.Module): 
                The neural network model.
            
            criterion (Loss): 
                The loss function.

        Returns:
        
            loss (torch.Tensor): 
                The computed loss for the batch.
            
        """
        
        batch_x, batch_y = batch_x

        # from collections import Counter
        # tmp = batch_y.data.cpu().numpy()
        # print(Counter(tmp))

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.long().to(self.device)

        z = net(batch_x)
        loss = criterion(z, batch_y)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        """
        Performs a forward inference pass.

        Args:
            
            batch_x (torch.Tensor):
                A batch of input data.
            
            net (nn.Module): 
                The neural network model.
            
            criterion (Loss): 
                The loss function used to calculate the anomaly score.

        Returns:
            
            batch_z (torch.Tensor): 
                The encoded batch of data in the feature space.
            
            s (torch.Tensor): 
                The anomaly scores for the batch.
            
        """
        
        batch_x = batch_x.float().to(self.device)
        batch_z = net(batch_x)
        s = criterion(batch_z)
        return batch_z, s

    def _set_c(self, net, dataloader, eps=0.1):
        """
        Initializes the center 'c' for the hypersphere.

        Args:
        
            net (nn.Module): 
                The neural network model.
            
            dataloader (DataLoader): 
                The data loader to compute the center from.
            
            eps (float): 
                A small value to ensure 'c' is away from zero, default is 0.1.

        Returns:
        
            c (torch.Tensor): 
                The initialized center of the hypersphere.
            
        """
        
        net.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)

        # if c is too close to zero, set to +- eps
        # a zero unit can be trivially matched with zero weights
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c
