# -*- coding: utf-8 -*-
"""
Deep anomaly detection with deviation networks.
PyTorch's implementation
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import get_network
from deepod.models.tabular.devnet import DevLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np


class DevNetTS(BaseDeepAD):
    """
    Deviation Networks for Weakly-supervised Anomaly Detection (KDD'19)
    :cite:`pang2019deep`

    Deviation Networks (DevNet) designed for weakly-supervised anomaly detection.
    This implementation is based on the architecture presented in the KDD'19 paper:
    "Deviation Networks for Weakly-supervised Anomaly Detection" by Pang et al.

    Args:
            
        hidden_dims (Union[list, str, int], optional): 
            The dimensions for the hidden layers. Can be a list of integers, a string of comma-separated integers, or a single integer.
            - If list, each item is a layer
            - If str, neural units of hidden layers are split by comma
            - If int, number of neural units of single hidden layer
            - Defaults to '100,50'.
            
        act (str, optional): 
            Activation function to use. Choices include 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'. Default is 'ReLU'.
            
        bias (bool, optional): 
            Whether to include a bias term in the linear layers. Default is False.
            
        n_heads (int, optional): 
            Number of heads in multi-head attention. Only used when network is 'transformer'. Default is 8.
            
        pos_encoding (str, optional): 
            The type of positional encoding to use. Only relevant when network is 'transformer'. Choices are 'fixed' or 'learnable'. Default is 'fixed'.
            
        norm (str, optional): 
            Normalization method in the Transformer. Only relevant when network is 'transformer'. Choices are 'LayerNorm' or 'BatchNorm'. Default is 'LayerNorm'.
            
        epochs (int, optional):
            Number of training epochs. Default is 100.
            
        batch_size (int, optional): 
            Batch size for training. Default is 64.
            
        lr (float, optional): 
            Learning rate for the optimizer. Default is 1e-3.
            
        network (str, optional): 
            Type of network architecture to use. Default is 'Transformer'.
            
        seq_len (int, optional): 
            Length of input sequences for models that require it. Default is 100.
            
        stride (int, optional): 
            Stride of the convolutional layers. Default is 1.
            
        rep_dim (int, optional): 
            The representation dimension. Unused in this model but kept for consistency. Default is 128.
            
        d_model (int, optional): 
            The number of expected features in the transformer model. Only used when network is 'transformer'. Default is 512.
            
        attn (str, optional): 
            Type of attention to use. Only used when network is 'transformer'. Default is 'self_attn'.
            
        margin (float, optional): 
            Margin for the deviation loss function. Default is 5.
            
        l (int, optional): 
            The size of the sample for the Gaussian distribution in the deviation loss function. Default is 5000.
            
        epoch_steps (int, optional): 
            Maximum number of steps per epoch. If -1, all batches will be processed. Default is -1.
            
        prt_steps (int, optional): 
            Number of epoch intervals for printing during training. Default is 10.
        
        device (str, optional): 
            The device to use for training ('cuda' or 'cpu'). Default is 'cuda'.
        
        verbose (int, optional): 
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 2.
        
        random_state (int, optional): 
            Seed for the random number generator for reproducibility. Default is 42.
            
    """
    
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 network='Transformer', seq_len=100, stride=1,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 n_heads=8, d_model=512, attn='self_attn', pos_encoding='fixed', norm='LayerNorm',
                 margin=5., l=5000,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        """
        Initialize the DevNetTS.
        """
        super(DevNetTS, self).__init__(
            data_type='ts', model_name='DevNet', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.margin = margin
        self.l = l

        self.hidden_dims = hidden_dims
        self.act = act
        self.bias = bias

        # parameters for Transformer
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = attn
        self.pos_encoding = pos_encoding
        self.norm = norm

        return

    def training_prepare(self, X, y):
        """
        Prepares the data and model for training by creating a balanced data loader, 
        initializing the network, and setting up the loss criterion.

        Args:
        
            X (np.ndarray): 
                The input features for training.
            
            y (np.ndarray): 
                The target labels for training, where 1 indicates an anomaly.

        Returns:
        
            train_loader (DataLoader): 
                A DataLoader with balanced mini-batches for training.
                
            net (nn.Module): 
                The initialized neural network model.
                
            criterion (Loss): 
                The loss function used during training.
                
        """
        
        # loader: balanced loader, a mini-batch contains a half of normal data and a half of anomalies
        n_anom = np.where(y == 1)[0].shape[0]
        n_norm = self.n_samples - n_anom
        weight_map = {0: 1. / n_norm, 1: 1. / n_anom}

        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data, label in dataset],
                                        num_samples=len(dataset), replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': 1,
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

        criterion = DevLoss(margin=self.margin, l=self.l)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        """
        Prepares the data for inference.

        Args:
        
            X (Tensor): 
                The input features for inference.

        Returns:
        
            test_loader (DataLoader): 
                A DataLoader for inference.
                
        """
        
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        """
        Performs a forward pass during training.

        Args:
        
            batch_x (tuple): 
                A batch of input features and target labels.
                
            net (nn.Module): 
                The neural network model.
            
            criterion (Loss): 
                The loss function used during training.

        Returns:
        
            loss (Tensor): 
                The computed loss for the batch.
        """
        
        batch_x, batch_y = batch_x
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.to(self.device)
        pred = net(batch_x)
        loss = criterion(batch_y, pred)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        """
        Performs a forward pass during inference.

        Args:
        
            batch_x (Tensor): 
                A batch of input features.
            
            net (nn.Module): 
                The neural network model.
            
            criterion (Loss): 
                The loss function used during training. Not used.

        Returns:
        
            batch_z (Tensor): 
                The batch of input features (unmodified).
            
            s (Tensor): 
                The computed scores for the batch.
            
        """
        
        batch_x = batch_x.float().to(self.device)
        s = net(batch_x)
        s = s.view(-1)
        batch_z = batch_x
        return batch_z, s
