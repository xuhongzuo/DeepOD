# -*- coding: utf-8 -*-
"""
One-class classification
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import get_network
from torch.utils.data import DataLoader
import torch


class DeepSVDDTS(BaseDeepAD):
    """
    Deep One-class Classification for Anomaly Detection (ICML'18)
     :cite:`ruff2018deepsvdd`
     

    Args:

        epochs (int, optional): 
            Number of training epochs. Default is 100.
            
        
        batch_size (int, optional): 
            Number of samples in a mini-batch. Default is 64.
        
        lr (float, optional): 
            Learning rate. Default is 1e-5.
        
        network (str, optional):
            Network structure for different data structures. Default is 'Transformer'.
        
        seq_len (int, optional): 
            Size of window used to create subsequences from the data. Default is 30.
        
        stride (int, optional): 
            Number of time points the window moves between subsequences. Default is 10.
        
        rep_dim (int, optional): 
            Dimensionality of the representation space. Default is 64.
        
        hidden_dims (Union[list, str, int], optional): 
            Dimensions for hidden layers. Default is '512'.
                - If list, each item is a layer
                - If str, neural units of hidden layers are split by comma
                - If int, number of neural units of single hidden layer
        
        act (str, optional): 
            Activation layer name. Choices are ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']. Default is 'GELU'.
        
        bias (bool, optional): 
            Whether to add a bias term in linear layers. Default is False.
        
        n_heads (int, optional): 
            Number of heads in multi-head attention. Default is 8.
        
        d_model (int, optional): 
            Number of dimensions in Transformer model. Default is 512.
        
        attn (str, optional): 
            Type of attention mechanism. Default is 'self_attn'.
        
        pos_encoding (str, optional): 
            Manner of positional encoding. Default is 'fixed'.
        
        norm (str, optional): 
            Manner of normalization in Transformer. Default is 'LayerNorm'.
        
        epoch_steps (int, optional): 
            Maximum steps in an epoch. Default is -1.
        
        prt_steps (int, optional): 
            Number of epoch intervals per printing. Default is 10.
        
        device (str, optional): 
            Torch device. Default is 'cuda'.
        
        verbose (int, optional): 
            Verbosity mode. Default is 2.
        
        random_state (int, optional): 
            Seed used by the random number generator. Default is 42.
    
    """
    
    def __init__(self, epochs=100, batch_size=64, lr=1e-5,
                 network='Transformer', seq_len=30, stride=10,
                 rep_dim=64, hidden_dims='512', act='GELU', bias=False,
                 n_heads=8, d_model=512, attn='self_attn', pos_encoding='fixed', norm='LayerNorm',
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        """
        Initializes the DeepSVDDTS model with the specified parameters.
        """
        
        super(DeepSVDDTS, self).__init__(
            model_name='DeepSVDD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
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
        Prepares the training process by setting up data loaders and initializing the network and loss criterion.

        Args:
        
            X (torch.Tensor): 
                Input tensor of the features.
            
            y (torch.Tensor): 
                Input tensor of the labels.

        Returns:
        
            train_loader (DataLoader):
                DataLoader for the training data.
            
            net (nn.Module): 
                Initialized neural network model.
            
            criterion (DSVDDLoss):
                Loss function for DeepSVDD.
                
        """
        
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

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

        # self.c = torch.randn(net.n_emb).to(self.device)
        self.c = self._set_c(net, train_loader)
        criterion = DSVDDLoss(c=self.c)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        """
        Prepares the model for inference by setting up data loaders.

        Args:
        
            X (torch.Tensor): 
                Input tensor of the features for inference.

        Returns:
        
            test_loader (DataLoader):
                DataLoader for inference.
                
        """
        
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        """
        Performs a forward pass during training.

        Args:
        
            batch_x (torch.Tensor): 
                Batch of input data.
            
            net (nn.Module): 
                The neural network model.
            
            criterion (DSVDDLoss):
                Loss function for DeepSVDD.

        Returns:
            
            loss (torch.Tensor):
                Computed loss for the batch.
            
        """
        
        batch_x = batch_x.float().to(self.device)
        z = net(batch_x)
        loss = criterion(z)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        """
        Performs a forward pass during inference.

        Args:
            
            batch_x (torch.Tensor): 
                Batch of input data.
            
            net (nn.Module): 
                The neural network model.
            
            criterion (DSVDDLoss): 
                Loss function for DeepSVDD to calculate anomaly score.

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
        Initializes the center 'c' for the hypersphere in the representation space.

        Args:
        
            net (nn.Module): 
                The neural network model.
            
            dataloader (DataLoader): 
                DataLoader for the data to compute the center from.
            
            eps (float, optional):
                Small value to ensure 'c' is away from zero. Default is 0.1.

        Returns:
        
            c (torch.Tensor):  
                The initialized center of the hypersphere.
            
        """
        
        net.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


class DSVDDLoss(torch.nn.Module):
    """

    Custom loss function for Deep Support Vector Data Description (Deep SVDD).
    
    This loss function computes the distance between each data point in the representation
    space and the center of the hypersphere and aims to minimize this distance for normal data points.

    Args:
    
        c (torch.Tensor):
            The center of the hypersphere in the representation space.
            
        reduction (str, optional): 
            Specifies the reduction to apply to the output. Choices are 'none', 'mean', 'sum'. Default is 'mean'.
                - If ``'none'``: no reduction will be applied;
                - If ``'mean'``: the sum of the output will be divided by the number of elements in the output;
                - If ``'sum'``: the output will be summed

    """
    
    def __init__(self, c, reduction='mean'):
        """
        Initializes the DSVDDLoss with the hypersphere center and reduction method.
        """
        
        super(DSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, reduction=None):
        """
        Calculates the Deep SVDD loss for a batch of representations.

        Args:
        
            rep (torch.Tensor): 
                The representation of the batch of data.
            
            reduction (str, optional): 
                The reduction method to apply. If None, will use the specified 'reduction' attribute. Default is None.

        Returns:
        
            loss (torch.Tensor): 
                The calculated loss based on the representations and the center 'c'.
                
        """
        
        loss = torch.sum((rep - self.c) ** 2, dim=1)

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
