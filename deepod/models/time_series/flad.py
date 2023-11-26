"""
FLAD is adapted from https://github.com/nuhdv/FLAD
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from deepod.core.base_model import BaseDeepAD
import torch.nn as nn
from torch.nn.utils import weight_norm
import math


class FLAD(BaseDeepAD):
    """
    "Fusion Learning Based Unsupervised Anomaly Detection for Multi-Dimensional Time Series" and published in Journal of Computer Reasearch and Development.

    Args:
    
        hidden_dims (int, optional): 
            Dimension of hidden layers. Optional, defaults to 32.
            
        rep_dim (int, optional):
            Dimension of representation layer. Optional, defaults to 32.
        
        kernel_size (int, optional):
            Size of the kernel in convolutional layers. Optional, defaults to 3.
        
        dropout (float, optional): 
            Dropout rate for regularization. Optional, defaults to 0.2.
        
        act (str, optional): 
            Activation function to use. Optional, defaults to 'ReLU'.
        
        num_channels (list[int], optional): 
            Number of channels for TCN layers. Optional, defaults to [64, 128, 256].
        
        bias (bool, optional): 
            Whether to use bias in convolutional layers. Optional, defaults to False.
        
        epoch_steps (int, optional): 
            Number of steps per epoch. Optional, defaults to -1.
        
        prt_steps (int, optional): 
            Interval of steps to print training progress. Optional, defaults to 10.
        
        device (str, optional): 
            Device to use for training. Optional, defaults to 'cuda'.
        
        verbose (int, optional):
            Verbosity mode. Optional, defaults to 2.
        
        random_state (int, optional):
            Seed for random number generators. Optional, defaults to 42.
        
        seq_len (int, optional): 
            Length of the input sequences for the model.
        
        stride (int, optional): 
            Stride size for convolutional operations.
        
        epochs (int, optional): 
            Number of epochs to train the model.
        
        batch_size (int, optional):
            Size of batches for training.
        
        lr (float, optional): 
            Learning rate for the optimizer.
        
    """
    
    def __init__(self, seq_len=100, stride=1, epochs=10, batch_size=32, lr=1e-3,
                 rep_dim=32, hidden_dims=32, kernel_size=3, act='ReLU', num_channels=[64, 128, 256], bias=False, dropout=0.2,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        """
        Initialize the FLAD.
        """
        
        super(FLAD, self).__init__(
            model_name='FLAD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
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
        self.num_channels = num_channels

        return

    def get_model(self, inputs):
        """
        Constructs the FLAD model consisting of TCN and Transformer encoders and decoders,
        along with the cross-stitch units for feature fusion.

        Args:
        
            inputs (int): 
                The number of input features.

        Returns:
        
            torch.nn.Module: 
                The constructed FLAD model.
                
        """
        
        backbone_dict, decoder_dict = {}, {}
        cross_stitch_kwargs = {'alpha': 0.8, 'beta': 0.2, 'stages': ['layer1', 'layer2', 'layer3'],
                               'channels': {'layer1': 64, 'layer2': 128, 'layer3': 256},
                               'num_channels': self.num_channels}

        TCN = SingleTaskModel(TCN_encoder(num_inputs=inputs, num_channels=self.num_channels, kernel_size=self.kernel_size, dropout=0.2),
                              TCN_decoder(num_inputs=inputs, num_channels=list(reversed(self.num_channels)), kernel_size=self.kernel_size, dropout=0.2), 'reconstruct')
        TCN = torch.nn.DataParallel(TCN)
        backbone_dict['reconstruct'] = TCN.module.encoder
        decoder_dict['reconstruct'] = TCN.module.decoder

        Trans = SingleTaskModel(Trans_encoder(num_inputs=inputs, feature_size=self.num_channels[-1], num_channels=self.num_channels, num_layers=1, dropout=0.1),
                                Trans_decoder(num_inputs=inputs, feature_size=self.num_channels[-1], num_layers=1, dropout=0.1), 'predict')
        Trans = torch.nn.DataParallel(Trans)
        backbone_dict['predict'] = Trans.module.encoder
        decoder_dict['predict'] = Trans.module.decoder
        model = CrossStitchNetwork(['reconstruct', 'predict'], torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict), **cross_stitch_kwargs)
        return model

    def training_prepare(self, X, y):
        """
        Prepares the training process by setting up the data loader, model, and loss criterion.

        Args:
        
            X (torch.Tensor): 
                The input features for training.
                
            y (torch.Tensor): 
                The target labels for training.

        Returns:
        
            tuple: 
                A tuple containing the DataLoader, the model, and the criterion.
            
        """
        
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = self.get_model(self.n_features)

        criterion = torch.nn.MSELoss(reduction="mean")

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        """
        Prepares the model for inference by setting up the data loader and setting the loss criterion
        to not reduce to a single value.

        Args:
        
            X (torch.Tensor):
                The input features for inference.

        Returns:
        
            DataLoader:
                The DataLoader configured for inference.
                
        """
        
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        """
        Performs a forward pass during training and computes the loss.

        Args:
        
            batch_x (torch.Tensor): 
                The batch of input data.
            
            net (torch.nn.Module):
                The network to which the forward pass should be applied.
            
            criterion (torch.nn.modules.loss._Loss): 
                The loss function to compute the loss.

        Returns:
        
            torch.Tensor: The computed loss.
            
        """
        
        ts_batch = batch_x.float().to(self.device)
        output, _ = net(ts_batch)
        loss = criterion(output[:, -1], ts_batch[:, -1])
        return loss

    def inference_forward(self, batch_x, net, criterion):
        """
        Performs a forward pass during inference and computes the error for each example in the batch.

        Args:
        
            batch_x (torch.Tensor): 
                The batch of input data.
            
            net (torch.nn.Module): 
                The network to which the forward pass should be applied.
            
            criterion (torch.nn.modules.loss._Loss): 
                The loss function to compute the error.

        Returns:
        
            tuple:
                A tuple containing the network output and error for each example.
            
        """
        
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        error = torch.nn.L1Loss(reduction='none')(output[:, -1], batch_x[:, -1])
        error = torch.sum(error, dim=1)
        return output, error


class SingleTaskModel(nn.Module):
    """
    A single-task baseline model that consists of an encoder and a decoder.

    Args:
    
        encoder (nn.Module): 
            The encoder module.
        
        decoder (nn.Module): 
            The decoder module.
        
        task (str): 
            The name of the task.
        
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}

class ChannelWiseMultiply(nn.Module):
    """
    A module that multiplies each channel of its input with a learnable parameter.

    Args:
    
        num_channels (int): 
            The number of channels in the input data.
            
    """
    
    def __init__(self, num_channels):
        super(ChannelWiseMultiply, self).__init__()
        self.param = nn.Parameter(torch.FloatTensor(num_channels), requires_grad=True)

    def init_value(self, value):
        with torch.no_grad():
            self.param.data.fill_(value)

    def forward(self, x):
        return torch.mul(self.param.view(1, -1, 1), x)


class CrossStitchUnit(nn.Module):
    """
    A unit for learning a combination of features from multiple tasks.

    Args:
    
        tasks (list of str): 
            The list of task names.
        
        num_channels (int): 
            The number of channels in the input data.
        
        alpha (float): 
            Initial value for diagonal elements in the cross-stitch matrix. Default is 0.8.
        
        beta (float): 
            Initial value for off-diagonal elements in the cross-stitch matrix. Default is 0.2.
        
    """
    
    def __init__(self, tasks, num_channels, alpha, beta):
        super(CrossStitchUnit, self).__init__()
        self.cross_stitch_unit = nn.ModuleDict(
            {t: nn.ModuleDict({t: ChannelWiseMultiply(num_channels) for t in tasks}) for t in tasks})

        for t_i in tasks:
            for t_j in tasks:
                if t_i == t_j:
                    self.cross_stitch_unit[t_i][t_j].init_value(alpha)
                else:
                    self.cross_stitch_unit[t_i][t_j].init_value(beta)

    def forward(self, task_features):
        out = {}
        for t_i in task_features.keys():
            prod = torch.stack([self.cross_stitch_unit[t_i][t_j](task_features[t_j]) for t_j in task_features.keys()])
            out[t_i] = torch.sum(prod, dim=0)
        return out

class CrossStitchNetwork(nn.Module):
    """
        Implementation of cross-stitch networks.
        We insert a cross-stitch unit, to combine features from the task-specific backbones after every stage.

        Args:
        
            backbone:
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}

            heads:
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}

            stages:
                list of stages where we instert a cross-stitch unit between the task-specific backbones.
                Note: the backbone modules require a method 'forward_stage' to get feature representations
                at the respective stages.

            channels:
                dict which contains the number of channels in every stage

            alpha, beta:
                floats for initializing cross-stitch units (see paper)

    """
    
    def __init__(self, TASKS, backbone: nn.ModuleDict, heads: nn.ModuleDict,
                 stages: list, channels: dict, alpha: float, beta: float, num_channels: list):
        super(CrossStitchNetwork, self).__init__()

        # Tasks, backbone and heads
        self.tasks = TASKS
        self.backbone = backbone
        self.heads = heads
        self.stages = stages

        # Cross-stitch units
        self.cross_stitch = nn.ModuleDict(
            {stage: CrossStitchUnit(self.tasks, channels[stage], alpha, beta) for stage in stages})

    def forward(self, x, tgt):
        x = x.permute(0, 2, 1)
        x = {task: x for task in self.tasks}  # Feed as input to every single-task network

        # Backbone
        for stage in self.stages:
            x['reconstruct'] = self.backbone['reconstruct'].forward_stage(x['reconstruct'], stage)
            x['predict'], w = self.backbone['predict'].forward_stage(x['predict'], stage)
            # Cross-stitch the task-specific features
            x = self.cross_stitch[stage](x)
        out = {task: self.heads[task](x[task], tgt) for task in self.tasks}

        return out

class Chomp1d(nn.Module):
    """
    A module that chops off the last few entries of a 1D convolution's output, 
    which is a common operation in temporal convolutional networks to maintain causality.

    Args:
    
        chomp_size (int): 
            The number of entries to chop off from the end of the tensor.
            
    """
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class pad1d(nn.Module):
    """
    A padding module that applies padding to the end of a 1D tensor along the temporal dimension.

    Args:
    
        pad_size (int): 
            The number of entries to pad.
            
    """
    
    def __init__(self, pad_size):
        super(pad1d, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        return torch.cat([x, x[:, :, -self.pad_size:]], dim = 2).contiguous()

class TemporalBlockTranspose(nn.Module):
    """
    A temporal block for a TCN that uses transposed convolutions for upsampling.

    Args:
    
        n_inputs (int): 
            The number of input channels.
            
        n_outputs (int): 
            The number of output channels.
        
        kernel_size (int): 
            The kernel size of the convolution.
        
        stride (int): 
            The stride of the convolution.
        
        dilation (int):
            The dilation of the convolution.
        
        padding (int): 
            The padding of the convolution.
        
        dropout (float):
            The dropout rate. Default is 0.2.
        
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
        dropout=0.2):
        super(TemporalBlockTranspose, self).__init__()
        self.conv1 = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.pad1 = pad1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.pad2 = pad1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.dropout1, self.relu1, self.pad1, self.conv1,
            self.dropout2, self.relu2, self.pad2, self.conv2)
        self.downsample = nn.ConvTranspose1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalBlock(nn.Module):
    """
    A temporal block for a TCN that encapsulates two convolutions, each followed by a non-linearity and dropout.

    Args:
    
        n_inputs (int): 
            The number of input channels.
        
        n_outputs (int):
            The number of output channels.
        
        kernel_size (int): 
            The kernel size of the convolution.
        
        stride (int): 
            The stride of the convolution.
        
        dilation (int): 
            The dilation of the convolution.
        
        padding (int): 
            The padding of the convolution.
        
        dropout (float):
            The dropout rate. Default is 0.2.
        
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PositionalEncoding(nn.Module):
    """
    A module that injects some information about the relative or absolute position of the tokens in the sequence.

    Args:
    
        d_model (int): 
            The dimensionality of the input embeddings.
            
        dropout (float): 
            The dropout rate. Default is 0.1.
            
        max_len (int): 
            The maximum length of the input sequences. Default is 5000.
            
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.src_mask = None
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        return self.dropout(x + self.pe[:x.size(0), :])

class TokenEmbedding(nn.Module):
    """
    A module that converts token indices into embeddings, commonly used as the first step in a Transformer model.

    Args:
    
        c_in (int): 
            The number of channels in the input data.
        
        d_model (int): 
            The dimensionality of the output embeddings.
            
    """
    
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x.permute(2, 0, 1)

class TCN_encoder(nn.Module):
    """
    An encoder module for a Temporal Convolutional Network.

    Args:
    
        num_inputs (int): 
            The number of input channels.
        
        num_channels (list of int): 
            The number of output channels for each layer.
        
        kernel_size (int): 
            The kernel size for the convolutional layers. Default is 2.
        
        dropout (float): 
            The dropout rate. Default is 0.2.
        
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_encoder, self).__init__()
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*self.layers)

    def forward_stage(self, x, stage):
        assert (stage in ['layer1', 'layer2', 'layer3', 'layer4'])
        if stage == 'layer1':
            x = self.layers[0](x)
            return x
        elif stage == 'layer2':
            x = self.layers[1](x)
            return x
        elif stage == 'layer3':
            x = self.layers[2](x)
            return x

    def forward(self, x):
        out = x.permute(0, 2, 1)
        return self.network(out)

class TCN_decoder(nn.Module):
    """
    A decoder module for a Temporal Convolutional Network.

    Args:
    
        num_inputs (int): 
            The number of input channels.
            
        num_channels (list of int): 
            The number of output channels for each layer.
        
        kernel_size (int):
            The kernel size for the convolutional layers. Default is 2.
        
        dropout (float): 
            The dropout rate. Default is 0.2.
        
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_decoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # no dilation in decoder
            in_channels = num_channels[i]
            out_channels = num_inputs if i == (num_levels - 1) else num_channels[i + 1]
            dilation_size = 2 ** (num_levels - 1 - i)
            padding_size = (kernel_size - 1) * dilation_size
            layers += [TemporalBlockTranspose(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=padding_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.fcn = nn.Sequential(nn.Linear(num_channels[0], num_inputs), nn.Sigmoid())

    def forward(self, x, tgt):
        out = self.network(x)
        out = out.permute(0, 2, 1)
        return out[:, -1].view(out.shape[0], 1, out.shape[2])

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a transformer encoder model.

    Args:
    
        d_model (int): 
            The number of expected features in the input (required).
            
        nhead (int): 
            The number of heads in the multiheadattention models (required).
        
        dim_feedforward (int): 
            The dimension of the feedforward network model. Default is 16.
        
        dropout (float): 
            The dropout value. Default is 0.1.
        
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weight


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a transformer decoder model.

    Args:
    
        d_model (int): 
            The number of expected features in the input (required).
        
        nhead (int): 
            The number of heads in the multiheadattention models (required).
        
        dim_feedforward (int):
            The dimension of the feedforward network model. Default is 16.
        
        dropout (float):
            The dropout value. Default is 0.1.
        
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Trans_encoder(nn.Module):
    """
    The encoder module for a transformer model.

    Args:
    
        num_inputs (int): 
            The number of input channels.
        
        feature_size (int): 
            The dimensionality of the input embeddings. Default is 512.
        
        num_channels (list of int):
            The number of channels for each layer. Default is [64, 128, 256].
        
        num_layers (int):
            The number of layers in the encoder. Default is 1.
        
        dropout (float): 
            The dropout rate. Default is 0.1.
        
    """
    
    def __init__(self, num_inputs, feature_size=512, num_channels=[64, 128, 256], num_layers=1, dropout=0.1):
        super(Trans_encoder, self).__init__()

        self.src_mask = None
        self.embedding = TokenEmbedding(c_in=num_inputs * 2, d_model=feature_size)
        # self.embed = TokenEmbedding(c_in=num_inputs, d_model=feature_size)
        self.pos_encoder = PositionalEncoding(feature_size, dropout=0.1)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout)
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.layer1 = self._make_layer(inputs=num_inputs, feature_size=num_channels[0], num_layers=num_layers,
                                       dropout=dropout)
        self.layer2 = self._make_layer(inputs=num_channels[0], feature_size=num_channels[1], num_layers=num_layers,
                                       dropout=dropout)
        self.layer3 = self._make_layer(inputs=num_channels[1], feature_size=num_channels[2], num_layers=num_layers,
                                       dropout=dropout)

    def _make_layer(self, inputs, feature_size, num_layers, dropout):
        # layers = []
        embedding = TokenEmbedding(c_in=inputs, d_model=feature_size)
        pos_encoder = PositionalEncoding(feature_size, dropout=0.1)
        encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return nn.Sequential(embedding, pos_encoder, transformer_encoder)

    def forward_stage(self, x, stage):
        assert(stage in ['layer1', 'layer2', 'layer3'])
        layer = getattr(self, stage)
        x ,w = layer(x)
        return x.permute(1, 2, 0), w

    def forward(self, src, c):
        src = self.embedding(torch.cat((src, c), dim=2))
        src = src.permute(1, 0, 2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output.permute(1, 2, 0)

class Trans_decoder(nn.Module):
    """
    The decoder module for a transformer model.

    Args:
    
        num_inputs (int): 
            The number of input channels.
        
        feature_size (int): 
            The dimensionality of the input embeddings. Default is 512.
        
        num_layers (int): 
            The number of layers in the decoder. Default is 1.
        
        dropout (float): 
            The dropout rate. Default is 0.1.
        
    """
    
    def __init__(self, num_inputs, feature_size=512, num_layers=1, dropout=0.1):
        super(Trans_decoder, self).__init__()

        self.embed = TokenEmbedding(c_in=num_inputs, d_model=feature_size)
        decoder_layer = TransformerDecoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, num_inputs)
        self.fcn = nn.Sequential(nn.Linear(feature_size, num_inputs), nn.Sigmoid())

    def forward(self, output, tgt):
        tgt = tgt.permute(0, 2, 1)
        out = self.transformer_decoder(self.embed(tgt), output.permute(2, 0, 1))
        out = self.decoder(out)
        return out.permute(1, 0, 2)[:, -1].view(out.shape[1], 1, out.shape[2])
