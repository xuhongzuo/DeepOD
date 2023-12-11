import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder

import numpy as np
from torch.utils.data import DataLoader
import math
from deepod.utils.utility import get_sub_seqs
from deepod.core.base_model import BaseDeepAD


class TranAD(BaseDeepAD):
    """
    TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data (VLDB'22)

    Args:
    
        seq_len (int): 
            The length of the input sequences for the model (default 100).
        
        stride (int): 
            The stride size for the sliding window mechanism (default 1).
        
        lr (float): 
            The learning rate for the optimizer (default 0.001).
        
        epochs (int): 
            The number of epochs to train the model (default 5).
        
        batch_size (int): 
            The size of the batches used during training (default 128).
        
        epoch_steps (int):
            The number of steps per epoch (default 20).
        
        prt_steps (int):
            The number of epochs after which to print progress (default 1).
        
        device (str): 
            The device on which to train the model ('cuda' or 'cpu') (default 'cuda').
        
        verbose (int): 
            The verbosity level of the training process (default 2).
        
        random_state (int): 
            The seed used by the random number generator (default 42).
        
    """
    
    def __init__(self, seq_len=100, stride=1, lr=0.001, epochs=5, batch_size=128,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 verbose=2, random_state=42):
        """
        Initializes the TranAD model with the specified parameters for training.
        """
        
        super(TranAD, self).__init__(
            model_name='TranAD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.w_size = None
        self.n_features = None
        return

    def fit(self, X, y=None):
        """
        Fits the TranAD model to the given multivariate time series data.

        Args:
        
            X (numpy.ndarray): 
                The input time series data.
            
            y (numpy.ndarray, optional): 
                The true labels for the data. This argument is not used.

        """
        
        self.n_features = X.shape[1]

        train_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        self.model = TranADNet(
            feats=self.n_features,
            n_window=self.seq_len
        ).to(self.device)

        dataloader = DataLoader(train_seqs, batch_size=self.batch_size,
                                shuffle=True, pin_memory=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.model.train()
        for e in range(self.epochs):
            loss = self.training(dataloader, epoch=e)
            print(f'Epoch {e+1},\t L1 = {loss}')

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()
        return

    def decision_function(self, X, return_rep=False):
        """
        Computes anomaly scores for the given time series data.

        Args:
        
            X (numpy.ndarray): 
                The input time series data.
                
            return_rep (bool, optional):
                Flag to determine whether to return the latent representations. Defaults to False.

        Returns:
        
            numpy.ndarray: 
                Anomaly scores for each instance in the time series data.
            
        """
        
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=False)

        self.model.eval()
        loss, _ = self.inference(dataloader)  # (n,d)
        loss_final = np.mean(loss, axis=1)  # (n,)

        padding_list = np.zeros([X.shape[0]-loss.shape[0], loss.shape[1]])
        loss_pad = np.concatenate([padding_list, loss], axis=0)
        loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])  # (8640,)

        return loss_final_pad

    def training(self, dataloader, epoch):
        """
        Conducts a single epoch of training over the provided data loader.

        Args:
        
            dataloader (DataLoader): 
                DataLoader containing the training batches.
            
            epoch (int): 
                The current epoch number.

        Returns:
        
            float: 
                The average loss over all batches in this epoch.
            
        """
        
        criterion = nn.MSELoss(reduction='none')

        n = epoch + 1
        l1s, l2s = [], []

        for ii, batch_x in enumerate(dataloader):
            local_bs = batch_x.shape[0]  #(128，30，19)
            window = batch_x.permute(1, 0, 2)  # (30, 128, 19)
            elem = window[-1, :, :].view(1, local_bs, self.n_features)  #(1, 128, 19)

            window = window.float().to(self.device)
            elem = elem.float().to(self.device)

            z = self.model(window, elem)
            l1 = (1/n) * criterion(z[0], elem) + (1-1/n) * criterion(z[1], elem)  #(1, 128, 19)

            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            self.optimizer.zero_grad()

            loss.backward(retain_graph=True)
            self.optimizer.step()

            if self.epoch_steps != -1:
                if ii > self.epoch_steps:
                    break

        self.scheduler.step()

        return np.mean(l1s)

    def inference(self, dataloader):
        """
        Conducts the inference phase, computing the anomaly scores for the provided data loader.

        Args:
        
            dataloader (DataLoader): 
                DataLoader containing the data for inference.

        Returns:
        
            Tuple[numpy.ndarray, list]: 
                An array of loss values and a list of predictions (currently unused).
                
        """
        
        criterion = nn.MSELoss(reduction='none')

        l1s = []
        preds = []
        for d in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, self.n_features)
            window = window.float().to(self.device)
            elem = elem.float().to(self.device)
            z = self.model(window, elem)
            if isinstance(z, tuple):
                z = z[1]
            l1 = criterion(z, elem)[0]
            l1 = l1.data.cpu()
            l1s.append(l1)

        l1s = torch.cat(l1s)
        l1s = l1s.numpy()
        return l1s, preds

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


# Proposed Model + Self Conditioning + Adversarial + MAML
class TranADNet(nn.Module):
    """
    The neural network architecture for TranAD, composed of a transformer encoder and two transformer decoders.
    
    Args:
    
        feats (int): 
            Number of features in the input data.
        
        n_window (int): 
            Number of time steps in the input data. Default is 10.
        
    """
    
    def __init__(self, feats, n_window=10):
        """
        Initializes the TranADNet with specified architecture parameters.
        """
        
        super(TranADNet, self).__init__()

        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        """
        Encodes the input sequence and concatenates it with the context to create a memory for the transformer.

        Args:
        
            src (Tensor): 
                The source sequence tensor.
            
            c (Tensor): 
                The context tensor.
            
            tgt (Tensor): 
                The target tensor.

        Returns:
        
            Tuple[Tensor, Tensor]: 
                A tuple containing the target tensor and the memory tensor.
            
        """
        
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        """
        Forward pass of the TranADNet model.

        Args:
        
            src (Tensor): 
                The source sequence tensor.
                
            tgt (Tensor): 
                The target tensor.

        Returns:
        
            Tuple[Tensor, Tensor]:
                The output from both decoder phases.
                
        """
        
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding which is added to the input embeddings at the bottom of the encoder stack.
    
    Args:
    
        d_model (int): 
            Dimension of the embeddings.
        
        dropout (float): 
            Dropout value. Default is 0.1.
        
        max_len (int): 
            Maximum length of the input sequences. Default is 5000.
        
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the PositionalEncoding module.
        """
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        """
        Adds positional encoding to the input tensor.

        Args:
        
            x (Tensor): 
                The input tensor to add positional encodings to.
            
            pos (int, optional): 
                The starting position index for positional encoding.

        Returns:
        
            Tensor: The input tensor with added positional encodings.
            
        """
        
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder stack.
    
    Args:
    
        d_model (int): 
            Dimension of the model.
        
        nhead (int): 
            Number of attention heads.
        
        dim_feedforward (int):
            Dimension of the feedforward network. Default is 16.
        
        dropout (float): 
            Dropout rate. Default is 0.

    """
    
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        """
        Initializes a layer of the transformer encoder stack.
        """
        
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for the TransformerEncoderLayer.

        Args:
        
            src (Tensor): 
                The source sequence tensor.
            
            src_mask (Tensor, optional): 
                The mask for the source sequence.
            
            src_key_padding_mask (Tensor, optional): 
                The padding mask for the source keys.

        Returns:
        
            Tensor: 
                The output tensor from the encoder layer.
            
        """
        
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of the transformer decoder stack.
    
    Args:
    
        d_model (int): 
            Dimension of the model.
        
        nhead (int): 
            Number of attention heads.
        
        dim_feedforward (int): 
            Dimension of the feedforward network. Default is 16.
        
        dropout (float): 
            Dropout rate. Default is 0.
        
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        """
        Initializes a layer of the transformer decoder stack.
        """
        
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for the TransformerDecoderLayer.

        Args:
        
            tgt (Tensor): 
                The target sequence tensor.
            
            memory (Tensor): 
                The memory tensor from the encoder.
            
            tgt_mask (Tensor, optional): 
                The mask for the target sequence.
            
            memory_mask (Tensor, optional): 
                The mask for the memory sequence.
            
            tgt_key_padding_mask (Tensor, optional): 
                The padding mask for the target keys.
            
            memory_key_padding_mask (Tensor, optional): 
                The padding mask for the memory keys.

        Returns:
        
            Tensor: 
                The output tensor from the decoder layer.
            
        """
        
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
