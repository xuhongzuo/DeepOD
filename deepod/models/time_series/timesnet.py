import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import math
import time
from deepod.utils.utility import get_sub_seqs
from deepod.core.base_model import BaseDeepAD


class TimesNet(BaseDeepAD):
    """
    TIMESNET: Temporal 2D-Variation Modeling for General Time Series Analysis (ICLR'23)

    This model applies temporal 2D-variation modeling to capture the intrinsic patterns in
    time series data for general analysis, specifically designed for anomaly detection tasks.

    Args:
    
        seq_len (int): 
            Length of the input sequences (default 100).
        
        stride (int): 
            Stride for sliding window on data (default 1).
        
        lr (float): 
            Learning rate for optimizer (default 0.0001).
        
        epochs (int):
            Number of epochs to train the model (default 10).
        
        batch_size (int):
            Size of batches for training (default 32).
        
        epoch_steps (int): 
            Number of steps per epoch (default 20).
        
        prt_steps (int): 
            Interval of epochs for printing training progress (default 1).
        
        device (str): 
            Device to use for training, e.g., 'cuda' (default 'cuda').
        
        pred_len (int):
            Prediction length for the model (default 0).
        
        e_layers (int): 
            Number of encoder layers (default 2).
        
        d_model (int):
            Dimensionality of the model (default 64).
        
        d_ff (int): 
            Dimensionality of the feedforward layer (default 64).
        
        dropout (float): 
            Dropout rate (default 0.1).
        
        top_k (int): 
            Top K frequencies for FFT period finding (default 5).
        
        num_kernels (int): 
            Number of kernels for inception block (default 6).
        
        verbose (int): 
            Verbosity level (default 2).
        
        random_state (int): 
            Seed for random number generation (default 42).
        
    """
    
    def __init__(self, seq_len=100, stride=1, lr=0.0001, epochs=10, batch_size=32,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 pred_len=0, e_layers=2, d_model=64, d_ff=64, dropout=0.1, top_k=5, num_kernels=6,
                 verbose=2, random_state=42):
        """
        Initializes TimesNet with the provided parameters.
        """
        
        super(TimesNet, self).__init__(
            model_name='TimesNet', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.top_k = top_k
        self.num_kernels = num_kernels

    def fit(self, X, y=None):
        """
        Fits the TimesNet model to the training data.

        Args:
        
            X (numpy.ndarray): 
                Training data with shape (samples, features).
            
            y (numpy.ndarray, optional): 
                Training labels. Not used in this method (default None).

        Side effects:
        
            Trains the model and updates the model's weights.
            Sets the decision_scores_ and labels_ attributes based on training data.
            
        """
        
        self.n_features = X.shape[1]

        train_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)

        self.net = TimesNetModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.n_features,
            c_out=self.n_features,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            top_k=self.top_k,
            num_kernels=self.num_kernels
        ).to(self.device)

        dataloader = DataLoader(train_seqs, batch_size=self.batch_size,
                                shuffle=True, pin_memory=True)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.net.train()
        for e in range(self.epochs):
            t1 = time.time()
            loss = self.training(dataloader)

            if self.verbose >= 1 and (e == 0 or (e + 1) % self.prt_steps == 0):
                print(f'epoch{e + 1:3d}, '
                      f'training loss: {loss:.6f}, '
                      f'time: {time.time() - t1:.1f}s')
        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()  # in base model
        return

    def decision_function(self, X, return_rep=False):
        """
        Computes the anomaly scores for each sample in X.

        Args:
        
            X (numpy.ndarray):
                Data to compute anomaly scores for.
            
            return_rep (bool, optional): 
                Flag to determine if representations should be returned (default False).

        Returns:
        
            numpy.ndarray: 
                Anomaly scores for each sample in X.
                
        """
                
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=False)

        self.net.eval()
        loss, _ = self.inference(dataloader)  # (n,d)
        loss_final = np.mean(loss, axis=1)  # (n,)

        padding_list = np.zeros([X.shape[0] - loss.shape[0], loss.shape[1]])
        loss_pad = np.concatenate([padding_list, loss], axis=0)
        loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])

        return loss_final_pad

    def training(self, dataloader):
        """
        Conducts a training pass on the given DataLoader.

        Args:
        
            dataloader (DataLoader):
                DataLoader containing the training batches.

        Returns:
        
            float: 
                Average training loss over all batches in the DataLoader.
            
        """
        
        criterion = nn.MSELoss()
        train_loss = []

        for ii, batch_x in enumerate(dataloader):
            self.optimizer.zero_grad()

            batch_x = batch_x.float().to(self.device)  # (bs, seq_len, dim)
            outputs = self.net(batch_x)  # (bs, seq_len, dim)
            loss = criterion(outputs[:, -1:, :], batch_x[:, -1:, :])
            train_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()

            if self.epoch_steps != -1:
                if ii > self.epoch_steps:
                    break
        self.scheduler.step()

        return np.average(train_loss)

    def inference(self, dataloader):
        """
        Performs inference on the data provided by the DataLoader.

        Args:
        
            dataloader (DataLoader): 
                DataLoader containing the test batches.

        Returns:
        
            tuple: 
                A tuple containing anomaly scores and predictions for the test data.
                
        """  
              
        criterion = nn.MSELoss(reduction='none')
        attens_energy = []
        preds = []

        # with torch.no_gard():
        for batch_x in dataloader:  # test_set
            batch_x = batch_x.float().to(self.device)

            outputs = self.net(batch_x)
            # criterion
            score = criterion(batch_x[:, -1:, :], outputs[:, -1:, :]).squeeze(1)  # (bs, dim)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0)  # anomaly scores
        test_energy = np.array(attens_energy)  # anomaly scores

        return test_energy, preds

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


# proposed model
class TimesNetModel(nn.Module):
    """
    The neural network model used within the TimesNet architecture.

    This class defines the model used for anomaly detection in time series data. It includes
    multiple TimesBlocks which are designed to capture both local and global temporal dependencies.

    Args:
    
        seq_len (int): 
            Length of input sequences. (default 100)
        
        pred_len (int):
            Length of prediction sequences. (default 0)
        
        enc_in (int): 
            Number of input features.
        
        c_out (int): 
            Number of output features.
        
        e_layers (int): 
            Number of encoding layers. (default 2)
        
        d_model (int): 
            Dimensionality of the model. (default 64)
        
        d_ff (int): 
            Dimensionality of the feed-forward layer. (default 64)
        
        dropout (float): 
            Dropout rate. (default 0.1)
        
        top_k (int): 
            Number of top frequencies for FFT period detection. (default 5)
        
        num_kernels (int): 
            Number of kernels for inception blocks.
        
    """
    
    def __init__(self, seq_len, pred_len, enc_in, c_out,
                 e_layers, d_model, d_ff, dropout, top_k, num_kernels):
        """
        Initializes the TimesNetModel with the given parameters.
        """
        
        super(TimesNetModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def anomaly_detection(self, x_enc):
        """
        Defines the forward pass of the TimesNetModel.

        Args:
        
            x_enc (Tensor): 
                The encoded input data.

        Returns:
        
            dec_out: 
                The output predictions of the model.
            
        """
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc):
        """
        Defines the forward pass of the TimesNetModel.

        Args:
        
            x_enc (Tensor): 
                The encoded input data.

        Returns:
        
            dec_out: 
                The output predictions of the model.
            
        """
        
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]


class TimesBlock(nn.Module):
    """
    Represents a single block within the TimesNet architecture.

    It applies a series of convolutions with different kernel sizes to capture
    a range of dependencies in the input sequence. FFT is used for period detection,
    followed by a 2D convolution to capture temporal variations.

    Args:
        seq_len (int): 
            The length of input sequences.
        
        pred_len (int): 
            The length of sequences to predict.
        
        top_k (int): 
            The number of top frequencies to consider in FFT.
        
        d_model (int): 
            The number of expected features in the encoder inputs.
        
        d_ff (int): 
            The dimensionality of the feedforward layer.
        
        num_kernels (int):
            The number of different kernel sizes to use in convolutions.
        
    """
    
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        """
        Initializes a TimesBlock with the provided parameters.
        """
        
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        """
        Defines the forward pass of the TimesBlock.

        Args:
        
            x (Tensor): 
                The input tensor to the TimesBlock.

        Returns:
        
            Tensor: 
                The output tensor after applying convolutions and aggregating periods.
                
        """
        
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class DataEmbedding(nn.Module):
    """
    Data embedding module for TimesNet, responsible for generating combined embeddings.

    The embeddings consist of token embeddings, positional embeddings, and optionally,
    time feature embeddings. A dropout is applied after combining the embeddings.

    Args:
    
        c_in (int): 
            The number of features (channels) in the input data.
        
        d_model (int): 
            The dimensionality of the output embeddings.
        
        embed_type (str, optional): 
            The type of temporal embedding to use (default 'timeF').
        
        freq (str, optional): 
            Frequency of the time features (default 'h' for hourly).
        
        dropout (float, optional): 
            The dropout rate (default 0.1).
        
    """
    
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1):
        """
        Initializes the DataEmbedding module with token, positional, and time feature embeddings.
        """
        
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Applies the embeddings to the input sequence and adds them together.

        Args:
        
            x (Tensor): 
                The input data tensor.
                
            x_mark (Tensor, optional): 
                The temporal marks tensor. If provided, temporal embeddings are added.

        Returns:
        
            Tensor: 
                The combined embeddings with dropout applied.
                
        """
        
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """
    Positional embedding using sine and cosine functions.

    Args:
    
        d_model (int):
            The dimensionality of the model.
        
        max_len (int, optional): 
            The maximum length of the input sequences (default 5000).
        
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initializes positional embeddings.
        """
        
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Retrieves the positional embeddings for the input sequence.

        Args:
        
            x (Tensor): 
                The input data tensor.

        Returns:
        
            Tensor: 
                The positional embeddings corresponding to the input sequence.
                
        """
        
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Token embedding using a 1D convolutional layer.

    Args:
    
        c_in (int): 
            The number of features (channels) in the input data.
        
        d_model (int): 
            The dimensionality of the output embeddings.
        
    """
    
    def __init__(self, c_in, d_model):
        """
        Initializes the token embedding layer.
        """
        
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Applies convolution to input sequence to generate token embeddings.

        Args:
        
            x (Tensor): 
                The input data tensor.

        Returns:
        
            Tensor:
                The token embeddings.
                
        """
        
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Fixed embedding layer that applies a non-learnable sinusoidal embedding to the input.

    This embedding is not learned during the training process but is based on the sine and cosine functions.

    Args:
    
        c_in (int): 
            The number of features (channels) in the input data.
        
        d_model (int): 
            The dimensionality of the output embeddings.
        
    """
    
    def __init__(self, c_in, d_model):
        """
        Initializes the FixedEmbedding layer with sinusoidal weights.
        """
        
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        Applies the fixed sinusoidal embedding to the input.

        Args:
        
            x (Tensor): 
                The input tensor for which to generate positional embeddings.

        Returns:
        
            Tensor: 
                The sinusoidally encoded embeddings.
                
        """
        
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    """
    Time feature embedding layer for encoding time-related features.

    Converts time features into dense embeddings using a linear transformation.
    
    Args:
    
        d_model (int): 
            The dimensionality of the output embeddings.
        
        embed_type (str, optional):
            Type of time feature embedding (default 'timeF').
        
        freq (str, optional): 
            Frequency of the time features (default 'h').
        
    """
    
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        Initializes the TimeFeatureEmbedding layer with a linear layer.
        """
        
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """Applies linear transformation to time features to generate embeddings.

        Args:
        
            x (Tensor): 
                The input tensor containing time features.

        Returns:
        
            Tensor: 
                The time feature embeddings.
                
        """
        
        return self.embed(x)


class Inception_Block_V1(nn.Module):
    """
    Inception block module used within the TimesNet model.

    Consists of multiple convolutional layers with different kernel sizes to capture features at various scales.
    
    Args:
    
        in_channels (int): 
            The number of channels in the input data.
        
        out_channels (int): 
            The number of channels in the output data.
        
        num_kernels (int, optional): 
            The number of different kernel sizes (default 6).
        
        init_weight (bool, optional): 
            Whether to initialize weights (default True).
        
    """
        
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        """
        Initializes the Inception_Block_V1 with multiple convolutional layers.
        """
        
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes weights for the convolutional layers.
        """
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Applies each convolutional layer in the module to the input and aggregates the results.

        Args:
        
            x (Tensor): 
                The input tensor.

        Returns:
        
            Tensor: 
                The combined output of all convolutional layers.
                
        """
        
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    """
    Performs Fast Fourier Transform (FFT) to find the top 'k' dominant frequencies and their periods.

    Args:
    
        x (Tensor): 
            The input data tensor.
        
        k (int, optional): 
            The number of top frequencies to identify (default is 2).

    Returns:
    
        tuple: A tuple containing the identified periods and the corresponding weights from FFT.
        
    """
    
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]
