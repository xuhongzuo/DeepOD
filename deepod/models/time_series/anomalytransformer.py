import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import math
import time
from deepod.utils.utility import get_sub_seqs
from deepod.core.base_model import BaseDeepAD


def my_kl_loss(p, q):
    """
    Custom Kullback-Leibler divergence loss calculation.

    Args:
    
        p (torch.Tensor): 
            The first probability distribution tensor.
            
        q (torch.Tensor): 
            The second probability distribution tensor to compare against.

    Returns:
    
        torch.Tensor: 
            The mean KL divergence computed over all dimensions except the last one.
    """
    
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class AnomalyTransformer(BaseDeepAD):
    """
    Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy
    (ICLR'22)
    
    Implements the Anomaly Transformer model for time series anomaly detection based on
    the paper "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy".

    Inherits from BaseDeepAD which contains base functionality for anomaly detection models.
 
    Args:
        seq_len (int, optional): 
            This parameter determines the length of the input sequences for the transformer. Default is 100.
            
        stride (int, optional): 
            This parameter determines the stride with which the input sequences are sampled. Default is 1.
            
        lr (float, optional): 
            This parameter sets the learning rate for the optimizer. Default is 0.001.
            
        epochs (int, optional): 
            This parameter sets the number of epochs for training the model. Default is 10.
            
        batch_size (int, optional): 
            This parameter sets the size of batches for training and inference. Default is 32.
            
        epoch_steps (int, optional): 
            This parameter sets the number of steps (batches) per epoch. Default is 20.
            
        prt_steps (int, optional): 
            This parameter sets the interval of epochs to print training progress. Default is 1.
            
        device (str, optional): 
            This parameter sets the device to train the model on, 'cuda' or 'cpu'. Default is 'cuda'.
            
        k (int, optional): 
            This parameter sets the hyperparameter k for loss calculation. Default is 3.
            
        verbose (int, optional): 
            This parameter sets the verbosity mode. Default is 2.
            
        random_state (int, optional): 
            This parameter sets the seed for random number generator for reproducibility. Default is 42.
            
    """
    
    def __init__(self, seq_len=100, stride=1, lr=0.0001, epochs=10, batch_size=32,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 k=3, verbose=2, random_state=42):
        """
        Initializes the AnomalyTransformer model with specified hyperparameters and training settings.
        """
        
        super(AnomalyTransformer, self).__init__(
            model_name='AnomalyTransformer', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        self.k = k

    def fit(self, X, y=None):
        """
        This method is used to train the AnomalyTransformer model on the provided dataset.

        Args:
        
            X (np.array, required): 
                This is the input data that the model will be trained on. It should be a numpy array where each row represents a different time series and each column represents a different time point.
                
            y (np.array, optional): 
                These are the true labels for the input data. If provided, they can be used to monitor the training process and adjust the model parameters. However, they are not necessary for the training process and their default value is None.

        Returns:
        
            None: 
                This method does not return any value. It modifies the state of the AnomalyTransformer object by training it on the provided data.
        """
        
        self.n_features = X.shape[1]

        train_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        self.net = AnomalyTransformerModel(
            win_size=self.seq_len,
            enc_in=self.n_features,
            c_out=self.n_features,
            e_layers=3,
            device=self.device
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
        This method computes the anomaly scores for the given input data. Anomaly scores are a measure of how much a data point deviates from what is considered normal or expected. A higher score indicates a higher likelihood of the data point being anomalous.

        Args:
        
            X (np.array, required): 
                The input data for which the anomaly scores are to be computed. It should be a numpy array where each row represents a different time series and each column represents a different time point.
                
            return_rep (bool, optional): 
                A flag that determines whether the representations should be returned along with the anomaly scores. These representations are the encoded versions of the input data as learned by the model. They can be useful for further analysis or for visualizing the data in a lower-dimensional space. The default value is False, which means that by default, the representations are not returned.

        Returns:
        
            np.array: 
                The anomaly scores for the input data. Each score corresponds to a data point in the input data. The scores are returned as a numpy array.
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
        This method defines the training process for one epoch. During each epoch, the model is trained on batches of input data provided by the DataLoader. The training process involves forward propagation, loss computation, backpropagation, and optimization steps. The loss function used is the Mean Squared Error (MSE) loss, which measures the average squared difference between the actual and predicted values. The loss is computed for each batch, and the average loss over all batches is returned.

        Args:
        
            dataloader (DataLoader): 
                The DataLoader object that provides batches of input data for training. Each batch is a tensor of shape (batch_size, sequence_length, number_of_features), where batch_size is the number of sequences in a batch, sequence_length is the length of each sequence, and number_of_features is the number of features in the data.

        Returns:
        
            float: 
                The average loss over all batches in the dataloader. This is a single floating-point number that represents the average of the MSE loss computed for each batch of data.
        """
        
        criterion = nn.MSELoss()
        loss_list = []

        for ii, batch_x in enumerate(dataloader):
            self.optimizer.zero_grad()

            input = batch_x.float().to(self.device)
            output, series, prior, _ = self.net(input)

            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.seq_len)).detach())) + torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.seq_len)).detach(),
                               series[u])))
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.seq_len)),
                    series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.seq_len)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = criterion(output, input)  # compute loss

            loss_list.append((rec_loss - self.k * series_loss).item())
            loss1 = rec_loss - self.k * series_loss
            loss2 = rec_loss + self.k * prior_loss

            # Minimax strategy
            loss1.backward(retain_graph=True)
            loss2.backward()
            self.optimizer.step()

            if self.epoch_steps != -1:
                if ii > self.epoch_steps:
                    break

        self.scheduler.step()

        return np.average(loss_list)

    def inference(self, dataloader):
        """
        This method performs inference on the data provided by the dataloader. It uses the trained model to generate anomaly scores and predictions for the input data.

        Args:
        
            dataloader (DataLoader): 
                The DataLoader object that provides the data for inference. It should contain the input data that we want to generate anomaly scores and predictions for.

        Returns:
        
            tuple: 
                A tuple containing two numpy arrays. The first array contains the anomaly scores for the input data, and the second array contains the predicted labels for the input data.
                
            - anomaly scores (np.array):
                An array of anomaly scores for the input data. Each score represents the degree of anomaly of the corresponding data point in the input data.
                
            - predictions (np.array):
                An array of predicted labels for the input data. Each label represents the predicted class (normal or anomalous) of the corresponding data point in the input data.
        """
        
        criterion = nn.MSELoss(reduction='none')
        temperature = 50
        attens_energy = []
        preds = []

        for input_data in dataloader:  # test_set
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.net(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.seq_len)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.seq_len)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.seq_len)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.seq_len)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0)  # anomaly scores
        test_energy = np.array(attens_energy)  # anomaly scores

        return test_energy, preds  # (n,d)

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


class AnomalyTransformerModel(nn.Module):
    """
    This class defines the architecture for the Anomaly Transformer model, which is specifically designed for
    detecting anomalies in time series data. The model is based on the Transformer architecture and includes
    an attention mechanism, multiple encoder layers, and a feed-forward network.
    
    Args:
        win_size (int): 
            The size of the window for the attention mechanism. This determines the number of time steps that the model looks at when computing attention weights.
            
        enc_in (int): 
            The number of features in the input data. This corresponds to the dimensionality of the input time series data.
            
        c_out (int): 
            The number of output features. This corresponds to the dimensionality of the output time series data.
            
        d_model (int, optional, default=512): 
            The dimensionality of the model. This affects the size of the internal representations that the model learns.
            
        n_heads (int, optional, default=8): 
            The number of attention heads. This determines the number of different attention weights that the model computes for each time step.
            
        e_layers (int, optional, default=3): 
            The number of layers in the encoder. Each layer includes an attention mechanism and a feed-forward network.
            
        d_ff (int, optional, default=512): 
            The dimensionality of the feed-forward network in the encoder. This affects the size of the internal representations that the model learns.
            
        dropout (float, optional, default=0.0): 
            The dropout rate. This is the probability that each element in the internal representations is set to zero during training. Dropout is a regularization technique that helps prevent overfitting.
            
        activation (str, optional, default='gelu'): 
            The activation function to use in the feed-forward network. This function introduces non-linearity into the model, allowing it to learn more complex patterns.
            
        output_attention (bool, optional, default=True): 
            Whether to output the attention weights. If true, the model outputs the attention weights in addition to the output time series data.
            
        device (str, optional, default='cuda'): 
            The device to use for tensor computations. This can be either 'cuda' for GPU computations or 'cpu' for CPU computations.
    """
    
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, device='cuda'):
        """
        Initializes the AnomalyTransformerModel.
        """
         
        super(AnomalyTransformerModel, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False,
                                         attention_dropout=dropout, output_attention=output_attention,
                                         device=device
                                         ),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        This method defines the forward pass of the AnomalyTransformerModel. It takes as input a tensor 'x' and returns the model's output. If the 'output_attention' attribute is set to true, it also returns the attention series, prior, and sigma tensors.

        Args:
        
            x (torch.Tensor): 
                The input data. It is a tensor that represents the input data that will be processed by the model.

        Returns:
        
            torch.Tensor: 
                The output of the model. It is a tensor that represents the output data generated by the model. If the 'output_attention' attribute is set to true, it also includes the attention series, prior, and sigma tensors.
        """
        
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]


class EncoderLayer(nn.Module):
    """
    The EncoderLayer class represents a single layer of the encoder part in a transformer model. This layer applies
    self-attention mechanism to the input data and then processes it through a feedforward neural network.
    
    Args:
    
        attention (nn.Module): 
            This is the attention mechanism that the layer will use. It is responsible for determining the importance of different parts of the input data.
            
        d_model (int): 
            This is the number of expected features in the input data. It defines the size of the input layer of the neural network.
        
        d_ff (int, optional, default=4*d_model): 
            This is the dimension of the feedforward network model. It defines the size of the hidden layer in the neural network.
            
        dropout (float, optional, default=0.1): 
            This is the dropout value. It is a regularization technique where randomly selected neurons are ignored during training to prevent overfitting.
            
        activation (str, optional, default="relu"):
            This is the activation function that the layer will use. It defines the output of a neuron given an input or set of inputs.
    """
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        Initializes the EncoderLayer with an attention mechanism and a feedforward network.
        """
        
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        Forward pass for the EncoderLayer.

        Args:
        
            x (torch.Tensor): 
                Input tensor.
                
            attn_mask (Optional[torch.Tensor]): 
                Attention mask.

        Returns:
        
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Output tensor, attention, mask, sigma.
        """
        
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    """
    Composes multiple EncoderLayer modules to form the encoder. This class is responsible for stacking multiple EncoderLayer modules to create the encoder for the Anomaly Transformer model.
        
    Args:
    
        attn_layers (List[nn.Module]): 
            A list of attention layers to be stacked. These layers will be sequentially stacked to form the encoder.
            
        norm_layer (nn.Module, optional): 
            The normalization layer to use at the end of the encoder. If provided, this layer will be applied to the output of the encoder.
    """
    
    def __init__(self, attn_layers, norm_layer=None):
        """
        Initializes the Encoder with a stack of attention layers and optional normalization.
        """
        
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Forward pass for the Encoder.

        Args:
        
            x (torch.Tensor): 
                Input tensor.
                
            attn_mask (Optional[torch.Tensor]): 
                Attention mask.

        Returns:
        
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: 
                Output tensor, list of attention series, list of prior series, list of sigma values.
        """
        
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class DataEmbedding(nn.Module):
    """
    Embeds input data with value and positional embeddings.
    
    Args:
    
        c_in (int, required): 
            The number of input channels.
            
        d_model (int, required): 
            The dimension of the model.
        
        dropout (float, optional, default=0.0): 
            The dropout probability.
    """
    
    def __init__(self, c_in, d_model, dropout=0.0):
        """
        Initializes the DataEmbedding module.
        """
        
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass for the DataEmbedding.

        Args:
        
            x (torch.Tensor): 
                Input tensor.

        Returns:
        
            torch.Tensor: 
                The output tensor after value and positional embedding and dropout.
        """
        
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    TokenEmbedding is a class for embedding tokens with value and positional embeddings.
    
    Args:
        
        c_in (int): 
            The number of input channels for token embedding.
            
        d_model (int): 
            The dimension of the model for token embedding.
    """
    
    def __init__(self, c_in, d_model):
        """
        Initializes the TokenEmbedding class.
        """
        
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Forward pass for the TokenEmbedding.
        
        Args:
        
            x (torch.Tensor): 
                Input tensor.
                
        Returns:
        
            torch.Tensor: 
                The output tensor after token embedding.
        """
        
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    """
    PositionalEmbedding is a class for embedding positions.
    
    Args:
    
        d_model (int): 
            The dimension of the model.
            
        max_len (int, optional): 
            The maximum length. Default is 5000.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEmbedding class.
        """
        
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for the PositionalEmbedding.

        Args:
        
            x (torch.Tensor): 
                Input tensor.

        Returns:
        
            torch.Tensor: 
                The output tensor after positional embedding.
        """
        
        return self.pe[:, :x.size(1)]


class TriangularCausalMask():
    """
    TriangularCausalMask is a class for creating a triangular causal mask.
    
    Args:
        
        B (int): 
            The batch size.
            
        L (int): 
            The sequence length.
            
        device (str, optional): 
            The device to use. Default is "cpu".
    """
    
    def __init__(self, B, L, device="cpu"):
        """
        Initializes the TriangularCausalMask class.
        """
        
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        Returns the mask.
        
        Returns:
        
            torch.Tensor: 
                The mask tensor.
        """
        
        return self._mask


class AnomalyAttention(nn.Module):
    """
    AnomalyAttention is a class for anomaly attention. It calculates attention scores based on the distance between queries and keys.
    
    Args:
    
        win_size (int): 
            The window size.
            
        mask_flag (bool, optional): 
            The mask flag. Default is True.
            
        scale (float, optional): 
            The scale. Default is None.
            
        attention_dropout (float, optional): 
            The attention dropout value. Default is 0.0.
            
        output_attention (bool, optional): 
            The output attention flag. Default is False.
            
        device (str, optional): 
            The device to use. Default is "cuda".     
    """
    
    def __init__(self, win_size, mask_flag=True, scale=None,
                 attention_dropout=0.0, output_attention=False, device='cuda'):
        """
        Initializes the AnomalyAttention class.
        """
        
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.device = device
        self.distances = torch.zeros((win_size, win_size)).to(device)
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        """
        Forward pass for the AnomalyAttention.

        Args:
        
            queries (torch.Tensor): 
                The tensor containing queries.
                
            keys (torch.Tensor): 
                The tensor containing keys.
                
            values (torch.Tensor): 
                The tensor containing values.
                
            sigma (torch.Tensor): 
                The tensor containing sigma values.
                
            attn_mask (torch.Tensor): 
                The tensor containing the attention mask.
                
        Returns:
        
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                The output tensor, series tensor, prior tensor, and sigma tensor.
        """
        
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    """
    AttentionLayer is a class for attention layer.
    
    Args:
        
        attention (nn.Module): 
            The attention module.
            
        d_model (int): 
            The dimension of the model.
            
        n_heads (int): 
            The number of heads.
            
        d_keys (int, optional): 
            The dimension of the keys. Default is None.
            
        d_values (int, optional): 
            The dimension of the values. Default is None.
    """
    
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        """
        Initializes the AttentionLayer class.
        """
        
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass for the AttentionLayer.
        
        Args:
        
            queries (torch.Tensor): 
                The input queries tensor.
                
            keys (torch.Tensor): 
                The input keys tensor.
                
            values (torch.Tensor): 
                The input values tensor.
                
            attn_mask (torch.Tensor): 
                The attention mask tensor.
                
        Returns:
        
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                The output tensor, series tensor, prior tensor, and sigma tensor.
        """
        
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
    
