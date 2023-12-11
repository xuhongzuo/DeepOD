# Revised by Yiyuan Yang on 2023/10/28
# There are some differences between the original code and the revised code.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import math
import time
from tkinter import _flatten
from einops import rearrange, reduce, repeat
from deepod.utils.utility import get_sub_seqs
from deepod.core.base_model import BaseDeepAD


def my_kl_loss(p, q):
    """
    Function to calculate the Kullback-Leibler (KL) divergence loss.
    
    Args:
    
        p (torch.Tensor): 
            Input probability distribution tensor representing the true distribution. Each element should represent a probability corresponding to the true distribution. The tensor should sum to 1 across dimensions that represent probability distributions.
            
        q (torch.Tensor): 
            Input probability distribution tensor representing the approximate distribution. Typically representing the model's predicted probabilities. It should have the same shape as 'p' and also sum to 1 across the appropriate dimensions.
            
    Returns:
    
        torch.Tensor: 
            The KL divergence loss between the true and approximate distributions. The loss is averaged over all distributions.
    """
    
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class DCdetector(BaseDeepAD):
    """
    DCdetector: Dual Attention Contrastive Representation Learning
    for Time Series Anomaly Detection (KDD'23)
    
    This class implements the DCdetector model. It uses a dual attention mechanism and contrastive representation learning to detect anomalies.
    The model is trained to learn representations of normal time series and then uses these representations to detect anomalies in new time series. 
    The dual attention mechanism allows the model to focus on different parts of the time series at different times, which can help it detect a wider range of anomalies.
    The contrastive representation learning encourages the model to learn representations that are similar for normal time series and different for anomalous time series.
    
    Args:
    
        seq_len (int, optional): 
            The length of the sequence. This parameter determines the length of the time series that the model will consider at a time. Default is 100.
        
        stride (int, optional): 
            The stride. This parameter determines the step size when moving the sliding window across the time series. Default is 1.
        
        lr (float, optional): 
            The learning rate. This parameter determines the step size that the model will take when updating its parameters during training. Default is 0.0001.
        
        epochs (int, optional):
            The number of epochs. This parameter determines the number of times the model will iterate over the entire dataset during training. Default is 5.
        
        batch_size (int, optional): 
            The batch size. This parameter determines the number of samples that the model will process at a time during training. Default is 128.
        
        epoch_steps (int, optional): 
            The number of epoch steps. This parameter determines the number of steps that the model will take in each epoch. Default is 20.
        
        prt_steps (int, optional): 
            The number of prt steps. This parameter determines the number of steps that the model will take before printing the training progress. Default is 1.
        
        device (str, optional): 
            The device to use. This parameter determines the device (CPU or GPU) that the model will use for computation. Default is 'cuda'.
        
        n_heads (int, optional): 
            The number of heads. This parameter determines the number of attention heads in the dual attention mechanism. Default is 1.
        
        d_model (int, optional): 
            The model dimension. This parameter determines the dimensionality of the model's output space. Default is 256.
        
        e_layers (int, optional): 
            The number of e layers. This parameter determines the number of encoder layers in the model. Default is 3.
        
        patch_size (list, optional): 
            The size of the patch. This parameter determines the size of the patches that the model will extract from the time series. Default is [5].
        
        verbose (int, optional): 
            The verbosity level. This parameter determines the amount of information the model will print during training. Default is 2.
        
        random_state (int, optional):
            The random state. This parameter determines the seed for the random number generator. Default is 42.
        
        threshold_ (int, optional):
            The threshold. This parameter determines the threshold for anomaly detection. Default is 1.
            
    """
    
    def __init__(self, seq_len=100, stride=1, lr=0.0001, epochs=5, batch_size=128,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 n_heads=1, d_model=256, e_layers=3, patch_size=None,
                 verbose=2, random_state=42, threshold_ = 1):
        """
        Initialize the DCdetector.
        """
        
        super(DCdetector, self).__init__(
            model_name='DCdetector', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        if patch_size is None:
            self.patch_size = [5]  # seq_len must be divisible by patch_size
        else:
            self.patch_size = patch_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.e_layers = e_layers
        self.threshold = threshold_
        self.criterion = nn.MSELoss()
        return

    def fit(self, X, y=None):
        """
        Fit the model to the input data. This method is responsible for training the DCdetector model on the provided time series data. 
        The training process involves extracting subsequences from the time series, constructing a model, and optimizing it over a number of epochs.

        The 'X' parameter is the only required input as this is an unsupervised model, meaning it does not require target labels for training. 
        The method involves setting up a DataLoader for the training sequences, defining an optimizer and a learning rate scheduler, 
        and iterating over the dataset for a specified number of epochs to train the model.
        
        Args:
        
            X (numpy.ndarray): 
                The input data for training. It is a numpy array of shape (n_samples, n_features), where 'n_samples' is the number of time series segments, 
                and 'n_features' is the number of observations per time step.

            y (numpy.ndarray, optional): 
                The target data is not used in this unsupervised training method. It is present to maintain consistency with supervised learning interfaces. 
                Default is None.
        """
        
        self.n_features = X.shape[1]

        train_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        self.model = DCdetectorModel(win_size=self.seq_len, enc_in=self.n_features, c_out=self.n_features, n_heads=self.n_heads,
                                d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size,
                                channel=self.n_features).to(self.device)

        dataloader = DataLoader(train_seqs, batch_size=self.batch_size,
                                shuffle=True, pin_memory=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.model.train()
        for e in range(self.epochs):
            loss = self.training(dataloader)
            print(f'Epoch {e + 1}')

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return

    def decision_function(self, X, return_rep=False):
        """
        Compute the anomaly scores or representations for the input data using the trained DCdetector model. 
        
        Args:
        
            X (numpy.ndarray): 
                The input data to compute the decision function for. It should have the same number of features as the data used to fit the model.

            return_rep (bool, optional): 
                A flag that determines whether to return the raw anomaly scores (False) or the learned representations (True). Default is False.
                
        Returns:
        
            numpy.ndarray: 
                An array of anomaly scores. The shape is (n_samples,), where 'n_samples' is the number of input sequences.
        
        """
        
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=True)

        loss, _ = self.inference(dataloader)  # (n,d)
        loss_final = np.mean(loss, axis=1)  # (n,)
        loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])

        return loss_final_pad

    def training(self, dataloader):
        """
        Train the model for one epoch using the provided DataLoader. It iterates over the dataset, calculates losses for each batch,
        and updates the model parameters using backpropagation.

        The method calculates two types of losses, which represent the model's performance in terms
        of its contrastive learning objective. These losses are combined to form the final loss used for backpropagation.
        
        Args:
        
            dataloader (torch.utils.data.DataLoader): 
                The DataLoader that provides batches of data for training the model.

        Returns:
        
            float: The average loss for the epoch, computed over all batches.
        """
        
        loss_list = []

        for ii, batch_x in enumerate(dataloader):
            self.optimizer.zero_grad()

            batch_x = batch_x.float().to(self.device)
            series, prior = self.model(batch_x)

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

            loss = prior_loss - series_loss
            loss_list.append(loss.item())

            loss.backward()
            self.optimizer.step()

            if self.epoch_steps != -1:
                if ii > self.epoch_steps:
                    break

        self.scheduler.step()

        return np.average(loss_list)

    def inference(self, dataloader):
        """
        Perform inference on the data provided by the DataLoader. It processes the data through the trained model
        to compute losses, which are then combined to generate anomaly scores.

        The method uses a temperature parameter to scale the losses and applies a softmax to obtain a probability
        distribution over the anomaly scores, which can be used to rank the inputs by their likelihood of being anomalies.

        Args:
        
            dataloader (torch.utils.data.DataLoader): 
                The DataLoader that provides batches of data for inference.

        Returns:
        
            tuple: A tuple containing two elements:
            
                - numpy.ndarray: 
                    An array of anomaly scores for each sequence in the data.
                
                - list: 
                    An empty list, as placeholders for any additional outputs.
        """
        
        temperature = 10 #hyperparameter need to be tuned
        attens_energy = []
        preds = []

        for input_data in dataloader:  # test_set
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
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
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0)  # anomaly scores
        test_energy = np.array(attens_energy)  # anomaly scores
        
        return test_energy, preds  # (n,d)

    def predict(self, X, return_confidence=True):
        """
        Predict whether each sequence in the input data is an anomaly or not. The method computes anomaly scores
        using the trained model and compares them to a threshold, classifying sequences as normal or anomalous.

        The threshold is determined by the percentile of anomaly scores on the test set, which is a hyperparameter
        that can be tuned. Additionally, the method can return confidence scores, which represent the model's certainty
        in its predictions.

        Args:
        
            X (numpy.ndarray): 
                The input data sequences.
                
            return_confidence (bool, optional): 
                Whether to return confidence scores along with predictions. Default is True.

        Returns:
        
            numpy.ndarray or tuple: 
                If return_confidence is False, returns an array of binary predictions (0 for normal, 1 for anomaly).
                If True, returns a tuple with the array of predictions and an array of confidence scores.
        """
        
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=True)

        attens_energy = []
        for input_data in dataloader: # threhold
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.seq_len)).detach())
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.seq_len)),
                        series[u].detach())
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.seq_len)).detach())
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.seq_len)),
                        series[u].detach())

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        thresh = np.percentile(test_energy, 100 - self.threshold) #hyperparameter need to be tuned
        print('Threshold:', thresh)
        
        temperature = 10 #hyperparameter need to be tuned
        attens_energy = []
        preds = []

        for input_data in dataloader:  # test_set
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
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
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0)  # anomaly scores
        test_energy = np.array(attens_energy)  # anomaly scores

        preds = (test_energy > thresh).astype(int)

        loss_final = np.mean(test_energy, axis=1)  # (n,)
        loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])        
        preds_final = np.mean(preds, axis=1)  # (n,)
        preds_final_pad = np.hstack([0 * np.ones(X.shape[0] - preds_final.shape[0]), preds_final])

        if return_confidence:
            confidence = self._predict_confidence(loss_final_pad)
            return preds_final_pad, confidence
        else:
            return preds_final_pad

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







# Proposed Model

class Encoder(nn.Module):
    """
    Represents an encoder module that applies a series of attention layers to transform the input data.

    This encoder is designed to handle inputs that are split into patches, both in terms of patch size and number.
    It applies attention mechanisms to learn the relationships between different patches and within patches themselves.
    The outputs of this encoder are lists that are generated by the attention layers, which can be further processed for tasks such as anomaly detection in time series.

    Args:
    
        attn_layers (List[nn.Module]): 
            A list of attention layers to be applied sequentially to the input data.
            
        norm_layer (nn.Module, optional): 
            A normalization layer applied to the output of the attention layers. Default is None.
    """
    
    def __init__(self, attn_layers, norm_layer=None):
        """
        Initializes the Encoder module with the provided attention layers and optional normalization layer.
        """
        
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        """
        Conducts the forward pass for the Encoder module, processing input data through a series of attention layers.

        Args:
        
            x_patch_size (Tensor): 
                Input data for the patch size.
                
            x_patch_num (Tensor): 
                Input data for the patch number.
                
            x_ori (Tensor): 
                Original input data.
                
            patch_index (int): 
                Index of the patch.
                
            attn_mask (Tensor, optional): 
                An optional mask for the attention mechanism. Default is None.
                
        Returns:
        
            tuple: 
                A tuple containing two lists:
                
                - series_list (List[Tensor]): 
                    List of outputs from the attention layers for the series.
                
                - prior_list (List[Tensor]): 
                    List of outputs from the attention layers for the prior.
        """
        
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class DCdetectorModel(nn.Module):
    """
    Defines the DCdetectorModel for time series anomaly detection.

    This model uses a dual attention encoder to process time series data, which has been split into multiple scales of patches.
    It learns representations at both patch and sub-patch scales and aggregates them to form a comprehensive feature set.
    The model can output attention weights for interpretability if required.

    Args:
    
        win_size (int): 
            The size of the window to look back in the time series.
            
        enc_in (int): 
            The number of input features.
            
        c_out (int): 
            The number of output channels.
            
        n_heads (int, optional): 
            The number of attention heads. Default is 1.
            
        d_model (int, optional): 
            The dimensionality of the model. Default is 256.
            
        e_layers (int, optional): 
            The number of encoder layers. Default is 3.
            
        patch_size (List[int], optional): 
            The sizes of patches to split the input data into. Default is [3, 5, 7].
            
        channel (int, optional): 
            The number of channels in the data. Default is 55.
            
        d_ff (int, optional): 
            The dimensionality of the feedforward network. Default is 512.
            
        dropout (float, optional): 
            The dropout rate. Default is 0.0.
            
        activation (str, optional): 
            The activation function to use. Default is 'gelu'.
            
        output_attention (bool, optional): 
            Whether to output attention weights. Default is True.
    """
    
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3, 5, 7], channel=55,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        """
        Initialize the DCdetectorModel.
        """
        
        super(DCdetectorModel, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size

        # Patching List
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size // patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)

        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout,
                                  output_attention=output_attention),
                    d_model, patch_size, channel, n_heads, win_size) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        Forward pass of the DCdetectorModel.

        Args:
        
            x (torch.Tensor): 
                Input tensor.

        Returns:
        
            tuple: 
                If output_attention is True, returns a tuple containing series_patch_mean and prior_patch_mean.
                If output_attention is False, returns None.
        """
        
        B, L, M = x.shape  # Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x)

        # Mutil-scale Patching Operation
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size

            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p=patchsize)
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)

            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)

        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))

        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None


class DataEmbedding(nn.Module):
    """
    Embeds input data with value and positional embeddings.
    
    Args:
    
        c_in (int, required): 
            The number of input channels.
            
        d_model (int, required): 
            The dimension of the model.
        
        dropout (float, optional, default=0.05): 
            The dropout probability.
    """
    
    def __init__(self, c_in, d_model, dropout=0.05):
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


class RevIN(nn.Module):
    """
    RevIN (Reversible Instance Normalization) is a normalization layer that can be 
    reversed to reconstruct the input from its normalized output.

    Args:
    
        num_features (int): 
            Number of features or channels in the input data. Default is None.
        
        eps (float): 
            A small value added to the denominator for numerical stability. Default is 1e-5.
            
        affine (bool): 
            If True, the layer has learnable affine parameters. Default is True.
    """
        
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Initializes the RevIN module with specified attributes.
        """
        
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Forward pass for the normalization or denormalization based on the mode.

        Args:
            x (torch.Tensor): 
                The input tensor to be normalized or denormalized.
            mode (str): 
                A string that is either 'norm' for normalization or 'denorm' for denormalization. Default is 'norm'.

        Returns:
        
            torch.Tensor: 
                The normalized or denormalized tensor.
        """
        
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """
        Initializes learnable affine parameters if the affine attribute is True.
        """
 
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight = self.affine_weight.to(
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.affine_bias = self.affine_bias.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def _get_statistics(self, x):
        """
        Computes the mean and standard deviation of the input tensor.

        Args:
        
            x (torch.Tensor): 
                The input tensor for which the statistics are to be computed.
        """
        
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        Normalizes the input tensor using the computed mean and standard deviation.

        Args:
        
            x (torch.Tensor): 
                The input tensor to be normalized.

        Returns:
        
            torch.Tensor: 
                The normalized tensor.
        """
        
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        Reverses the normalization process to reconstruct the original input tensor.

        Args:
            x (torch.Tensor): 
                The normalized tensor to be denormalized.

        Returns:
        
            torch.Tensor: 
                The denormalized tensor.
        """
        
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class AttentionLayer(nn.Module):
    """
    A custom AttentionLayer module that computes self-attention with patch embeddings.

    Args:
    
        attention: 
            The attention mechanism to be used.
            
        d_model (int): 
            Dimension of the input feature space.
            
        patch_size (int): 
            Size of the patches to be extracted from input.
        
        channel (int): 
            Number of input channels.
        
        n_heads (int): 
            Number of attention heads.
        
        win_size (int): 
            Window size for attention mechanism.
        
        d_keys (int): 
            Dimension of keys (defaults to d_model/n_heads if not provided).
        
        d_values (int):
            Dimension of values (defaults to d_model/n_heads if not provided).
    """
        
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        """
        Initializes the AttentionLayer with the given parameters.
        """
        
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask):
        """
        Propagates the input through the layer.

        Args:
        
            x_patch_size (torch.Tensor): 
                Input features based on patch size. Default is None.
                
            x_patch_num (torch.Tensor): 
                Input features based on patch number. Default is None.
                
            x_ori (torch.Tensor): 
                Original input features. Default is None.
                
            patch_index (int): 
                Index of the current patch. Default is None.
                
            attn_mask (torch.Tensor): 
                Attention mask to avoid attending to certain positions. Default is None.
        
        Returns:
        
            A tuple of (series, prior) where series is the attended features and
            prior is the prior attention distribution if output_attention is true.
        """
        
        # patch_size
        B, L, M = x_patch_size.shape
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_projection(queries_patch_size).view(B, L, H, -1)
        keys_patch_size = self.patch_key_projection(keys_patch_size).view(B, L, H, -1)

        # patch_num
        B, L, M = x_patch_num.shape
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_projection(queries_patch_num).view(B, L, H, -1)
        keys_patch_num = self.patch_key_projection(keys_patch_num).view(B, L, H, -1)

        # x_ori
        B, L, _ = x_ori.shape
        values = self.value_projection(x_ori).view(B, L, H, -1)

        series, prior = self.inner_attention(
            queries_patch_size, queries_patch_num,
            keys_patch_size, keys_patch_num,
            values, patch_index,
            attn_mask
        )

        return series, prior


class DAC_structure(nn.Module):
    """
    A module that defines a Dual Attention Concatenative (DAC) structure for 
    attention-based neural networks, specifically designed for tasks that require 
    attention mechanisms over patches of an input tensor.

    Args:
    
        win_size (tuple of int): 
            The size of the window over which attention is computed.
            
        patch_size (tuple of int): 
            The size of each patch within the window.
        
        channel (int): 
            The number of input channels.
            
        mask_flag (bool, optional): 
            Whether to apply an attention mask. Defaults to True.
            
        scale (float, optional):
            Scale factor for attention scores. If None, it is set dynamically. Defaults to None.
            
        attention_dropout (float, optional): 
            Dropout rate for the attention weights. Defaults to 0.05.
            
        output_attention (bool, optional): 
            Whether to output the attention maps. Defaults to False.
    """
    
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05,
                 output_attention=False):
        """
        Initializes the DAC_structure with the specified parameters.
        """
        
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, values, patch_index,
                attn_mask):
        """
        Forward pass for the DAC_structure.

        Args:
        
            queries_patch_size (Tensor): 
                Query tensor segmented by patch size.
                
            queries_patch_num (Tensor): 
                Query tensor segmented by patch number.
                
            keys_patch_size (Tensor): 
                Key tensor segmented by patch size.
                
            keys_patch_num (Tensor): 
                Key tensor segmented by patch number.
                
            values (Tensor): 
                Value tensor.
                
            patch_index (int): 
                Index of the current patch being processed.
                
            attn_mask (Tensor): 
                Mask tensor for attention.

        Returns:
        
            Tuple[Tensor, Tensor]：
                if output_attention is True, containing attention maps for patch size and patch number respectively. Otherwise, returns None.
        """
        
        # Patch-wise Representation
        B, L, H, E = queries_patch_size.shape  # batch_size*channel, patch_num, n_head, d_model/n_head
        scale_patch_size = self.scale or 1. / math.sqrt(E)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size,
                                         keys_patch_size)  # batch*ch, nheads, p_num, p_num
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1))  # B*D_model H N N

        # In-patch Representation
        B, L, H, E = queries_patch_num.shape  # batch_size*channel, patch_size, n_head, d_model/n_head
        scale_patch_num = self.scale or 1. / math.sqrt(E)
        scores_patch_num = torch.einsum("blhe,bshe->bhls", queries_patch_num,
                                        keys_patch_num)  # batch*ch, nheads, p_size, p_size
        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1))  # B*D_model H S S

        # Upsampling
        series_patch_size = repeat(series_patch_size, 'b l m n -> b l (m repeat_m) (n repeat_n)',
                                   repeat_m=self.patch_size[patch_index], repeat_n=self.patch_size[patch_index])
        series_patch_num = series_patch_num.repeat(1, 1, self.window_size // self.patch_size[patch_index],
                                                   self.window_size // self.patch_size[patch_index])
        series_patch_size = reduce(series_patch_size, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
        series_patch_num = reduce(series_patch_num, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)

        if self.output_attention:
            return series_patch_size, series_patch_num
        else:
            return (None)
