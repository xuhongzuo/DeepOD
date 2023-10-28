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
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class DCdetector(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, lr=0.0001, epochs=3, batch_size=256,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 n_heads=1, d_model=256, e_layers=3, patch_size=None,
                 verbose=2, random_state=42):
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
        self.criterion = nn.MSELoss()
        return

    def fit(self, X, y=None):
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
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=True)

        loss, preds = self.inference(dataloader)  # (n,d)
        
        loss_final = np.mean(loss, axis=1)  # (n,)
        loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])
        
        preds_final = np.mean(preds, axis=1)  # (n,)
        preds_final_pad = np.hstack([0 * np.ones(X.shape[0] - preds_final.shape[0]), preds_final])

        return loss_final_pad, preds_final_pad

    def training(self, dataloader):
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
        thresh = np.percentile(test_energy, 100 - 1) #hyperparameter need to be tuned
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


# Proposed Model

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class DCdetectorModel(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3, 5, 7], channel=55,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
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
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
        return self.pe[:, :x.size(1)]


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight = self.affine_weight.to(
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.affine_bias = self.affine_bias.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
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
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05,
                 output_attention=False):
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
