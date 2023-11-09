# -*- coding: utf-8 -*-
"""
Transformer structure
adapted from https://github.com/gzerveas/mvts_transformer
"""


import math
import torch
from typing import Optional
from torch.nn.modules import TransformerEncoderLayer
from torch.nn import functional as F
from torch import Tensor
from deepod.core.networks.network_utility import _handle_n_hidden, _instantiate_class



class TokenEmbedding(torch.nn.Module):
    def __init__(self, n_features, d_model, kernel_size=3, bias=True):
        super(TokenEmbedding, self).__init__()
        # padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = torch.nn.Conv1d(in_channels=n_features, out_channels=d_model,
                                         kernel_size=kernel_size, padding=1, padding_mode='circular', bias=bias)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedPositionalEncoding(torch.nn.Module):
    r"""
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
        adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

    Args:
        d_model:
            the embed dim (required).

        dropout:
            the dropout value (default=0.1).

        max_len:
            the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        # self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        pe.requires_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Parameters
        ----------
        x: torch.Tensor, required
            shape= (sequence length, batch size, embed dim)
            the sequence fed to the positional encoder model (required).

        Returns
        -------
        output: torch.Tensor, required
            shape=(sequence length, batch size, embed dim)
        """
        x = self.pe[:, :x.size(1)]
        # x = x + self.pe[:x.size(0), :]
        return x
        # return self.dropout(x)


class LearnablePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = torch.nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        torch.nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerEncoderLayer(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='ReLU',
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                     batch_first=batch_first,
                                                     **factory_kwargs)
        # Implementation of Feedforward model
        self.conv1 = torch.nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1)

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        assert activation in ['ReLU', 'GELU'], \
            f"activation should be ReLU/GELU, not {activation}"
        self.activation = _instantiate_class("torch.nn.modules.activation", activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class TransformerBatchNormEncoderLayer(torch.nn.modules.Module):
    r"""
    This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multi-head attention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = torch.nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        assert activation in ['ReLU', 'GELU'], \
            f"activation should be ReLU/GELU, not {activation}"
        self.activation = _instantiate_class("torch.nn.modules.activation", activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(torch.nn.Module):
    """
    Transformer for encoding/representing input time series sequences
    """

    def __init__(self, n_features, n_output=20, seq_len=100, d_model=128,
                 n_heads=8, n_hidden='512', dropout=0.1,
                 token_encoding='convolutional', pos_encoding='fixed', activation='GELU', bias=False,
                 attn='self_attn', norm='LayerNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = seq_len
        self.d_model = d_model
        n_hidden, n_layers = _handle_n_hidden(n_hidden)

        # parameter check
        assert token_encoding in ['linear', 'convolutional'], \
            f"use 'linear' or 'convolutional', {token_encoding} is not supported in token_encoding"
        assert pos_encoding in ['learnable', 'fixed'],\
            f"use 'learnable' or 'fixed', {pos_encoding} is not supported in pos_encoding"
        assert norm in ['LayerNorm', 'BatchNorm'],\
            f"use 'learnable' or 'fixed', {norm} is not supported in norm"

        if token_encoding == 'linear':
            self.project_inp = torch.nn.Linear(n_features, d_model, bias=bias)
        elif token_encoding == 'convolutional':
            self.project_inp = TokenEmbedding(n_features, d_model, kernel_size=3, bias=bias)

        if pos_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=seq_len)
        elif pos_encoding == "fixed":
            self.pos_enc =  FixedPositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=seq_len)

        if norm == 'LayerNorm':
            # d_model -> n_hidden -> d_model
            encoder_layer = TransformerEncoderLayer(d_model, n_heads,
                                                    n_hidden, dropout*(1.0 - freeze),
                                                    activation=activation)
        elif norm == 'BatchNorm':
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, n_heads,
                                                             n_hidden, dropout*(1.0 - freeze),
                                                             activation=activation)

        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        assert activation in ['ReLU', 'GELU'], \
            f"activation should be ReLU/GELU, not {activation}"
        self.act = _instantiate_class("torch.nn.modules.activation", activation)

        self.dropout = torch.nn.Dropout(dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        # self.output_layer = torch.nn.Linear(d_model * seq_len, n_output, bias=bias)
        self.output_layer = torch.nn.Linear(d_model, n_output, bias=bias)

    def forward(self, X, padding_masks=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]

        # inp = X.permute(1, 0, 2)
        # inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        # inp = self.pos_enc(inp)  # add positional encoding

        # data embedding
        inp = self.project_inp(X) + self.pos_enc(X)
        # inp = self.dropout(inp)
        inp = inp.permute(1, 0, 2)

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks if padding_masks is not None else None)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        if padding_masks is None:
            padding_masks = torch.ones(X.shape[0], X.shape[1], dtype=torch.uint8).to(X.device)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # (batch_size, seq_len, 1) zero-out padding embeddings
        output = output[:, -1] # (batch_size, d_model)
        # output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output


# if __name__ == '__main__':
#     model = TSTransformerEncoder(n_features=19, seq_len=100,
#                                  token_encoding='linear',
#                                  d_model=512, n_heads=8, n_hidden='256,256',
#                                  n_output=128)
#
#     print(model)
#     a = torch.randn(256, 100, 19)
#     padding_masks = torch.ones(256, 100, dtype=int)
#     b = model(a)
#     print(b.shape)
