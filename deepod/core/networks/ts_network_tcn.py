# TCN is partially adapted from https://github.com/locuslab/TCN

import torch
from torch.nn.utils import weight_norm
from deepod.core.networks.network_utility import _instantiate_class, _handle_n_hidden


class TcnAE(torch.nn.Module):
    """Temporal Convolutional Network-based AutoEncoder"""
    def __init__(self, n_features, n_hidden='500,100', n_emb=20, activation='ReLU', bias=False,
                 kernel_size=2, dropout=0.2):
        super(TcnAE, self).__init__()

        if type(n_hidden) == int:
            n_hidden = [n_hidden]
        if type(n_hidden) == str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        encoder_layers = []
        # encoder
        for i in range(num_layers+1):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
            encoder_layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                                stride=1, dilation=dilation_size,
                                                padding=padding_size, dropout=dropout, bias=bias,
                                                activation=activation)]

        # decoder
        decoder_n_hidden = n_hidden[::-1]
        decoder_layers = []
        for i in range(num_layers+1):
            # no dilation in decoder
            in_channels = n_emb if i == 0 else decoder_n_hidden[i-1]
            out_channels = n_features if i==num_layers else decoder_n_hidden[i]
            dilation_size = 2 ** (num_layers-i)
            padding_size = (kernel_size-1) * dilation_size
            decoder_layers += [TcnResidualBlockTranspose(in_channels, out_channels, kernel_size,
                                                         stride=1, dilation=dilation_size,
                                                         padding=padding_size, dropout=dropout, bias=bias,
                                                         activation=activation)]

        # # to register parameters in list of layers, each layer must be an object
        # self.enc_layer_names = ["enc_" + str(num) for num in range(len(encoder_layers))]
        # self.dec_layer_names = ["dec_" + str(num) for num in range(len(decoder_layers))]
        # for name, layer in zip(self.enc_layer_names, self.encoder_layers):
        #     setattr(self, name, layer)
        # for name, layer in zip(self.dec_layer_names, self.decoder_layers):
        #     setattr(self, name, layer)

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        enc = self.encoder(out)
        dec = self.decoder(enc)
        return dec.permute(0, 2, 1), enc.permute(0, 2, 1)


class TCNnet(torch.nn.Module):
    """Temporal Convolutional Network (TCN) for encoding/representing input time series sequences"""
    def __init__(self, n_features, n_hidden='8', n_output=20,
                 kernel_size=2, bias=False,
                 dropout=0.2, activation='ReLU'):
        super(TCNnet, self).__init__()
        self.layers = []
        self.num_inputs = n_features

        if type(n_hidden) == int:
            n_hidden = [n_hidden]
        if type(n_hidden) == str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        if dropout is None:
            dropout = 0.0

        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_hidden[i]
            self.layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size,
                                             padding=padding_size, dropout=dropout,
                                             bias=bias, activation=activation)]
        self.network = torch.nn.Sequential(*self.layers)
        self.l1 = torch.nn.Linear(n_hidden[-1], n_output, bias=bias)

    def forward(self, x):
        out = self.network(x.transpose(2, 1)).transpose(2, 1)[:, -1]
        rep = self.l1(out)
        return rep
        # # x shape[bs, seq_len, embed]
        # x = x.permute(0, 2, 1)
        # out = self.network(x) # output shape is [bs, n_output, seq_len]
        # return out.permute(0, 2, 1)


class TcnResidualBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, activation='ReLU', bias=True):
        super(TcnResidualBlock, self).__init__()

        self.conv1 = weight_norm(torch.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                 stride=stride, padding=padding, bias=bias,
                                                 dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(torch.nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                 stride=stride, padding=padding, bias=bias,
                                                 dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1,
                                       self.conv2, self.chomp2, self.act2, self.dropout2)
        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.act = _instantiate_class("torch.nn.modules.activation", activation)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x shape:(bs, embed, seq_len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out+res)


class TcnResidualBlockTranspose(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, activation='ReLU', bias=False):
        super(TcnResidualBlockTranspose, self).__init__()
        self.conv1 = weight_norm(torch.nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
                                                          stride=stride, padding=padding, bias=bias,
                                                          dilation=dilation))

        self.pad1 = Pad1d(padding)
        self.act1 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(torch.nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                                          stride=stride, padding=padding, bias=bias,
                                                          dilation=dilation))
        self.pad2 = Pad1d(padding)
        self.act2 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(self.dropout1, self.act1, self.pad1, self.conv1,
                                       self.dropout2, self.act2, self.pad2, self.conv2)
        self.downsample = torch.nn.ConvTranspose1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.act = _instantiate_class("torch.nn.modules.activation", activation)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class Pad1d(torch.nn.Module):
    def __init__(self, pad_size):
        super(Pad1d, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        return torch.cat([x, x[:, :, -self.pad_size:]], dim = 2).contiguous()


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Clipped module, clipped the extra padding
        """
        return x[:, :, :-self.chomp_size].contiguous()

