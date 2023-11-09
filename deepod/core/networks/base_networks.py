import torch
import numpy as np
from torch.nn.utils import weight_norm
from deepod.core.networks.ts_network_transformer import TSTransformerEncoder
from deepod.core.networks.ts_network_dilated_conv import DilatedConvEncoder
from deepod.core.networks.ts_network_tcn import TCNnet, TcnAE
# from deepod.core.base_transformer_network_dev import TSTransformerEncoder
from deepod.core.networks.network_utility import _instantiate_class, _handle_n_hidden
import torch.nn.modules.activation

sequential_net_name = ['TCN', 'GRU', 'LSTM', 'Transformer', 'ConvSeq', 'DilatedConv']


def get_network(network_name):
    maps = {
        'MlpAE': MlpAE,
        'TcnAE': TcnAE,
        'MLP': MLPnet,
        'GRU': GRUNet,
        'LSTM': LSTMNet,
        'TCN': TCNnet,
        'Transformer': TSTransformerEncoder,
        'ConvSeq': ConvSeqEncoder,
        'DilatedConv': DilatedConvEncoder
    }

    if network_name in maps.keys():
        return maps[network_name]
    else:
        raise NotImplementedError(f'network is not supported. '
                                  f'please use network structure in {maps.keys()}')


class ConvNet(torch.nn.Module):
    """Convolutional Network

    Args:
        n_features (int):
            number of input data features
        kernel_size (int):
            kernel size (Default=1)
        n_hidden (int):
            number of hidden units in hidden layers (Default=8)
        n_layers (int):
            number of layers (Default=5)
        activation (str):
            name of activation layer,
            activation should be implemented in torch.nn.module.activation
            (Default='ReLU')
        bias (bool):
            use bias or not
            (Default=False)

    """
    def __init__(self, n_features, kernel_size=1, n_hidden=8, n_layers=5,
                 activation='ReLU', bias=False):
        super(ConvNet, self).__init__()

        self.layers = []

        in_channels = n_features
        for i in range(n_layers+1):
            self.layers += [
                torch.nn.Conv1d(in_channels, n_hidden,
                                kernel_size=kernel_size,
                                bias=bias)
            ]
            if i != n_layers:
                self.layers += [
                    # torch.nn.LeakyReLU(inplace=True)
                    _instantiate_class(module_name="torch.nn.modules.activation",
                                       class_name=activation)
                ]
            in_channels = n_hidden

        self.net = torch.nn.Sequential(*self.layers)

        return

    def forward(self, x):
        return self.net(x)


class MlpAE(torch.nn.Module):
    """MLP-based AutoEncoder"""
    def __init__(self, n_features, n_hidden='500,100', n_emb=20, activation='ReLU',
                 bias=False, batch_norm=False,
                 skip_connection=None, dropout=None
                 ):
        super(MlpAE, self).__init__()

        if type(n_hidden)==int:
            n_hidden = [n_hidden]
        if type(n_hidden)==str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        self.encoder_layers = []
        for i in range(num_layers+1):
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
            self.encoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=bias, batch_norm=batch_norm,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=skip_connection if i != num_layers else 0,
                                                dropout=dropout if i != num_layers else None)]

        self.decoder_layers = []
        for i in range(num_layers+1):
            in_channels = n_emb if i == 0 else n_hidden[num_layers-i]
            out_channels = n_features if i == num_layers else n_hidden[num_layers-1-i]
            self.decoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=bias, batch_norm=batch_norm,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=skip_connection if i != num_layers else 0,
                                                dropout=dropout if i != num_layers else None)]

        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        self.decoder = torch.nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        enc = self.encoder(x)
        xx = self.decoder(enc)
        return xx, enc


class MLPnet(torch.nn.Module):
    """MLP-based Representation Network"""
    def __init__(self, n_features, n_hidden='500,100', n_output=20, mid_channels=None,
                 activation='ReLU', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_output = n_output

        if type(n_hidden)==int:
            n_hidden = [n_hidden]
        if type(n_hidden)==str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        # for only use one kind of activation layer
        if type(activation) == str:
            activation = [activation] * num_layers
            activation.append(None)

        assert len(activation) == len(n_hidden)+1, 'activation and n_hidden are not matched'

        self.layers = []
        for i in range(num_layers+1):
            in_channels, out_channels = self.get_in_out_channels(i, num_layers, n_features,
                                                                 n_hidden, n_output, skip_connection)
            self.layers += [
                LinearBlock(in_channels, out_channels,
                            mid_channels=mid_channels,
                            bias=bias, batch_norm=batch_norm,
                            activation=activation[i],
                            skip_connection=skip_connection if i != num_layers else 0,
                            dropout=dropout if i !=num_layers else None)
            ]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.network(x)
        return x

    def get_in_out_channels(self, i, num_layers, n_features, n_hidden, n_output, skip_connection):
        if skip_connection is None:
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_output if i == num_layers else n_hidden[i]
        elif skip_connection == 'concat':
            in_channels = n_features if i == 0 else np.sum(n_hidden[:i])+n_features
            out_channels = n_output if i == num_layers else n_hidden[i]
        else:
            raise NotImplementedError('')
        return in_channels, out_channels


class LinearBlock(torch.nn.Module):
    """Linear Block"""
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 activation='Tanh', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(LinearBlock, self).__init__()

        self.skip_connection = skip_connection

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

        # Tanh, ReLU, LeakyReLU, Sigmoid
        if activation is not None:
            self.act_layer = _instantiate_class("torch.nn.modules.activation", activation)
        else:
            self.act_layer = torch.nn.Identity()

        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

        self.batch_norm = batch_norm
        if batch_norm is True:
            dim = out_channels if mid_channels is None else mid_channels
            self.bn_layer = torch.nn.BatchNorm1d(dim, affine=bias)

    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.act_layer(x1)

        if self.batch_norm is True:
            x1 = self.bn_layer(x1)

        if self.dropout is not None:
            x1 = self.dropout_layer(x1)

        if self.skip_connection == 'concat':
            x1 = torch.cat([x, x1], axis=1)

        return x1


# class GRUNet(torch.nn.Module):
#     def __init__(self, n_features, hidden_dim=20, n_output=20, layers=1):
#         super(GRUNet, self).__init__()
#         self.gru = torch.nn.GRU(n_features, hidden_size=hidden_dim,
#                                 batch_first=True,
#                                 num_layers=layers)
#         self.hidden2output = torch.nn.Linear(hidden_dim, n_output)
#
#     def forward(self, x):
#         _, hn = self.gru(x)
#         out = hn[0, :]
#         out = self.hidden2output(out)
#         return out


class GRUNet(torch.nn.Module):
    """GRU Network"""
    def __init__(self, n_features, n_hidden='20', n_output=20,
                 bias=False, dropout=None, activation='ReLU'):
        super(GRUNet, self).__init__()

        hidden_dim, n_layers = _handle_n_hidden(n_hidden)

        if dropout is None:
            dropout = 0.0

        self.gru = torch.nn.GRU(n_features, hidden_dim, n_layers,
                                batch_first=True,
                                bias=bias,
                                dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        out, h = self.gru(x)
        out = self.fc(out[:, -1])
        return out


class LSTMNet(torch.nn.Module):
    """LSTM Network"""
    def __init__(self, n_features, n_hidden='20', n_output=20,
                 bias=False, dropout=None, activation='ReLU'):
        super(LSTMNet, self).__init__()

        hidden_dim, n_layers = _handle_n_hidden(n_hidden)

        if dropout is None:
            dropout = 0.0

        self.lstm = torch.nn.LSTM(n_features, hidden_size=hidden_dim,
                                  batch_first=True,
                                  bias=bias,
                                  dropout=dropout,
                                  num_layers=n_layers)
        self.fc = torch.nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        out, (hn, c) = self.lstm(x)
        out = self.fc(out[:, -1])
        return out


class ConvSeqEncoder(torch.nn.Module):
    """
    this network architecture is from NeurTraL-AD
    """
    def __init__(self, n_features, n_hidden='100', n_output=128, n_layers=3, seq_len=100,
                 bias=True, batch_norm=True, activation='ReLU'):
        super(ConvSeqEncoder, self).__init__()

        n_hidden, _ = _handle_n_hidden(n_hidden)

        self.bias = bias
        self.batch_norm = batch_norm
        self.activation = activation

        enc = [self._make_layer(n_features, n_hidden, (3,1,1))]
        in_dim = n_hidden
        window_size = seq_len
        for i in range(n_layers - 2):
            out_dim = n_hidden*2**i
            enc.append(self._make_layer(in_dim, out_dim, (3,2,1)))
            in_dim =out_dim
            window_size = np.floor((window_size+2-3)/2)+1

        self.enc = torch.nn.Sequential(*enc)
        self.final_layer = torch.nn.Conv1d(in_dim, n_output, int(window_size), 1, 0)

    def _make_layer(self, in_dim, out_dim, conv_param):
        down_sample = None
        if conv_param is not None:
            down_sample = torch.nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                                          kernel_size=conv_param[0], stride=conv_param[1], padding=conv_param[2],
                                          bias=self.bias)
        elif in_dim != out_dim:
            down_sample = torch.nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                                          kernel_size=1, stride=1, padding=0,
                                          bias=self.bias)

        layer = ConvResBlock(in_dim, out_dim, conv_param, down_sample=down_sample,
                             batch_norm=self.batch_norm, bias=self.bias, activation=self.activation)

        return layer

    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.enc(x)
        z = self.final_layer(z)
        return z.squeeze(-1)


class ConvResBlock(torch.nn.Module):
    """Convolutional Residual Block"""
    def __init__(self, in_dim, out_dim, conv_param=None, down_sample=None,
                 batch_norm=False, bias=False, activation='ReLU'):
        super(ConvResBlock, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_dim, in_dim,
                                     kernel_size=1, stride=1, padding=0, bias=bias)

        if conv_param is not None:
            self.conv2 = torch.nn.Conv1d(in_dim, in_dim,
                                         conv_param[0], conv_param[1], conv_param[2],bias=bias)
        else:
            self.conv2 = torch.nn.Conv1d(in_dim, in_dim,
                                         kernel_size=3, stride=1, padding=1, bias=bias)

        self.conv3 = torch.nn.Conv1d(in_dim, out_dim,
                                     kernel_size=1, stride=1, padding=0, bias=bias)

        if batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(in_dim)
            self.bn2 = torch.nn.BatchNorm1d(in_dim)
            self.bn3 = torch.nn.BatchNorm1d(out_dim)
            if down_sample:
                self.bn4 = torch.nn.BatchNorm1d(out_dim)

        self.act = _instantiate_class("torch.nn.modules.activation", activation)
        self.down_sample = down_sample
        self.batch_norm = batch_norm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)
            if self.batch_norm:
                residual = self.bn4(residual)

        out += residual
        out = self.act(out)

        return out

#
# if __name__ == '__main__':
#     model = ConvSeqEncoder(n_features=19, n_hidden='512', n_layers=4, seq_len=30, batch_norm=False,
#                            n_output=1, activation='LeakyReLU')
#     print(model)
#     a = torch.randn(32, 30, 19)
#
#     b =  model(a)
#     print(b.shape)