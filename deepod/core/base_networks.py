import importlib
import warnings
import math
import torch
import numpy as np
from torch.nn.utils import weight_norm
from deepod.core.base_transformer_network import TSTransformerEncoder
from deepod.core.utils import _instantiate_class, _handle_n_hidden


sequential_net_name = ['TCN', 'GRU', 'LSTM', 'Transformer']


def get_network(network_name):
    maps = {
        'MlpAE': MlpAE,
        'TcnAE': TcnAE,
        'MLP': MLPnet,
        'GRU': GRUNet,
        'LSTM': LSTMNet,
        'TCN': TCNnet,
        'Transformer': TSTransformerEncoder
    }

    if network_name in maps.keys():
        return maps[network_name]
    else:
        raise NotImplementedError(f'network is not supported. '
                                  f'please use network structure in {maps.keys()}')


class ConvNet(torch.nn.Module):
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


class TcnAE(torch.nn.Module):
    def __init__(self, n_features, n_hidden='500,100', n_emb=20, activation='ReLU', bias=None,
                 kernel_size=2, dropout=0.2):
        super(TcnAE, self).__init__()

        if type(n_hidden)==int:
            n_hidden = [n_hidden]
        if type(n_hidden)==str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        self.encoder_layers = []
        # encoder
        for i in range(num_layers+1):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
            self.encoder_layers += [TcnResidualBlock(in_channels, out_channels, kernel_size,
                                                     stride=1, dilation=dilation_size,
                                                     padding=padding_size, dropout=dropout, bias=bias,
                                                     activation=activation)]

        # decoder
        decoder_n_hidden = n_hidden[::-1]
        self.decoder_layers = []
        for i in range(num_layers+1):
            # no dilation in decoder
            in_channels = n_emb if i == 0 else decoder_n_hidden[i-1]
            out_channels = n_features if i==num_layers else decoder_n_hidden[i]
            dilation_size = 2 ** (num_layers-i)
            padding_size = (kernel_size-1) * dilation_size
            self.decoder_layers += [TcnResidualBlockTranspose(in_channels, out_channels, kernel_size,
                                                              stride=1, dilation=dilation_size,
                                                              padding=padding_size, dropout=dropout,
                                                              activation=activation)]


        # to register parameters in list of layers, each layer must be an object
        self.enc_layer_names = ["enc_" + str(num) for num in range(len(self.encoder_layers))]
        self.dec_layer_names = ["dec_" + str(num) for num in range(len(self.decoder_layers))]
        for name, layer in zip(self.enc_layer_names, self.encoder_layers):
            setattr(self, name, layer)
        for name, layer in zip(self.dec_layer_names, self.decoder_layers):
            setattr(self, name, layer)

        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        self.decoder = torch.nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        enc = self.encoder(out)
        dec = self.decoder(enc)
        return dec.permute(0, 2, 1), enc.permute(0, 2, 1)


class MLPnet(torch.nn.Module):
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


class TCNnet(torch.nn.Module):
    """TCN is adapted from https://github.com/locuslab/TCN"""
    def __init__(self, n_features, n_hidden='500,100', n_output=20,
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
        self.out_act = _instantiate_class("torch.nn.modules.activation", activation)
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
        return self.out_act(out + res)


class TcnResidualBlockTranspose(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, activation='ReLU'):
        super(TcnResidualBlockTranspose, self).__init__()
        self.conv1 = weight_norm(torch.nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
                                                          stride=stride, padding=padding,
                                                          dilation=dilation))
        # [Batch, input_channel, seq_len + padding]

        self.pad1 = Pad1d(padding)
        self.act1 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(torch.nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.pad2 = Pad1d(padding)
        self.act2 = _instantiate_class("torch.nn.modules.activation", activation)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(self.dropout1, self.act1, self.pad1, self.conv1,
                                       self.dropout2, self.act2, self.pad2, self.conv2)
        self.downsample = torch.nn.ConvTranspose1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = torch.nn.ReLU()
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



class Trans_encoder(torch.nn.Module):
    def __init__(self, num_inputs, feature_size=512, num_channels=[64, 128, 256, 512], num_layers=1, dropout=0.1):
        super(Trans_encoder, self).__init__()

        self.src_mask = None
        self.embedding = TokenEmbedding(c_in=num_inputs * 2, d_model=feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.layer1 = self._make_layer(inputs=num_inputs, feature_size=num_channels[0], num_layers=num_layers,
                                       dropout=dropout)
        self.layer2 = self._make_layer(inputs=num_channels[0], feature_size=num_channels[1], num_layers=num_layers,
                                       dropout=dropout)
        self.layer3 = self._make_layer(inputs=num_channels[1], feature_size=num_channels[2], num_layers=num_layers,
                                       dropout=dropout)
        # self.layer4 = self._make_layer(inputs=num_channels[2], feature_size=num_channels[3], num_layers=num_layers,
        #                                dropout=dropout)

    def _make_layer(self, inputs, feature_size, num_layers, dropout):
        embedding = TokenEmbedding(n_inputs=inputs, n_outputs=feature_size)
        pos_encoder = PositionalEncoding(feature_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return torch.nn.Sequential(embedding, pos_encoder, transformer_encoder)

    def forward_stage(self, x, stage):
        assert(stage in ['layer1', 'layer2', 'layer3', 'layer4'])
        layer = getattr(self, stage)
        x = layer(x)
        return x.permute(1, 2, 0)
    def forward(self, src, c):
        # c = c.permute(1, 0, 2)
        # src = src.permute(1, 0, 2)
        src = self.embedding(torch.cat((src, c), dim=2))
        src = src.permute(1, 0, 2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        # output = self.decoder(output)
        # return output.permute(1, 0, 2)
        # return out.permute(1, 0, 2)
        return output.permute(1, 2, 0)



    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask


class TokenEmbedding(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.conv = torch.nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs,
                                         kernel_size=3, padding=padding,
                                         padding_mode='circular')
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight,mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        x = self.conv(x)
        return x.permute(2, 0, 1)



class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.src_mask = None
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term2 = torch.exp(torch.arange(0, d_model if(d_model % 2) == 0 else d_model-1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
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
        # return **{'src': x + self.pe[:x.size(0), :], 'mask': mask}
        # return [x + self.pe[:x.size(0), :], mask]
        return x + self.pe[:x.size(0), :]



if __name__ == '__main__':
    model = TcnAE(n_features=16, n_hidden='32', n_emb=8)
    a = torch.randn(128, 100, 16)

    b, _ =  model(a)
    print(b.shape)
