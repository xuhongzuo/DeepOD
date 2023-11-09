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

    """
    def __init__(self, seq_len=100, stride=1, lr=0.0001, epochs=10, batch_size=32,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 pred_len=0, e_layers=2, d_model=64, d_ff=64, dropout=0.1, top_k=5, num_kernels=6,
                 verbose=2, random_state=42):
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
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """
    def __init__(self, seq_len, pred_len, enc_in, c_out,
                 e_layers, d_model, d_ff, dropout, top_k, num_kernels):
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
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
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
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
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
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]