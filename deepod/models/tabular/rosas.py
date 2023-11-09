import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet


class RoSAS(BaseDeepAD):
    """
    RoSAS: Deep semi-supervised anomaly detection with contamination-resilient
    continuous supervision (IP&M'23)
    """
    def __init__(self, epochs=100, batch_size=128, lr=0.005,
                 rep_dim=32, hidden_dims='32', act='LeakyReLU', bias=False,
                 margin=5., alpha=0.5, T=2, k=2,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(RoSAS, self).__init__(
            data_type='tabular', model_name='RoSAS', epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        # network parameters
        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.k = k
        self.margin = margin
        self.alpha = alpha
        self.T = T
        return

    def training_prepare(self, X, y):
        train_loader = _RoSASLoader(X, y, batch_size=self.batch_size)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'rep_dim': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        net = _RoSASNet(**network_params).to(self.device)

        criterion = _RoSASLoss(
            margin=self.margin, alpha=self.alpha,
            T=self.T, k=self.k,
            reduction='mean'
        )

        return train_loader, net, criterion

    def training_forward(self, batch_x, net, criterion):
        anchor, pos, neg = batch_x[:, 0], batch_x[:, 1], batch_x[:, 2]

        anchor = anchor.float().to(self.device)
        pos = pos.float().to(self.device)
        neg = neg.float().to(self.device)

        anchor_emb, anchor_s = net(anchor)
        pos_emb, pos_s = net(pos)
        neg_emb, neg_s = net(neg)
        embs = [anchor_emb, pos_emb, neg_emb]

        if self.k == 2:
            x_i = torch.cat((anchor, pos, neg), 0)
            target_i = torch.cat(
                (torch.ones_like(anchor_s) * -1, torch.ones_like(anchor_s) * -1, torch.ones_like(neg_s)), 0)

            indices_j = torch.randperm(x_i.size(0)).to(self.device)
            x_j = x_i[indices_j]
            target_j = target_i[indices_j]

            Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
            lambdas = Beta.sample(target_i.flatten().shape).to(self.device)[:, 1]

            x_tilde = x_i * lambdas.view(lambdas.size(0), 1) + x_j * (1 - lambdas.view(lambdas.size(0), 1))
            _, score_tilde = net(x_tilde)
            _, score_xi = net(x_i)
            _, score_xj = net(x_j)

            score_mix = score_xi * lambdas.view(lambdas.size(0), 1) + score_xj * (1 - lambdas.view(lambdas.size(0), 1))
            y_tilde = target_i * lambdas.view(lambdas.size(0), 1) + target_j * (1 - lambdas.view(lambdas.size(0), 1))

        else:
            # # # ----------------------- n-samples mixup --------------------------- #
            x_i = torch.cat((anchor, pos, neg), 0)
            target_i = torch.cat((torch.ones_like(anchor_s)*-1, torch.ones_like(anchor_s)*-1, torch.ones_like(neg_s)), 0)
            _, score_xi = net(x_i)

            x_dup = [x_i]
            target_dup = [target_i]
            score_dup = [score_xi]
            for k in range(1, self.k):
                indices_j = torch.randperm(x_i.size(0)).to(self.device)
                x_j = x_i[indices_j]
                target_j = target_i[indices_j]
                _, score_xj = net(x_j)

                x_dup.append(x_j)
                target_dup.append(target_j)
                score_dup.append(score_xj)

            Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
            lambdas_dup = Beta.sample((target_i.flatten().shape[0], self.k)).to(self.device)[:, :, 1]

            s = torch.sum(lambdas_dup, 1).unsqueeze(0).T.repeat(1, self.k)
            lambdas_dup = lambdas_dup / s

            x_tilde = lambdas_dup[:, 0].unsqueeze(0).T * x_i
            y_tilde = lambdas_dup[:, 0].unsqueeze(0).T * target_i
            score_mix = lambdas_dup[:, 0].unsqueeze(0).T * score_xi
            for k in range(1, self.k):
                x_tilde += lambdas_dup[:, k].unsqueeze(0).T * x_dup[k]
                y_tilde += lambdas_dup[:, k].unsqueeze(0).T * target_dup[k]
                score_mix += lambdas_dup[:, k].unsqueeze(0).T * score_dup[k]

            _, score_tilde = net(x_tilde)

        loss, loss1, loss2, loss_out, loss_consistency = self.criterion(
            embs, score_tilde, score_mix, y_tilde
        )
        return loss

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, shuffle=False)
        return test_loader

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_z, batch_score = net(batch_x)
        batch_score = batch_score.reshape([batch_x.shape[0], ])
        return batch_z, batch_score


class _RoSASLoader:
    def __init__(self, x, y, batch_size=256, steps_per_epoch=None):
        self.x = x
        self.y = y

        self.anom_idx = np.where(y==1)[0]
        self.norm_idx = np.where(y==0)[0]
        self.unlabeled_idx = np.where(y==0)[0]

        self.batch_size = batch_size

        self.counter = 0

        self.steps_per_epoch = steps_per_epoch if steps_per_epoch is not None \
            else int(len(x) / self.batch_size)

        return

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        self.counter += 1

        batch_x = self.batch_generation()
        batch_x = torch.from_numpy(batch_x)

        if self.counter > self.steps_per_epoch:
            raise StopIteration
        return batch_x

    def batch_generation(self):
        this_anchor_idx = np.random.choice(self.norm_idx, self.batch_size, replace=False)
        this_pos_idx = np.random.choice(self.unlabeled_idx, self.batch_size, replace=False)
        this_anom_idx = np.random.choice(self.anom_idx, self.batch_size)

        batch_x = np.array([[self.x[a], self.x[p], self.x[n]]
                            for a, p, n in zip(this_anchor_idx, this_pos_idx, this_anom_idx)])
        # batch_y = np.array([[self.y[a], self.y[p], self.y[n]]
        #                     for a, p, n in zip(this_anchor_idx, this_pos_idx, this_anom_idx)])

        return batch_x


class _RoSASLoss(torch.nn.Module):
    def __init__(self, margin=1., alpha=1., T=2, k=2, reduction='mean'):
        super(_RoSASLoss, self).__init__()
        self.loss_tri = torch.nn.TripletMarginLoss(margin=margin, reduction=reduction)
        self.loss_reg = torch.nn.SmoothL1Loss(reduction=reduction)

        self.T = T
        self.alpha = alpha
        self.k = k
        self.reduction = reduction
        return

    def forward(self, embs, score_tilde, score_mix, y_tilde, pre_emb_loss=None, pre_score_loss=None):
        anchor_emb, pos_emb, neg_emb = embs
        loss_emb = self.loss_tri(anchor_emb, pos_emb, neg_emb)
        # loss_emb_mean = torch.mean(loss_emb)

        loss_out = self.loss_reg(score_tilde, y_tilde)
        loss_consistency = self.loss_reg(score_tilde, score_mix)
        loss_score = loss_out + loss_consistency

        if self.reduction == 'mean' and pre_emb_loss is not None:
            # # adaptive weighting
            k1 = torch.exp((loss_emb / pre_emb_loss) / self.T) if pre_emb_loss != 0 else 0
            k2 = torch.exp((loss_score / pre_score_loss) / self.T) if pre_score_loss != 0 else 0
            loss = (k1 / (k1 + k2)) * loss_emb + (k2 / (k1 + k2)) * loss_score
        else:
            loss = 0.5 * loss_emb + 0.5 * loss_score

        return loss, loss_emb, loss_score, loss_out, loss_consistency


class _RoSASNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden='100,50', rep_dim=64,
                 activation='LeakyReLU', bias=False):
        super(_RoSASNet, self).__init__()

        network_params = {
            'n_features': n_features,
            'n_hidden': n_hidden,
            'n_output': rep_dim,
            'activation': activation,
            'bias': bias
        }
        self.enc_net = MLPnet(**network_params)

        self.hidden_layer2 = torch.nn.Linear(rep_dim, int(rep_dim/2), bias=bias)
        self.out_layer = torch.nn.Linear(int(rep_dim/2), 1)

    def forward(self, x):
        emb_x = self.enc_net(x)

        s = F.leaky_relu(self.hidden_layer2(emb_x))
        s = torch.tanh(self.out_layer(s))

        return emb_x, s
