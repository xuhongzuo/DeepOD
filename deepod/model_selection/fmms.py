# -*- coding: utf-8 -*-
"""
Factorization Machine-based Unsupervised Model Selection Method
@Author: Ruyi Zhang & Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import torch
import torch.utils.data as Data
import numpy as np
from deepod.model_selection.gene_feature import generate_meta_features
from sklearn.model_selection import train_test_split


class FMMS:
    def __init__(self, embedding_size=4, batch=4, lr=0.001, epoch=50,
                 prt_steps=10, verbose=1, random_state=0):
        """
        Factorization Machine-based Unsupervised Model Selection Method.
        FMMS is trained by historical performance on a large suite of data collection
        and the characteristics of these datasets. Fitted FMMS can be used to
        recommend more suitable detection model on new datasets according to
        their characteristics.

        Parameters
        ----------

        embedding_size: int, optional (default=4)
            The dimension of the auxiliary vector

        batch: int, optional (default=4)
            Batch size

        lr: float, optional (default=0.001)
            Learning rate

        epoch: int, optional (default=50)
            Training epoch

        verbose: int, optional (default=1)
            Verbosity mode

        prt_steps: int, optional (default=10)
            Number of epoch intervals per printing

        random_state: int, optional (default=0)
            Random statement

        """

        self.embedding_size = embedding_size
        self.batch = batch
        self.lr = lr
        self.epoch = epoch
        self.random_state = random_state

        self.verbose = verbose
        self.prt_steps = prt_steps

        self.net = None
        self.model_size = None
        self.feature_size = None

        return

    def fit(self, Fmap, Pmap, save_path=None):
        """
        Fit model selection model.

        Parameters
        ----------

        Fmap: np.array (D*F), required
            The feature Map of the historical dataset.

        Pmap: np.array (M*D), required
            The performance of the candidate models on the historical dataset.

        save_path: str, optional (default=None)
            The location where the trained model is stored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.feature_size = Fmap.shape[1]
        self.model_size = Pmap.shape[1]

        f_train, f_valid, p_train, p_valid = train_test_split(Fmap, Pmap, test_size=0.2,
                                                              random_state=self.random_state)
        f_valid = torch.from_numpy(f_valid)
        p_valid = torch.from_numpy(p_valid)

        self.net = FMMSNet(feature_size=self.feature_size, model_size=self.model_size,
                           embedding_size=self.embedding_size)

        train_dataset = Data.TensorDataset(torch.tensor(f_train), torch.tensor(p_train))
        loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch,
            shuffle=False
        )

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_valid_set = []
        np_ctr = 1

        total_loss = 0.
        cnt = 0
        for i in range(self.epoch):
            # early stop
            if f_valid is not None:
                if i >= 2 and (loss_valid_set[-2] - loss_valid_set[-1]) / loss_valid_set[-1] <= 0.001:
                    np_ctr += 1
                else:
                    np_ctr = 1
                if np_ctr > 5:
                    break

            for step, (batch_x, batch_y) in enumerate(loader):
                optimizer.zero_grad()
                output = self.net(batch_x)
                loss_train = self.cos_loss(output, batch_y)

                # @TODO: seems not used
                l2_regularization = torch.tensor(0).float()
                for param in self.net.parameters():
                    l2_regularization += torch.norm(param, 2)

                loss_train.backward()
                optimizer.step()

                total_loss += loss_train.item()
                cnt += 1

            # validation phase
            valid_output = self.net(f_valid)
            loss_valid = self.cos_loss(valid_output, p_valid)
            loss_valid_set.append(loss_valid.item())

            if self.verbose >= 1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch {i+1:3d}, '
                      f'training loss: {total_loss / cnt:.6f}, ')

        # save trained models
        if save_path is not None:
            torch.save(self.net, save_path)

        return self

    def predict(self, x=None, f=None, topn=5, load_path=None):
        """

        Parameters
        ----------
        x: np.array, optional (default=None)
            Target dataset

        f: np.array, optional (default=None)
            Features of the target dataset, x and f should be

        topn: int, optional (default=5)
            Number of the recommended models

        load_path: str, optional (default=None)

        Returns
        -------

        """
        if load_path is not None:
            self.net = torch.load(load_path)
            self.feature_size = self.net.feature_size
            self.model_size = self.net.embedding_size

        if f is None and x is None:
            raise AttributeError('Either x or f should be fitted.')
        if f is not None and x is not None:
            raise Warning('f is deprecated by re-generate features from the target dataset x')

        if x is not None:
            f = generate_meta_features(x)[0]

        assert f.shape[1] == self.feature_size, \
            f'The feature size of historical dataset ({f.shape[0]}) ' \
            f'and target dataset ({self.feature_size}) does not match'

        pred = self.net(torch.tensor(f))
        pred = pred.detach().numpy()[0]
        ranking = np.argsort(pred)  # ranking

        recommend = ranking[::-1]
        # @TODO return the ranking of model names
        print(f"Predicted Top {topn} better models are {recommend[:topn]}")

        return

    @staticmethod
    def cos_loss(pred, real):
        # pred = pred.type(torch.DoubleTensor)
        # real = torch.tensor(real).double()
        pred = pred.double()
        real = real.double()

        mul = pred * real
        mul = torch.sum(mul, dim=1)
        length = torch.norm(pred, p=2, dim=1) * torch.norm(real, p=2, dim=1)
        loss = 1 - mul / length
        loss = sum(loss) / len(pred)
        return loss


class FMMSNet(torch.nn.Module):
    def __init__(self, feature_size, model_size, embedding_size):
        super(FMMSNet, self).__init__()
        self.feature_size = feature_size  # denote as F, the size of the feature dictionary
        self.embedding_size = embedding_size  # denote as K, the size of the feature embedding
        self.model_size = model_size  # denote as M, the size of the model list
        self.linear = torch.nn.Sequential(torch.nn.Linear(self.feature_size, self.model_size, bias=True))
        self.weight = torch.nn.Parameter(torch.rand(self.embedding_size, self.feature_size, self.model_size))

    def forward(self, x):
        # FM part
        outFM = self.linear(x.clone().detach().float())
        for i in range(self.embedding_size):
            v = self.weight[i]
            xv = torch.mm(x.clone().detach().float(), v)
            xv2 = torch.pow(xv, 2)

            z = torch.pow(x.clone().detach().float(), 2)
            P = torch.pow(v, 2)
            zp = torch.mm(z, P)

            outFM = outFM + (xv2 - zp) / 2
        out = outFM
        return out

    def show(self):
        for parameters in self.parameters():
            print(parameters)

        for name, parameters in self.named_parameters():
            print(name, ':', parameters.size())
