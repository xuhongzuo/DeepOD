# -*- coding: utf-8 -*-
"""
Factorization Machine-based Unsupervised Model Selection Method
@Author: Ruyi Zhang
"""

import torch
import torch.utils.data as Data
import numpy as np
from deepod.model_selection.gene_feature import generate_meta_features
from sklearn.model_selection import train_test_split


class FMMS:
    def __init__(self,
                 embedding_size=4, batch=4, lr=0.001, epoch=50, random_state=0):
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
            
        random_state: int, optional (default=0)
            Random statement

        """
        self.net = None
        self.model_size = None
        self.feature_size = None
        self.embedding_size = embedding_size
        self.batch = batch
        self.lr = lr
        self.epoch = epoch
        self.random_state = random_state
        return

    def fit(self, Fmap, Pmap, save_path=None):
        """
        Parameters
        ----------

        Fmap: np.array (D*F), required
            The feature Map of the historical dataset.

        Pmap: np.array (M*D), required
            The performance of the candidate models on the historical dataset.

        save_path: str, optional (default=None)
            The location where the trained model is stored.

        -------

        """
        self.feature_size = Fmap.shape[1]
        self.model_size = Pmap.shape[1]

        f_train, f_valid, p_train, p_valid = train_test_split(Fmap, Pmap, test_size=0.2, random_state=self.random_state)

        self.net = FMMSNet(feature_size=self.feature_size, model_size=self.model_size, embedding_size=self.embedding_size)

        train_dataset = Data.TensorDataset(torch.tensor(f_train), torch.tensor(p_train))
        loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch,
            shuffle=False
        )

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_valid_set = []
        np_ctr = 1

        for epoch in range(self.epoch):
            # early stop
            if f_valid is not None:
                if epoch >= 2 and (loss_valid_set[-2] - loss_valid_set[-1]) / loss_valid_set[-1] <= 0.001:
                    np_ctr += 1
                else:
                    np_ctr = 1
                if np_ctr > 5:
                    break

            # train
            for step, (batch_x, batch_y) in enumerate(loader):
                optimizer.zero_grad()
                output = self.net(batch_x)
                loss_train = self.cos_loss(output, batch_y)
                l2_regularization = torch.tensor(0).float()
                for param in self.net.parameters():
                    l2_regularization += torch.norm(param, 2)


                loss_train.backward()

                optimizer.step()
                # print("batch loss:", step, loss_train.item())

            # valid
            Pmapvalidpred = self.net(torch.tensor(f_valid))
            loss_valid = self.cos_loss(Pmapvalidpred, p_valid)
            loss_valid_set.append(loss_valid.item())

            print("epoch: %d" % epoch, "loss_train:", loss_train.item())

        # # save trained models
        if save_path is not None:

            torch.save(self.net, save_path)

    def predict(self, x=None, f=None, topn=5, load_path=None):
        if load_path is not None:
            self.net = torch.load(load_path)
            self.feature_size = self.net.feature_size
            self.model_size = self.net.embedding_size
        if f is None:
            f = generate_meta_features(x)[0]

        assert f.shape[0] == self.feature_size, \
            f'The feature size of historical dataset ({f.shape[0]}) ' \
            f'and target dataset (%{self.feature_size}) does not match'

        pred = self.net(torch.tensor([f]))
        pred = pred.detach().numpy()[0]
        ranking = np.argsort(pred) # ranking

        recommend = ranking[::-1]
        # @TODO return the ranking of model names
        print(f"Predicted Top {topn} better models are {recommend[:topn]}")

        return

    @staticmethod
    def cos_loss(pred, real):
        pred = pred.type(torch.DoubleTensor)
        real = torch.tensor(real).double()
        mul = pred * real
        mul = torch.sum(mul, dim=1)
        length = torch.norm(pred, p=2, dim=1) * torch.norm(real, p=2, dim=1)
        loss = 1 - mul/length
        loss = sum(loss)/len(pred)
        return loss


class FMMSNet(torch.nn.Module):
    def __init__(self, feature_size, model_size, embedding_size):
        super(FMMSNet, self).__init__()
        self.feature_size = feature_size        # denote as F, size of the feature dictionary
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding
        self.model_size = model_size            # denote as M, size of the model list
        self.linear = torch.nn.Sequential(torch.nn.Linear(self.feature_size, self.model_size, bias=True))
        self.weight = torch.nn.Parameter(torch.rand(self.embedding_size, self.feature_size, self.model_size))

    def forward(self, x):         # 1*M
        # FM part
        outFM = self.linear(x.clone().detach().float())
        for i in range(self.embedding_size):
            v = self.weight[i]
            xv = torch.mm(x.clone().detach().float(), v)
            xv2 = torch.pow(xv, 2)

            z = torch.pow(x.clone().detach().float(), 2)
            P = torch.pow(v, 2)
            zp = torch.mm(z, P)

            outFM = outFM + (xv2 - zp)/2
        out = outFM
        return out

    def show(self):
        for parameters in self.parameters():
            print(parameters)

        for name, parameters in self.named_parameters():
            print(name, ':', parameters.size())

