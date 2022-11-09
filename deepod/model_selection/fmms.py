# -*- coding: utf-8 -*-
"""
Factorization Machine-based Unsupervised Model Selection Method
@Author: Ruyi Zhang
"""

import torch
import torch.utils.data as Data
import numpy as np
from gene_feature import generate_meta_features


class FMMS:
    def __init__(self, Fmap, Pmap,
                 Fmapvalid=None, Pmapvalid=None,
                 embedding_size=4, batch=4, lr=0.001, epoch=50,
                 opt='adam', loss='cos'):
        """
        Factorization Machine-based Unsupervised Model Selection Method.
        FMMS is trained by historical performance on a large suite of data collection
        and the characteristics of these datasets. Fitted FMMS can be used to
        recommend more suitable detection model on new datasets according to
        their characteristics.

        Parameters
        ----------
        Fmap: The feature Map of the historical dataset (D*F)
        Pmap: The performance of the candidate models on the historical dataset (M*D)
        embedding_size: The dimension of the auxiliary vector
        batch: Batch size
        lr: Learning rate
        epoch: Training epoch
        opt: Optimizer
        loss: Loss function

        """
        self.feature_size = Fmap.shape[1]
        self.model_size = Pmap.shape[1]
        self.net = FMMSNet(feature_size=self.feature_size, model_size=self.model_size, embedding_size=embedding_size)
        self.Fmap = Fmap
        self.Pmap = Pmap
        self.Fmapvalid = Fmapvalid
        self.Pmapvalid = Pmapvalid
        self.batch = batch
        self.lr = lr
        self.epoch = epoch

        optim_lst = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad}

        loss_lst = {
            'rmse': self.rmse_loss, 'mse': self.mse_loss,
            'cos': self.cos_loss, 'L1': self.l1_loss,
            'sL1': self.SmoothL1_loss, 'kd': self.KLDiv_loss,
            'DCG': self.DCG_loss}

        self.opt = optim_lst[opt]
        self.loss = loss_lst[loss]
        return

    def fit(self, save=False):
        train_dataset = Data.TensorDataset(torch.tensor(self.Fmap), torch.tensor(self.Pmap))
        loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch,
            shuffle=False
        )

        optimizer = self.opt(self.net.parameters(), lr=self.lr)
        loss_valid_set = []
        np_ctr = 1      # 提升幅度小于0.1%的累计次数

        for epoch in range(self.epoch):
            if self.Fmapvalid is not None:
                if epoch >= 2 and (loss_valid_set[-2] - loss_valid_set[-1]) / loss_valid_set[-1] <= 0.001:
                    np_ctr += 1
                else:
                    np_ctr = 1
                if np_ctr > 5:
                    break

            # train
            for step, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
                # 此处省略一些训练步骤
                optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
                output = self.net(batch_x)
                # 平方差
                loss_train = self.loss(output, batch_y)
                l2_regularization = torch.tensor(0).float()
                # 加入l2正则
                for param in self.net.parameters():
                    l2_regularization += torch.norm(param, 2)
                # loss = rmse_loss + l2_regularization
                loss_train.backward()
                # loss_train.backward(torch.ones(train_params['batch'], model_size))
                optimizer.step()  # 进行更新
                print("batch loss:", step, loss_train.item())

            # valid
            if self.Fmapvalid is not None:
                Pmapvalidpred = self.net(torch.tensor(self.Fmapvalid))
                loss_valid = self.loss(Pmapvalidpred, self.Pmapvalid)
                loss_valid_set.append(loss_valid.item())

            print("epoch: %d" % epoch, "loss_train:", loss_train.item())

        # # save trained models
        if save:
            torch.save(self.net.state_dict(), "models/fmms.pt")

    def predict(self, x=None, f=None, topn=5, load=False):
        if load:
            self.net.load_state_dict(torch.load("models/fmms.pt"))
        if f is None:
            f = generate_meta_features(x)[0]

        assert  f.shape[0] == self.feature_size, \
            f'The feature size of historical dataset ({f.shape[0]}) ' \
            f'and target dataset (%{self.feature_size}) does not match'

        pred = self.net(torch.tensor([f]))
        pred = pred.detach().numpy()[0]
        ranking = np.argsort(pred) # ranking

        recommend = ranking[::-1]
        print(f"Predicted Top {topn} better models are {recommend[:topn]}")

        return

    @staticmethod
    def topn_loss(ypred, yreal, n=10):
        data_size, model_size = yreal.shape

        ypred_idx = np.array([np.argmax(ypred[i]) for i in range(data_size)])    # 每个data选出最大model对应的idx
        ypred_max = np.array([yreal[i][idx] for i, idx in enumerate(ypred_idx)])

        # 选出实际上的topn
        topn = np.ones((data_size, n))          # data*n
        for ii in range(data_size):             # 对每个数据
            best_value = list(np.sort(yreal[ii])[::-1])         # 第ii列，即第ii个数据（nan已经被填充为0，因此不会被排为最大，当所有数都是nan时，也不会报错）
            topn[ii] = best_value[:n]                           # 第ii个数据的前n个最大值

        correct = 0
        for ii, pred in enumerate(ypred_max):  # 对每个数据
            if pred in topn[ii]:
                correct += 1

        return -correct/data_size

    @staticmethod
    def rmse_loss(pred, real):
        loss_func = torch.nn.MSELoss(reduction='mean')
        mse_loss = loss_func(pred, torch.tensor(real).float())
        loss = torch.sqrt(mse_loss)
        return loss

    @staticmethod
    def mse_loss(pred, real):
        loss_func = torch.nn.MSELoss(reduction='mean')
        loss = loss_func(pred, torch.tensor(real).float())
        return loss

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

    @staticmethod
    def l1_loss(pred, real):
        L1 = torch.nn.L1Loss()
        loss = L1(pred, torch.tensor(real).float())
        return loss

    @staticmethod
    def SmoothL1_loss(pred, real):
        sl1 = torch.nn.SmoothL1Loss()
        loss = sl1(pred, torch.tensor(real).float())
        return loss

    @staticmethod
    def KLDiv_loss(pred, real):
        kd = torch.nn.KLDivLoss()
        loss = kd(pred, torch.tensor(real).float())
        return loss

    @staticmethod
    def DCG_loss(pred, real):
        data_size, model_size = real.shape
        treal = torch.tensor(real).double()
        DCG = torch.zeros(1).type(torch.DoubleTensor)
        for ii in range(data_size):
            for jj in range(model_size):
                part = sum(torch.sigmoid(pred[ii] - pred[ii][jj])).type(torch.DoubleTensor)
                DCG = DCG + (torch.pow(10, treal[ii][jj])-1)/torch.log2(1+part)
        return -DCG/data_size




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
        for parameters in self.parameters():  # 打印出参数矩阵及值
            print(parameters)

        for name, parameters in self.named_parameters():  # 打印出每一层的参数的大小
            print(name, ':', parameters.size())


