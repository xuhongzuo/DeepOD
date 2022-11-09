import pandas as pd

import config
import evaluation
import utils
from FMMS import FMMS
import torch
import torch.utils.data as Data
import pickle

optlsit = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad}

losslist = {
    'rmse': utils.rmse_loss, 'mse': utils.mse_loss,
    'cos': utils.cos_loss, 'L1': utils.l1_loss,
    'sL1': utils.SmoothL1_loss, 'kd': utils.KLDiv_loss,
    'DCG': utils.DCG_loss}


class run_FMMS():
    def __init__(self, Fmap, Pmap, embedding_size=4, batch=4, lr=0.001, epoch=50, opt='adam', loss='cos'):
        """
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
        self.fmms = FMMS(feature_size=Fmap.shape[1], model_size=Pmap.shape[1], embedding_size=embedding_size)
        self.Fmap = Fmap
        self.Pmap = Pmap
        self.batch = batch
        self.lr = lr
        self.epoch = epoch
        self.opt = optlsit[opt]
        self.loss = losslist[loss]
        return

    def fit(self):
        train_dataset = Data.TensorDataset(torch.tensor(self.Fmap), torch.tensor(self.Pmap))
        loader = Data.DataLoader(
            dataset=train_dataset,  # torch TensorDataset format
            batch_size=self.batch,  # 最新批数据
            shuffle=False  # 是否随机打乱数据
        )
        optimizer = self.opt(self.fmms.parameters(), lr=self.lr)
        loss_valid_set = []
        np_ctr = 1      # 提升幅度小于0.1%的累计次数
        for epoch in range(self.epoch):  # 对数据集进行训练
            # 早停机制
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
                output = self.fmms(batch_x)
                # 平方差
                loss_train = self.loss(output, batch_y)
                l2_regularization = torch.tensor(0).float()
                # 加入l2正则
                for param in self.fmms.parameters():
                    l2_regularization += torch.norm(param, 2)
                # loss = rmse_loss + l2_regularization
                loss_train.backward()
                # loss_train.backward(torch.ones(train_params['batch'], model_size))
                optimizer.step()  # 进行更新
                print("batch loss:", step, loss_train.item())

            print("epoch: %d" % epoch, "loss_train:", loss_train.item())
        # # 保存训练好的模型
        # torch.save(self.fmms.state_dict(), "models/%s/FMMS_%s_%s.pt" % (config.dataset, path, txt))

    def predict(self, Feature):
        pred = self.fmms(torch.tensor([Feature]))
        pred = pred.detach().numpy()
        return pred