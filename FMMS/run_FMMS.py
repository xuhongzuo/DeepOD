import pandas as pd

import config
import utils
from FMMS import FMMS
import torch
import torch.utils.data as Data
import numpy as np
from gene_feature import generate_meta_features

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
    def __init__(self, Fmap, Pmap, Fmapvalid=None, Pmapvalid=None, embedding_size=4, batch=4, lr=0.001, epoch=50, opt='adam', loss='cos'):
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
        self.feature_size = Fmap.shape[1]
        self.model_size = Pmap.shape[1]
        self.fmms = FMMS(feature_size=self.feature_size, model_size=self.model_size, embedding_size=embedding_size)
        self.Fmap = Fmap
        self.Pmap = Pmap
        self.Fmapvalid = Fmapvalid
        self.Pmapvalid = Pmapvalid
        self.batch = batch
        self.lr = lr
        self.epoch = epoch
        self.opt = optlsit[opt]
        self.loss = losslist[loss]
        return

    def fit(self, save=False):
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

            # valid
            if self.Fmapvalid is not None:
                Pmapvalidpred = self.fmms(torch.tensor(self.Fmapvalid))
                loss_valid = self.loss(Pmapvalidpred, self.Pmapvalid)
                loss_valid_set.append(loss_valid.item())

            print("epoch: %d" % epoch, "loss_train:", loss_train.item())
        # # 保存训练好的模型
        if save:
            torch.save(self.fmms.state_dict(), "models/fmms.pt")

    def predict(self, x=None, f=None, topn=5, load=False):
        if load:
            self.fmms.load_state_dict(torch.load("models/fmms.pt"))
        if f is None:
            f = generate_meta_features(x)[0]
        if f.shape[0] != self.feature_size:
            print('The feature size of historical dataset (%d) and target dataset (%d) is not match' % (f.shape[0], self.feature_size))
        pred = self.fmms(torch.tensor([f]))         # 预测的性能得分（仅保证相对排序准确，不保证绝对值准确）
        pred = pred.detach().numpy()[0]
        argsort_pred = np.argsort(pred)             # 对应的排名
        recommend = argsort_pred[::-1]
        print("预测性能最优的%d位模型的索引值:" % topn, recommend[:topn])
        return


if __name__ == '__main__':
    ptrain, ptest, ftrain, ftest = utils.get_data(0.1)
    ptrain = ptrain[:, :config.modelnum]
    ptest = ptest[:, :config.modelnum]
    ftrain, ptrain, fvalid, pvalid = utils.train_test_val_split(ftrain, ptrain, 0.1)
    rfmms = run_FMMS(ftrain, ptrain, fvalid, pvalid)
    rfmms.fit(save=True)
    rfmms.predict(f=ftest[0], load=True)

