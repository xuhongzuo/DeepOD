import pandas as pd

import config
import evaluation
import utils
from FMMS import FMMS
import torch
import torch.utils.data as Data
import pickle


def Train(params, train_params, xtrain, xvalid, xtest, ytrain, yvalid, ytest, txt=''):
    path = config.get_para(train_params, params)
    train_dataset = Data.TensorDataset(torch.tensor(xtrain), torch.tensor(ytrain))
    # 训练集分批处理
    loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=train_params['batch'],  # 最新批数据
        shuffle=False  # 是否随机打乱数据
    )
    fmms = FMMS(**params)
    # 训练网络
    optimizer = train_params['opt'](fmms.parameters(), lr=train_params['lr'])
    loss_train_set = []
    loss_valid_set = []
    loss_test_set = []
    ERank_train = []
    ETopn_train = []
    ERank_valid = []
    ETopn_valid = []
    ERank_test = []
    ETopn_test = []
    np_ctr = 1
    for epoch in range(train_params['epoch']):  # 对数据集进行训练
        # improvement is smaller than 1 perc
        if config.dataset == 'pmf':
            if epoch >= 2 and (loss_valid_set[-2] - loss_valid_set[-1])/loss_valid_set[-1] <= 0.001:
                np_ctr += 1
            else:
                np_ctr = 1
            if np_ctr > 5:
                break

        for step, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
            # 此处省略一些训练步骤
            optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
            output = fmms(batch_x)
            # 平方差
            loss_train = train_params['loss'](output, batch_y)
            l2_regularization = torch.tensor(0).float()
            # 加入l2正则
            for param in fmms.parameters():
                l2_regularization += torch.norm(param, 2)
            # loss = rmse_loss + l2_regularization
            loss_train.backward()
            # loss_train.backward(torch.ones(train_params['batch'], model_size))
            optimizer.step()  # 进行更新
            print("batch loss:", step, loss_train.item())
        # 将每一次训练的数据进行存储，然后用于绘制曲线

        # train
        ytrainpred = fmms(torch.tensor(xtrain))
        loss_train = train_params['loss'](ytrainpred, ytrain)
        ytrainpred = ytrainpred.detach().numpy()
        train_rank, _ = evaluation.ERank(ytrainpred, ytrain)
        train_topn, _ = evaluation.ETopnk(ytrainpred, ytrain, int(config.modelnum*0.05), 5)
        loss_train_set.append(loss_train.item())
        ETopn_train.append(train_rank)
        ERank_train.append(train_topn)

        # valid
        yvalidpred = fmms(torch.tensor(xvalid))
        loss_valid = train_params['loss'](yvalidpred, yvalid)
        yvalidpred = yvalidpred.detach().numpy()
        valid_rank, _ = evaluation.ERank(yvalidpred, yvalid)
        valid_topn, _ = evaluation.ETopnk(yvalidpred, yvalid, int(config.modelnum*0.05), 5)
        loss_valid_set.append(loss_valid.item())
        ETopn_valid.append(valid_rank)
        ERank_valid.append(valid_topn)

        # test
        ytestpred = fmms(torch.tensor(xtest))
        loss_test = train_params['loss'](ytestpred, ytest)
        ytestpred = ytestpred.detach().numpy()
        test_rank, _ = evaluation.ERank(ytestpred, ytest)
        test_topn, _ = evaluation.ETopnk(ytestpred, ytest, int(config.modelnum*0.05), 5)
        loss_test_set.append(loss_test.item())
        ETopn_test.append(test_topn)
        ERank_test.append(test_rank)

        print("epoch: %d" % epoch, "loss_train:", loss_train.item(), "loss_test", loss_test.item())
    # print(y_train[0:5],"  ",output[0:5])
    # plot.plot_convergence(loss_train_set, loss_test_set, path='fmms_'+path)
    # # 保存训练好的模型
    torch.save(fmms.state_dict(), "models/%s/FMMS_%s_%s.pt" % (config.dataset, path, txt))
    log = {
        'loss_train_set': loss_train_set,
        'loss_valid_set': loss_valid_set,
        'loss_test_set': loss_test_set,
        'ETopn_train': ETopn_train,
        'ERank_train': ERank_train,
        'ETopn_valid': ETopn_valid,
        'ERank_valid': ERank_valid,
        'ETopn_test': ETopn_test,
        'ERank_test': ERank_test
    }
    log_df = pd.DataFrame(log)
    return log_df


def Test(params, train_params, x_test, y_test, txt=''):
    # test
    path = config.get_para(train_params, params)
    print(path)
    fmms = FMMS(**params)
    fmms.load_state_dict(torch.load("models/%s/FMMS%s_%s.pt" % (config.dataset, path, txt)))
    pred = fmms(torch.tensor(x_test))
    loss = train_params['loss'](pred, y_test)
    pred = pred.detach().numpy()
    print("test_loss", loss)
    result = {'fmms': pred}
    pickle.dump(result, open("results/%s/result_fmms_%s_%s.pkl" % (config.dataset, path, txt), 'wb'))

    return pred


def main(txt=''):
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)
    ytrain = ytrain[:, :config.modelnum]
    ytest = ytest[:, :config.modelnum]

    optlsit = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adagrad': torch.optim.Adagrad}
    losslist = {'rmse': utils.rmse_loss, 'mse': utils.mse_loss,
                'cos': utils.cos_loss, 'L1': utils.l1_loss,
                'sL1': utils.SmoothL1_loss, 'kd': utils.KLDiv_loss,
                'DCG': utils.DCG_loss
                }
    params = {
        'embedding_size': 4,
        'feature_size': xtrain.shape[1],
        'model_size': ytrain.shape[1],
        'layer_size': 3,
        'hiddensize': 64
    }
    opt = 'adam'
    l = 'cos'
    train_params = {
        'batch': 4,
        'lr': 0.001,
        'epoch': 50,
        'opt': optlsit[opt],
        'optname': opt,
        'loss': losslist[l],
        'lossname': l,
    }
    # ytrain = utils.scaleY(ytrain)
    # ytest = utils.scaleY(ytest)
    xtrain, ytrain, xvalid, yvalid = utils.train_test_val_split(xtrain, ytrain, rate)
    log = Train(params, train_params, xtrain, xvalid, xtest, ytrain, yvalid, ytest)
    # log.to_csv('./experiments/convergence/fmms%s_%s.csv'%(config.get_para(train_params, params),txt))  # fmms在pmf数据集上参数为path时第r次实验的中间结果
    ypred = Test(params, train_params, xtest, ytest)
    evaluation.ERank(ypred, ytest, 'fmms')                                          # 所推荐的模型的平均排名
    evaluation.ETopnk(ypred, ytest, int(config.modelnum*0.05), 5, 'fmms')           # 所推荐的模型中，排在前n个的里有几个


if __name__ == '__main__':
    main()