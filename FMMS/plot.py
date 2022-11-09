import numpy as np
import matplotlib.pyplot as plt
import pickle

import config
import evaluation
import utils
import pandas as pd


def plot_parameter():
    data = pd.read_csv('./plot_data/parameter.csv')
    paras_name = ['Batch', 'Learning Rate', 'Embedding Size']
    paras = ['batch', 'lr', 'embedding']
    fig = plt.figure(dpi=300, figsize=(16, 4))

    for ii, para in enumerate(paras):
        ax1 = plt.subplot(1, 3, ii+1)

        if para == 'lr':
            x = data[para]
            ax1.set_xscale('log')
            ax1.set_xlim(5*10**-5, 10**-1)
        else:
            x = [1, 2, 3, 4, 5, 6]
            x_index = [str(i) for i in data[para]]
        figdata = data[para + 'Rank']
        std_table = data[para + 'Rank_std']  # 计算标准差
        # ax1.errorbar(x, figdata, yerr=std_table, xlolims=True, label='Rank')
        ax1.errorbar(x, figdata, yerr=std_table, ecolor='k', elinewidth=0.5, marker='s', fmt='#ffaa00', mfc='#ffaa00',
                     mec='k', mew=1, ms=10, alpha=1, capsize=5, capthick=3, label="Rank")
        if para == 'lr':
            pass
        else:
            _ = plt.xticks(x, x_index)  # 显示坐标字
        ax1.set_ylabel('Rank', fontsize=15)
        ax1.legend(bbox_to_anchor=(0.5,1.2), fontsize=15)
        ax1.set_xlabel(paras_name[ii], fontsize=15)
        ax1.tick_params(labelsize=15)

        # // 右侧坐标
        ax2 = ax1.twinx()
        figdata = data[para + 'TopNK']
        std_table = data[para + 'TopNK_std']  # 计算标准差
        ax2.errorbar(x, figdata, yerr=std_table, ecolor='#1240ab', elinewidth=0.5, marker='8', fmt='#6c8cd5', mfc='#6c8cd5',
                     mec='#1240ab', mew=1, ms=10, alpha=1, capsize=5, capthick=3, label="TopNK")
        ax2.set_ylabel('TopNK', fontsize=15)
        ax2.legend(bbox_to_anchor=(0.5,1.2), loc=2, fontsize=15)
        ax2.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig('./figs/%s/parameters.eps' % config.dataset)
    plt.savefig('./figs/%s/parameters.png' % config.dataset)
    plt.show()


def plot_convergence(loss_train_set, loss_test_set, ylabel='cos', path=''):
    epoch = len(loss_train_set)
    plt.plot(range(epoch), loss_train_set, label='Training data')
    plt.plot(range(epoch), loss_test_set, label='Testing data')
    plt.ylabel(ylabel)
    plt.xlabel('Number of Epochs')
    plt.title('Convergence Test')
    plt.grid()
    plt.legend()
    # plt.savefig('./figs/%s/convergence_%s.png' % (config.dataset, path))
    plt.show()


def plot_detail():
    return

def main():
    ytrain, ytest, xtrain, xtest = utils.get_data()
    ytest = ytest[:, :2000]
    data = []
    pathlist = ['result_tsfm', 'result_metaod']
    yLabel = ['tsFM', 'MetaOD']
    for ii, path in enumerate(pathlist):
        # path = config.get_para()
        results = pickle.load(open("results/%s.pkl" % path, 'rb'), encoding='iso-8859-1')
        result = results[yLabel[ii]]
        erank, rank = evaluation.ERank(result, ytest)
        # rank = rank[:20]
        data.append(rank)
        print(rank)
    # result = [[10.838943,9.124021 ,  6.635454],[2,3,4]]
    # 定义热图的横纵坐标
    # xLabel = ['A', 'B', 'C']
    # 作图阶段
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    # ax.set_xticks(range(len(xLabel)))
    # ax.set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title("This is a title")
    # show
    plt.show()


def test():
    data = pd.read_csv('./plot_data/parameter.csv')
    paras = ['batch', 'lr', 'embedding']
    para = 'batch'

    plt.xlabel(para)  # 设置x轴名称
    plt.ylabel("Rank")  # 设置y轴名称
    # plt.title("这是标题")  # 设置标题

    # x = data[para]

    plt.show()


if __name__ == '__main__':
    # loss_train_set = np.array([1, 2, 3, 4, 5])
    # loss_test_set = np.array([1, 2, 1, 2, 3])
    # plot_convergence(loss_train_set, loss_test_set)
    plot_parameter()
    # test()