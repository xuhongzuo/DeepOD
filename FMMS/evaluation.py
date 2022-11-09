import numpy as np


def ETopnk(ypred, yreal, n, k, txt=''):
    data_size, model_size = yreal.shape

    ypred_idx = np.array([np.argsort(ypred[i])[-k:][::-1] for i in range(data_size)])    # 排名前n的
    ypred_max = np.array([np.max(yreal[i][idx]) for i, idx in enumerate(ypred_idx)])

    # 选出实际上的topn
    topn = np.ones((data_size, n))          # data*n
    for ii in range(data_size):             # 对每个数据
        best_value = list(np.sort(yreal[ii])[::-1])         # 第ii列，即第ii个数据（nan已经被填充为0，因此不会被排为最大，当所有数都是nan时，也不会报错）
        topn[ii] = best_value[:n]                           # 第ii个数据的前n个最大值

    correct = 0
    for ii, pred in enumerate(ypred_max):  # 对每个数据，前k个预测里是否有在实际前n个里的
        if pred in topn[ii]:
            correct += 1

    print("Topnk_%s: " %txt + str(correct) + '/' + str(data_size))
    return correct/data_size, correct


def ETopn(ypred, yreal, n, txt=''):
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

    print("Topn_%s: " % txt + str(correct) + '/' + str(data_size))
    return correct


def ERank(ypred, yreal, txt=''):
    data_size, model_size = yreal.shape
    ypred_idx = np.array([np.argmax(ypred[i]) for i in range(data_size)])    # 每个data选出最大model对应的idx
    ypred_max = np.array([yreal[i][idx] for i, idx in enumerate(ypred_idx)])

    rank_yreal = np.zeros([data_size, model_size])
    # 对原始数据进行排名，计算排名
    for ii in range(data_size):  # 对每个数据
        rank_yreal[ii] = list(np.sort(yreal[ii])[::-1])  # 排序后再写回（降序）

    # 每个数据推荐的模型的平均排名
    rank = np.zeros(data_size)
    for ii, pred in enumerate(ypred_max):  # 对每个数据
        rank[ii] = np.where(rank_yreal[ii] == pred)[0][0]  # 第一个（最高排名）
    print("AvgRank_%s: " % txt + str(sum(rank)/data_size/model_size))
    return sum(rank)/data_size/model_size, rank


def ERankk(ypred, yreal, k, txt=''):
    data_size, model_size = yreal.shape

    ypred_idx = np.array([np.argsort(ypred[i])[-k:][::-1] for i in range(data_size)])    # 排名前n的
    ypred_max = np.array([np.max(yreal[i][idx]) for i, idx in enumerate(ypred_idx)])

    rank_yreal = np.zeros([data_size, model_size])
    # 对原始数据进行排名，计算排名
    for ii in range(data_size):  # 对每个数据
        rank_yreal[ii] = list(np.sort(yreal[ii])[::-1])  # 排序后再写回（降序）

    # 每个数据推荐的模型的平均排名
    rank = np.zeros(data_size)
    for ii, pred in enumerate(ypred_max):  # 对每个数据
        rank[ii] = np.where(rank_yreal[ii] == pred)[0][0]  # 第一个（最高排名）
    print("AvgRankk_%s: " % txt + str(sum(rank)/data_size/model_size))
    return sum(rank)/data_size/model_size


if __name__ == '__main__':
    modelnum = 2000
    import utils
    import pickle
    import config
    ytrain, ytest, xtrain, xtest = utils.get_data()
    ytest = ytest[:, :2000]
    ytrain = ytrain[:, :2000]
    results = pickle.load(open('results/%s/result_compare.pkl' % config.dataset, 'rb'), encoding='iso-8859-1')
    # print(results)
    ETopnk(results['feature'], ytest, int(modelnum*0.05), 1, 'feature')
    ERank(results['feature'], ytest, 'feature')
    ETopnk(results['random1x'], ytest, int(modelnum*0.05), 1, 'random1x')
    ERank(results['random1x'], ytest, 'random1x')
    ETopnk(results['random2x'], ytest, int(modelnum*0.05), 1, 'random2x')
    ERank(results['random2x'], ytest, 'random2x')
    ETopnk(results['random4x'], ytest, int(modelnum*0.05), 1, 'random4x')
    ERank(results['random4x'], ytest, 'random4x')
    ETopnk(results['random16x'], ytest, int(modelnum*0.05), 1, 'random16x')
    ERank(results['random16x'], ytest, 'random16x')
