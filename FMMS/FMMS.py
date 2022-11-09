import torch
from torch import nn
import numpy as np


class TSFM(nn.Module):
    def __init__(self, feature_size, embedding_size, model_size, layer_size, hiddensize, FM, DNN):
        super(TSFM, self).__init__()
        self.feature_size = feature_size        # denote as N, size of the feature dictionary
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding
        self.model_size = model_size            # denote as M, size of the model list
        self.layer_size = layer_size
        self.linear = nn.Sequential(nn.Linear(self.feature_size, self.model_size, bias=True))
        self.weight = nn.Parameter(torch.rand(self.embedding_size, self.feature_size, self.model_size))
        self.hiddensize = hiddensize
        self.deeplayer = self.init_layer()
        self.FM = FM
        self.DNN = DNN

    def init_layer(self):
        layers = dict()
        layers['Lin'] = nn.Linear(self.feature_size, self.hiddensize, bias=True)
        for ii in range(self.embedding_size):
            layers['L%s' % ii] = nn.Linear(self.hiddensize, self.hiddensize, bias=True)
        layers['Lout'] = nn.Linear(self.hiddensize, self.model_size, bias=True)
        return layers

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

        # DNN part
        # outDNN = self.deeplayer['Lin'](x.clone().detach().float())
        # for i in range(self.layer_size):
        #     outDNN = self.deeplayer['L%s' % i](outDNN)
        # outDNN = self.deeplayer['Lout'](outDNN)

        # if self.FM and self.DNN:
        #     out = outFM + outDNN
        # elif self.FM:
        #     out = outFM
        # else:
        #     out = outDNN
        out = outFM
        return out

    def show(self):
        for parameters in self.parameters():  # 打印出参数矩阵及值
            print(parameters)

        for name, parameters in self.named_parameters():  # 打印出每一层的参数的大小
            print(name, ':', parameters.size())