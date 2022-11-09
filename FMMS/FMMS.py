import torch
from torch import nn
import numpy as np


class FMMS(nn.Module):
    def __init__(self, feature_size, model_size, embedding_size):
        super(FMMS, self).__init__()
        self.feature_size = feature_size        # denote as F, size of the feature dictionary
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding
        self.model_size = model_size            # denote as M, size of the model list
        self.linear = nn.Sequential(nn.Linear(self.feature_size, self.model_size, bias=True))
        self.weight = nn.Parameter(torch.rand(self.embedding_size, self.feature_size, self.model_size))

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