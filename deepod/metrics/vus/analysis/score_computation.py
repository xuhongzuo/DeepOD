
import numpy as np
import math
import pandas as pd
from tqdm import tqdm as tqdm
import time
from sklearn.preprocessing import MinMaxScaler
import random


import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from deepod.metrics.vus.utils.slidingWindows import find_length
from deepod.metrics.vus.utils.metrics import metricor

from deepod.metrics.vus.models.distance import Fourier
from deepod.metrics.vus.models.feature import Window
from deepod.metrics.vus.models.cnn import cnn
from deepod.metrics.vus.models.AE_mlp2 import AE_MLP2
from deepod.metrics.vus.models.lstm import lstm
from deepod.metrics.vus.models.ocsvm import OCSVM
from deepod.metrics.vus.models.poly import POLY
from deepod.metrics.vus.models.pca import PCA
from deepod.metrics.vus.models.norma import NORMA
from deepod.metrics.vus.models.matrix_profile import MatrixProfile
from deepod.metrics.vus.models.lof import LOF
from deepod.metrics.vus.models.iforest import IForest

def find_section_length(label,length):
    best_i = None
    best_sum = None
    current_subseq = False
    for i in range(len(label)):
        changed = False
        if label[i] == 1:
            if current_subseq == False:
                current_subseq = True
                if best_i is None:
                    changed = True
                    best_i = i
                    best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                else:
                    if np.sum(label[max(0,i-200):min(len(label),i+9800)]) < best_sum:
                        changed = True
                        best_i = i
                        best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                    else:
                        changed = False
                if changed:
                    diff = i+9800 - len(label)

                    pos1 = max(0,i-200 - max(0,diff))
                    pos2 = min(i+9800,len(label))
        else:
            current_subseq = False
    if best_i is not None:
        return best_i-pos1,(pos1,pos2)
    else:
        return None,None

def generate_data(filepath,init_pos,max_length):
    
    df = pd.read_csv(filepath, header=None).to_numpy()
    name = filepath.split('/')[-1]
    #max_length = 30000
    data = df[init_pos:init_pos+max_length,0].astype(float)
    label = df[init_pos:init_pos+max_length,1]
    
    pos_first_anom,pos = find_section_length(label,max_length)
    
    data = df[pos[0]:pos[1],0].astype(float)
    label = df[pos[0]:pos[1],1]
    
    slidingWindow = find_length(data)
    #slidingWindow = 70
    X_data = Window(window = slidingWindow).convert(data).to_numpy()

    data_train = data[:int(0.1*len(data))]
    data_test = data

    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
    
    return pos_first_anom,slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label

def compute_score(methods,slidingWindow,data,X_data,data_train,data_test,X_train,X_test):
    
    methods_scores = {}
    for method in methods:
        start_time = time.time()
        if method == 'IForest':
            clf = IForest(n_jobs=1)
            x = X_data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

        elif method == 'LOF':
            clf = LOF(n_neighbors=20, n_jobs=1)
            x = X_data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

        elif method == 'MatrixProfile':
            clf = MatrixProfile(window = slidingWindow)
            x = data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

        elif method == 'NormA':
            clf = NORMA(pattern_length = slidingWindow, nm_size=3*slidingWindow)
            x = data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            score = np.array([score[0]]*((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

        elif method == 'PCA':
            clf = PCA()
            x = X_data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

        elif method == 'POLY':
            clf = POLY(power=3, window = slidingWindow)
            x = data
            clf.fit(x)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

        elif method == 'OCSVM':
            X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
            X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T
            clf = OCSVM(nu=0.05)
            clf.fit(X_train_, X_test_)
            score = clf.decision_scores_
            score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

        elif method == 'LSTM':
            clf = lstm(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 50, patience = 5, verbose=0)
            clf.fit(data_train, data_test)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

        elif method == 'AE':
            clf = AE_MLP2(slidingWindow = slidingWindow, epochs=100, verbose=0)
            clf.fit(data_train, data_test)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

        elif method == 'CNN':
            clf = cnn(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 100, patience = 5, verbose=0)
            clf.fit(data_train, data_test)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

        #end_time = time.time()
        #time_exec = end_time - start_time
        #print(method,"\t time: {}".format(time_exec))
        methods_scores[method] = score
        
    return methods_scores




