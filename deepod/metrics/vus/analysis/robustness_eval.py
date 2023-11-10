import numpy as np
from tqdm import tqdm as tqdm
import random


import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from deepod.metrics.vus.utils.metrics import metricor



def generate_new_label(label,lag):
    if lag < 0:
        return np.array(list(label[-lag:]) + [0]*(-lag))
    elif lag > 0:
        return np.array([0]*lag + list(label[:-lag]))
    elif lag == 0:
        return label

def compute_anomaly_acc_lag(methods_scores,label,slidingWindow,methods_keys):
    
    lag_range = list(range(-slidingWindow//4,slidingWindow//4,5))
    methods_acc = {}
    for i,methods_score in enumerate(tqdm(methods_keys)):
        dict_acc = {
            'R_AUC_ROC':      [],
            'AUC_ROC':        [],
            'R_AUC_PR':       [],
            'AUC_PR':         [],
            'VUS_ROC':        [],
            'VUS_PR':         [],
            'Precision':      [],
            'Recall':         [],
            'F':              [],
            'ExistenceReward':[],
            'OverlapReward':  [],
            'Precision@k':    [],
            'Rprecision':     [],
            'Rrecall':        [],
            'RF':             []}
        
        for lag in tqdm(lag_range):
            new_label = generate_new_label(label,lag)
            
            grader = metricor()  

            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=methods_scores[methods_score], window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, methods_scores[methods_score], plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, methods_scores[methods_score])  
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,methods_scores[methods_score],2*slidingWindow)
            L1 = [ elem for elem in L]

            dict_acc['R_AUC_ROC']      +=[R_AUC]
            dict_acc['AUC_ROC']        +=[L1[0]]
            dict_acc['R_AUC_PR']       +=[R_AP]
            dict_acc['AUC_PR']         +=[AP]
            dict_acc['VUS_ROC']        +=[avg_auc_3d]
            dict_acc['VUS_PR']         +=[avg_ap_3d]
            dict_acc['Precision']      +=[L1[1]]
            dict_acc['Recall']         +=[L1[2]]
            dict_acc['F']              +=[L1[3]]
            dict_acc['ExistenceReward']+=[L1[5]]
            dict_acc['OverlapReward']  +=[L1[6]]
            dict_acc['Precision@k']    +=[L1[9]]
            dict_acc['Rprecision']     +=[L1[7]]
            dict_acc['Rrecall']        +=[L1[4]]
            dict_acc['RF']             +=[L1[8]]

        methods_acc[methods_score] = dict_acc
    return methods_acc


def compute_anomaly_acc_percentage(methods_scores,label,slidingWindow,methods_keys,pos_first_anom):
    
    
    list_pos = []
    step_a = max(0,(len(label) - pos_first_anom-200))//20
    step_b = max(0,pos_first_anom-200)//20
    pos_a = min(len(label),pos_first_anom + 200)
    pos_b = max(0,pos_first_anom - 200)
    list_pos.append((pos_b,pos_a))
    for pos_iter in range(20):
        pos_a = min(len(label),pos_a + step_a)
        pos_b = max(0,pos_b - step_b)
        list_pos.append((pos_b,pos_a))
    methods_acc = {}
    print(list_pos)
    for i,methods_score in enumerate(tqdm(methods_keys)):
        dict_acc = {
            'R_AUC_ROC':      [],
            'AUC_ROC':        [],
            'R_AUC_PR':       [],
            'AUC_PR':         [],
            'VUS_ROC':        [],
            'VUS_PR':         [],
            'Precision':      [],
            'Recall':         [],
            'F':              [],
            'ExistenceReward':[],
            'OverlapReward':  [],
            'Precision@k':    [],
            'Rprecision':     [],
            'Rrecall':        [],
            'RF':             []}
        
        for end_pos in tqdm(list_pos):
            new_label = label[end_pos[0]:end_pos[1]]
            new_score = np.array(methods_scores[methods_score])[end_pos[0]:end_pos[1]]
            grader = metricor()  

            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=new_score, window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, new_score, plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, new_score)  
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,new_score,2*slidingWindow)
            L1 = [ elem for elem in L]

            dict_acc['R_AUC_ROC']      +=[R_AUC]
            dict_acc['AUC_ROC']        +=[L1[0]]
            dict_acc['R_AUC_PR']       +=[R_AP]
            dict_acc['AUC_PR']         +=[AP]
            dict_acc['VUS_ROC']        +=[avg_auc_3d]
            dict_acc['VUS_PR']         +=[avg_ap_3d]
            dict_acc['Precision']      +=[L1[1]]
            dict_acc['Recall']         +=[L1[2]]
            dict_acc['F']              +=[L1[3]]
            dict_acc['ExistenceReward']+=[L1[5]]
            dict_acc['OverlapReward']  +=[L1[6]]
            dict_acc['Precision@k']    +=[L1[9]]
            dict_acc['Rprecision']     +=[L1[7]]
            dict_acc['Rrecall']        +=[L1[4]]
            dict_acc['RF']             +=[L1[8]]

        methods_acc[methods_score] = dict_acc
    return methods_acc

def compute_anomaly_acc_noise(methods_scores,label,slidingWindow,methods_keys):
    
    lag_range = list(range(-slidingWindow//2,slidingWindow//2,10))
    methods_acc = {}
    for i,methods_score in enumerate(tqdm(methods_keys)):
        dict_acc = {
            'R_AUC_ROC':      [],
            'AUC_ROC':        [],
            'R_AUC_PR':       [],
            'AUC_PR':         [],
            'VUS_ROC':        [],
            'VUS_PR':         [],
            'Precision':      [],
            'Recall':         [],
            'F':              [],
            'ExistenceReward':[],
            'OverlapReward':  [],
            'Precision@k':    [],
            'Rprecision':     [],
            'Rrecall':        [],
            'RF':             []}
        
        for lag in tqdm(lag_range):
            new_label = label
            
            grader = metricor()  

            noise = np.random.normal(-0.1,0.1,len(methods_scores[methods_score]))
            
            new_score = np.array(methods_scores[methods_score]) + noise
            new_score = (new_score - min(new_score))/(max(new_score) - min(new_score))
            
            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=new_score, window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, new_score, plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, new_score)  
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,new_score,2*slidingWindow)
            L1 = [ elem for elem in L]

            dict_acc['R_AUC_ROC']      +=[R_AUC]
            dict_acc['AUC_ROC']        +=[L1[0]]
            dict_acc['R_AUC_PR']       +=[R_AP]
            dict_acc['AUC_PR']         +=[AP]
            dict_acc['VUS_ROC']        +=[avg_auc_3d]
            dict_acc['VUS_PR']         +=[avg_ap_3d]
            dict_acc['Precision']      +=[L1[1]]
            dict_acc['Recall']         +=[L1[2]]
            dict_acc['F']              +=[L1[3]]
            dict_acc['ExistenceReward']+=[L1[5]]
            dict_acc['OverlapReward']  +=[L1[6]]
            dict_acc['Precision@k']    +=[L1[9]]
            dict_acc['Rprecision']     +=[L1[7]]
            dict_acc['Rrecall']        +=[L1[4]]
            dict_acc['RF']             +=[L1[8]]

        methods_acc[methods_score] = dict_acc
    return methods_acc


def compute_anomaly_acc_pairwise(methods_scores,label,slidingWindow,method1,method2):
    
    lag_range = list(range(-slidingWindow//4,slidingWindow//4,5))
    methods_acc = {}
    method_key = [method1]
    if method2 is not None:
        method_key = [method1,method2]
    for i,methods_score in enumerate(tqdm(method_key)):
        dict_acc = {
            'R_AUC_ROC':      [],
            'AUC_ROC':        [],
            'R_AUC_PR':       [],
            'AUC_PR':         [],
            'VUS_ROC':        [],
            'VUS_PR':         [],
            'Precision':      [],
            'Recall':         [],
            'F':              [],
            'ExistenceReward':[],
            'OverlapReward':  [],
            'Precision@k':    [],
            'Rprecision':     [],
            'Rrecall':        [],
            'RF':             []}
        
        for lag in tqdm(range(60)):
            new_lag = random.randint(-slidingWindow//4,slidingWindow//4)
            new_label = generate_new_label(label,new_lag)
            
            noise = np.random.normal(-0.1,0.1,len(methods_scores[methods_score]))
            new_score = np.array(methods_scores[methods_score]) + noise
            new_score = (new_score - min(new_score))/(max(new_score) - min(new_score))
            
            grader = metricor()  

            R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=new_score, window=slidingWindow, plot_ROC=True) 
            L, fpr, tpr= grader.metric_new(new_label, new_score, plot_ROC=True)
            precision, recall, AP = grader.metric_PR(new_label, new_score)  
            #range_anomaly = grader.range_convers_new(new_label)
            Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d = generate_curve(new_label,new_score,2*slidingWindow)
            L1 = [ elem for elem in L]

            dict_acc['R_AUC_ROC']      +=[R_AUC]
            dict_acc['AUC_ROC']        +=[L1[0]]
            dict_acc['R_AUC_PR']       +=[R_AP]
            dict_acc['AUC_PR']         +=[AP]
            dict_acc['VUS_ROC']        +=[avg_auc_3d]
            dict_acc['VUS_PR']         +=[avg_ap_3d]
            dict_acc['Precision']      +=[L1[1]]
            dict_acc['Recall']         +=[L1[2]]
            dict_acc['F']              +=[L1[3]]
            dict_acc['ExistenceReward']+=[L1[5]]
            dict_acc['OverlapReward']  +=[L1[6]]
            dict_acc['Precision@k']    +=[L1[9]]
            dict_acc['Rprecision']     +=[L1[7]]
            dict_acc['Rrecall']        +=[L1[4]]
            dict_acc['RF']             +=[L1[8]]

        methods_acc[methods_score] = dict_acc
    return methods_acc


def normalize_dict_exp(methods_acc_lag,methods_keys):
    key_metrics = [
        'VUS_ROC',
        'VUS_PR',
        'R_AUC_ROC',
        'R_AUC_PR',
        'AUC_ROC',
        'AUC_PR',
        'Rprecision',
        'Rrecall',
        'RF',
        'Precision',
        'Recall',
        'F',
        'Precision@k'
    ][::-1]
    
    norm_methods_acc_lag = {}
    for key in methods_keys:
        norm_methods_acc_lag[key] = {}
        for key_metric in key_metrics:
            ts = methods_acc_lag[key][key_metric]
            new_ts = list(np.array(ts) -  np.mean(ts))
            norm_methods_acc_lag[key][key_metric] = new_ts
    return norm_methods_acc_lag
        
def group_dict(methods_acc_lag,methods_keys):
    key_metrics = [
        'VUS_ROC',
        'VUS_PR',
        'R_AUC_ROC',
        'R_AUC_PR',
        'AUC_ROC',
        'AUC_PR',
        'Rprecision',
        'Rrecall',
        'RF',
        'Precision',
        'Recall',
        'F',
        'Precision@k'
    ][::-1]
    
    norm_methods_acc_lag = {key:[] for key in key_metrics}
    for key in methods_keys:
        for key_metric in key_metrics:
            ts = list(methods_acc_lag[key][key_metric])
            new_ts = list(np.array(ts) -  np.mean(ts))
            norm_methods_acc_lag[key_metric] += new_ts
    return norm_methods_acc_lag


def generate_curve(label,score,slidingWindow):
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d

