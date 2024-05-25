from sklearn import metrics
import numpy as np
import math


class metricor:
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias 

    def range_convers_new(self, label):
        '''
        input: arrays of binary values 
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        L = []
        i = 0
        j = 0 
        while j < len(label):
            # print(i)
            while label[i] == 0:
                i+=1
                if i >= len(label):
                    break
            j = i+1
            # print('j'+str(j))
            if j >= len(label):
                if j==len(label):
                    L.append((i,j-1))
    
                break
            while label[j] != 0:
                j+=1
                if j >= len(label):
                    L.append((i,j-1))
                    break
            if j >= len(label):
                break
            L.append((i, j-1))
            i = j
        return L

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            
            x1 = np.arange(e,min(e+window//2,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))
            
            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))
            
        label = np.minimum(np.ones(length), label)
        return label
    
    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            l0 = int((e-s+1)*percentage)
            
            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))
            
            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))
            
        label = np.minimum(np.ones(length), label)
        return label
    
    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred
        
        TP = np.sum(product)
        
        # recall = min(TP/P,1)
        P_new = (P+np.sum(labels))/2      # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP/P_new,1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))
        
        
        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1]+1)])>0:
                existence += 1
                
        existence_ratio = existence/len(L)
        # print(existence_ratio)
        
        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall*existence_ratio
        
        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))
        
        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP/N_new
        
        Precision_RangeAUC = TP/np.sum(pred)
        
        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC
    
    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)
        
        P = np.sum(labels)
        if AUC_type == 'window':
            labels = self.extend_postive_range(labels, window=window)
        else:   
            labels = self.extend_postive_range_individual(labels, percentage=percentage)
        
        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]
        
        for i in np.linspace(0, len(score)-1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score>= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P,L)
            
            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)
            
        TPR_list.append(1)
        FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)
        
        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)
        
        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1])/2
        AUC_range = np.sum(width*height)
        
        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = (prec[1:] + prec[:-1])/2
        AP_range = np.sum(width_PR*height_PR)
        
        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec
        
        return AUC_range

    # TPR_FPR_window
    def RangeAUC_volume(self, labels_original, score, windowSize):
        score_sorted = -np.sort(-score)
        
        tpr_3d=[]
        fpr_3d=[]
        prec_3d=[]
        
        auc_3d=[]
        ap_3d=[]
        
        window_3d = np.arange(0, windowSize+1, 1)
        P = np.sum(labels_original)
       
        for window in window_3d:
            labels = self.extend_postive_range(labels_original, window)
            
            # print(np.sum(labels))
            L = self.range_convers_new(labels)
            TPR_list = [0]
            FPR_list = [0]
            Precision_list = [1]
            
            for i in np.linspace(0, len(score)-1, 250).astype(int):
                threshold = score_sorted[i]
                # print('thre='+str(threshold))
                pred = score>= threshold
                TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P,L)
                
                TPR_list.append(TPR)
                FPR_list.append(FPR)
                Precision_list.append(Precision)
                
            TPR_list.append(1)
            FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)
            
            
            tpr = np.array(TPR_list)
            fpr = np.array(FPR_list)
            prec = np.array(Precision_list)
            
            tpr_3d.append(tpr)
            fpr_3d.append(fpr)
            prec_3d.append(prec)
            
            width = fpr[1:] - fpr[:-1]
            height = (tpr[1:] + tpr[:-1])/2
            AUC_range = np.sum(width*height)
            auc_3d.append(AUC_range)
            
            width_PR = tpr[1:-1] - tpr[:-2]
            height_PR = (prec[1:] + prec[:-1])/2
            AP_range = np.sum(width_PR*height_PR)
            ap_3d.append(AP_range)

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d)/len(window_3d), sum(ap_3d)/len(window_3d)

        
        