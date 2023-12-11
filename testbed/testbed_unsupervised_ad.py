# -*- coding: utf-8 -*-
"""
testbed of unsupervised tabular anomaly detection
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import argparse
import getpass
import time
import numpy as np
import importlib as imp
from utils import get_data_lst, read_data
from deepod.metrics import tabular_metrics


dataset_root = f'/home/{getpass.getuser()}/dataset/1-tabular/'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--input_dir", type=str,
                    default='ADBench-classical',
                    help="the path of the data sets")
parser.add_argument("--output_dir", type=str, default='@record/',
                    help="the output file path")
parser.add_argument("--dataset", type=str, default='FULL',
                    help="FULL represents all the csv file in the folder, "
                         "or a list of data set names split by comma")
parser.add_argument("--model", type=str, default='DeepSVDD', help="",)
parser.add_argument("--auto_hyper", default=True, action='store_true', help="")

parser.add_argument("--normalization", type=str, default='min-max', help="",)
parser.add_argument('--silent_header', action='store_true')
parser.add_argument("--flag", type=str, default='')
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)
data_lst = get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)
print(os.path.join(dataset_root, args.input_dir))
print(data_lst)

module = imp.import_module('deepod.models.tabular')
model_class = getattr(module, args.model)

cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
result_file = os.path.join(args.output_dir, f'{args.model}.{args.input_dir}.{args.flag}.csv')

if not args.silent_header:
    f = open(result_file, 'a')
    print('\n---------------------------------------------------------', file=f)
    print(f'model: {args.model}, collection: {args.input_dir}, '
          f'datasets: {args.dataset}, normalization: {args.normalization}, {args.runs}runs, ', file=f)
    print('---------------------------------------------------------', file=f)
    print('data, auc-roc, std, auc-pr, std, f1, std, time', file=f)
    f.close()


for file in data_lst:
    dataset_name = os.path.splitext(os.path.split(file)[1])[0]

    print(f'\n-------------------------{dataset_name}-----------------------')

    split = '50%-normal'
    print(f'train-test split: {split}, normalization: {args.normalization}')
    x_train, y_train, x_test, y_test = read_data(file=file, split=split,
                                                 normalization=args.normalization,
                                                 seed=42)
    if x_train is None:
        continue

    auc_lst, ap_lst, f1_lst = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    t1_lst, t2_lst = [], []
    runs = args.runs

    model_configs = {}
    if args.auto_hyper:
        clf = model_class(random_state=42)

        # check whether the anomaly detection model supports ray tuning
        if not hasattr(clf, 'fit_auto_hyper'):
            warnings.warn(f'anomaly detection model {args.model} '
                          f'does not support auto tuning hyper-parameters currently.')
            break

        print(f'\nRunning [1/{args.runs}] of [{args.model}] on Dataset [{dataset_name}] (rat tune)')
        tuned_model_configs = clf.fit_auto_hyper(X=x_train,
                                                 X_test=x_test, y_test=y_test,
                                                 n_ray_samples=1, time_budget_s=None)
        model_configs = tuned_model_configs
        print(f'model parameter configure update to: {model_configs}')
        scores = clf.decision_function(x_test)

        auc, ap, f1 = tabular_metrics(y_test, scores)

        print(f'{dataset_name}, {auc:.4f}, {ap:.4f}, {f1:.4f}, {args.model}')

    for i in range(runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

        clf = model_class(**model_configs, random_state=42+i)
        clf.fit(x_train)

        train_time = time.time()
        scores = clf.decision_function(x_test)
        done_time = time.time()

        auc, ap, f1 = tabular_metrics(y_test, scores)
        auc_lst[i], ap_lst[i], f1_lst[i] = auc, ap, f1
        t1_lst.append(train_time - start_time)
        t2_lst.append(done_time - start_time)

        print(f'{dataset_name}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {f1_lst[i]:.4f}, '
              f'{t1_lst[i]:.1f}/{t2_lst[i]:.1f}, {args.model}, {str(model_configs)}')

    avg_auc, avg_ap, avg_f1 = np.average(auc_lst), np.average(ap_lst), np.average(f1_lst)
    std_auc, std_ap, std_f1 = np.std(auc_lst), np.std(ap_lst), np.std(f1_lst)
    avg_time1 = np.average(t1_lst)
    avg_time2 = np.average(t2_lst)

    f = open(result_file, 'a')
    txt = f'{dataset_name}, ' \
          f'{avg_auc:.4f}, {std_auc:.4f}, ' \
          f'{avg_ap:.4f}, {std_ap:.4f}, ' \
          f'{avg_f1:.4f}, {std_f1:.4f}, ' \
          f'{avg_time1:.1f}/{avg_time2:.1f}, {args.model}, {str(model_configs)}'
    print(txt, file=f)
    print(txt)
    f.close()
