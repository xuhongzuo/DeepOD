import os
import pickle
import argparse
import getpass
import time
import numpy as np
import utils
import importlib as imp


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
                         "or a list of data set names splitted by comma")
parser.add_argument("--model", type=str, default='DeepSVDD', help="",)
parser.add_argument("--normalization", type=str, default='min-max', help="",)
parser.add_argument('--silent_header', action='store_true')
parser.add_argument("--flag", type=str, default='')
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)
data_lst = utils.get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)
print(os.path.join(dataset_root, args.input_dir))
print(data_lst)

module = imp.import_module('deepod.models')
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


avg_auc_lst, avg_ap_lst, avg_f1_lst = [], [], []
for file in data_lst:
    dataset_name = os.path.splitext(os.path.split(file)[1])[0]

    print(f'\n-------------------------{dataset_name}-----------------------')

    split = '50%-normal'
    print(f'train-test split: {split}, normalization: {args.normalization}')
    x_train, y_train, x_test, y_test = utils.read_data(file=file,
                                                       split=split,
                                                       normalization=args.normalization,
                                                       seed=42)
    if x_train is None:
        continue

    auc_lst, ap_lst, f1_lst = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    t1_lst, t2_lst = np.zeros(args.runs), np.zeros(args.runs)
    clf = None
    for i in range(args.runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

        clf = model_class(epochs=50, random_state=42+i)
        clf.fit(x_train)
        train_time = time.time()
        scores = clf.decision_function(x_test)
        done_time = time.time()

        auc, ap, f1 = utils.evaluate(y_test, scores)
        auc_lst[i], ap_lst[i], f1_lst[i] = auc, ap, f1
        t1_lst[i] = train_time - start_time
        t2_lst[i] = done_time - start_time

        print(f'{dataset_name}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {f1_lst[i]:.4f}, '
              f'{t1_lst[i]:.1f}/{t2_lst[i]:.1f}, {args.model}')

    avg_auc, avg_ap, avg_f1 = np.average(auc_lst), np.average(ap_lst), np.average(f1_lst)
    std_auc, std_ap, std_f1 = np.std(auc_lst), np.std(ap_lst), np.std(f1_lst)
    avg_time1 = np.average(t1_lst)
    avg_time2 = np.average(t2_lst)

    f = open(result_file, 'a')
    txt = f'{dataset_name}, ' \
          f'{avg_auc:.4f}, {std_auc:.4f}, ' \
          f'{avg_ap:.4f}, {std_ap:.4f}, ' \
          f'{avg_f1:.4f}, {std_f1:.4f}, ' \
          f'{avg_time1:.1f}/{avg_time2:.1f}'
    print(txt, file=f)
    print(txt)
    f.close()

    avg_auc_lst.append(avg_auc)
    avg_ap_lst.append(avg_ap)
    avg_f1_lst.append(avg_f1)


