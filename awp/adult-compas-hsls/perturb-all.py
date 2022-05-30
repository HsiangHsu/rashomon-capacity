# -*- coding: utf-8 -*-
## standard packages
import pandas as pd
import numpy as np
import pickle as pkl
from time import localtime, strftime
import time
import random
import argparse
import sys
sys.path.insert(1, '../utils/')

from sklearn.model_selection import train_test_split

## custom packages
from DataLoader import *
from capacity import *
from training import *

## accelerating
from itertools import islice
import multiprocessing
from multiprocessing import Pool

## set arguments
parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--dataset', type=str, choices=['adult', 'compas', 'hsls'], default='adult')
parser.add_argument('--nrepeat', type=int, default=5) # 5
parser.add_argument('--test_split', type=float, default=0.1)
parser.add_argument('--nepoch', type=int, default=10)
parser.add_argument('--nneuron', type=int, default=100)
parser.add_argument('--nlayer', type=int, default=5)
parser.add_argument('--trainlr', type=float, default=1e-3)
parser.add_argument('--searchlr', type=float, default=1e-3)
parser.add_argument('--moment', type=float, default=0.9)
parser.add_argument('--prob_target', type=float, default=0.90)
parser.add_argument('--loss_tolerance', type=float, default=0.01)
parser.add_argument('--nepochsearch', type=int, default=5)
parser.add_argument('--lamb', type=float, default=0.1)
args = parser.parse_args()
configuration_dict = vars(args)

""" logging """
## Settings
start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filepath = './'
datapath = '../results/'
filename = args.dataset + '-' + str(args.nrepeat) + '-perturb-all'
log = open(filepath + filename+'-log.txt','w')
log.write('=== {} ===\n'.format(start_time_str))
log.write('Argument Summary\n')
for key in configuration_dict.keys():
    log.write(' {}: {}\n'.format(key, configuration_dict[key]))
log.flush()

## environment setting
log.write('Environment Summary\n')
log.write(' Python version: {}.{}.{}\n'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
log.write(' PyTorch version: {}\n'.format(torch.__version__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.write(' device: {}\n'.format(device))
log.write(' cores: {}\n'.format(multiprocessing.cpu_count()-1))
log.flush()

## load data
log.write('loading dataset...\n')
## prepare X, y
if args.dataset == 'adult':
    df = load_data(name=args.dataset)
    X = df.drop(['income'], axis=1).values
    y = df['income'].values
elif args.dataset == 'compas':
    df = load_data(name=args.dataset)
    X = df.drop(['is_recid'], axis=1).values
    y = df['is_recid'].values
elif args.dataset == 'hsls':
    df = load_hsls('../data/HSLS/', 'hsls_df_knn_impute_past_v2.pkl', [])
    X = df.drop(['gradebin'], axis=1).values
    y = df['gradebin'].values
else:
    log.write(' Invalid dataset\n')
    log.flush()

# n = 32512
n = int(np.round(X.shape[0]*(1-args.test_split)))-1
nclass = len(set(y))
ntest = X.shape[0]-n

log.write(' X.shape = {}, y.shape = {}, nclass = {}\n'.format(X.shape, y.shape, nclass))
log.write(' X_train.shape = {}\n'.format(n))
log.flush()

# print(n)

base_train_loss = np.zeros((args.nrepeat, ))
base_train_acc = np.zeros((args.nrepeat, ))
base_test_loss = np.zeros((args.nrepeat, ))
base_test_acc = np.zeros((args.nrepeat, ))

rashomon_prob = np.zeros((args.nrepeat, ntest, nclass, nclass))
rashomon_loss = np.zeros((args.nrepeat, ntest, nclass))
sample_train_cap = np.zeros((args.nrepeat, ntest))
y_train_all = np.zeros((args.nrepeat, ntest))
# base_test_loss_all = np.zeros((args.nrepeat, ntest))

for j in range(args.nrepeat):
    
    #Creation of Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ntest, random_state=j)
    # print(X_train.shape)
    ## train base model 
    base_training_time = time.localtime()
    log.write('Repeat {:2d}/{:2d}\n'.format(j+1, args.nrepeat))
    log.flush()

    ## random seed
    np.random.seed(j)
    random.seed(j)
    torch.manual_seed(j)
    nfeature = X_train.shape[1]
    nclass = len(set(y_train))
    nn_arch = [nfeature, nclass, args.nneuron, args.nlayer]


    ## create and train model
    base_model = MLP(nn_arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(base_model.parameters(), lr=args.trainlr, momentum=args.moment)
    base_model = train_model(base_model, X_train, y_train, args.nepoch, optimizer, criterion, device)

    ## evaluation
    base_train_loss[j], base_train_acc[j] = evaluate(base_model, X_train, y_train, criterion, device)
    base_test_loss[j], base_test_acc[j] = evaluate(base_model, X_test, y_test, criterion, device)
    # base_test_loss_all[j, :] = evaluate_all_loss(base_model, X_test, y_test, device)

    log.write(' Base train loss {:.4f}, train acc {:.4f}'.format(base_train_loss[j], base_train_acc[j]))
    log.write(' Base test loss {:.4f}, test acc {:.4f}'.format(base_test_loss[j], base_test_acc[j]))
    log.write(' Base training time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(base_training_time))/60))
    log.flush()

    ## set target sample
    total_search_time = time.localtime()
    log.write(' Start searching...\n')
    log.flush()
    for i in range(ntest):
        search_time = time.localtime()
        log.write('repeat: {}/{}, sample: {}/{}\n'.format(j, args.nrepeat, i, ntest))
        log.flush()
        rashomon_prob[j, i, :, :], rashomon_loss[j, i, :] = perturb_all_weights3(base_model, nn_arch, args, criterion, X_train, y_train, X_test, y_test, i, nclass, torch.scalar_tensor(args.prob_target), args.loss_tolerance, device)


    log.write(' Total searching time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(total_search_time))/60))
    log.flush()

    ## compute capacity
    total_capacity_time = time.localtime()
    log.write(' Computing capacity...\n')
    log.flush()
    for i in range(ntest):
        capacity_time = time.localtime()
        channel = rashomon_prob[j, i, :, :]
        sample_train_cap[j, i], _, _ = blahut_arimoto(channel)

    log.write(' Total capacity time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(total_capacity_time))/60))
    log.flush() 



savename = datapath + filename + '-' + str(n) + '-' + str(args.loss_tolerance) + '.npz'
np.savez_compressed(savename,
                    base_train_acc=base_train_acc,
                    base_test_acc=base_test_acc,
                    base_train_loss=base_train_loss,
                    base_test_loss=base_test_loss,
                    rashomon_prob=rashomon_prob,
                    rashomon_loss=rashomon_loss,
                    sample_train_cap=sample_train_cap
                    )
log.write('finished!!!\n')
log.close()




































