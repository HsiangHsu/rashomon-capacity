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

## set arguments
parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--nrepeat', type=int, default=1) # 5
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--model', type=str, choices=['vgg', 'resnet18', 'alexnet'], default='vgg')
parser.add_argument('--test_split', type=float, default=0.3)
parser.add_argument('--train_batch_size', type=int, default=40)
parser.add_argument('--test_batch_size', type=int, default=10000)
parser.add_argument('--nepoch', type=int, default=4)
parser.add_argument('--trainlr', type=float, default=1e-3)
parser.add_argument('--searchlr', type=float, default=1e-4)
parser.add_argument('--moment', type=float, default=0.9)
parser.add_argument('--prob_target', type=float, default=0.90)
parser.add_argument('--loss_tolerance', type=float, default=0.01)
parser.add_argument('--nepochsearch', type=int, default=5)
parser.add_argument('--lamb', type=float, default=0.1)
parser.add_argument('--searchbatch', type=int, default=1000)
args = parser.parse_args()
configuration_dict = vars(args)

""" logging """
## Settings
start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filepath = './'
datapath = '../results/'
filename = args.dataset + '-' + str(args.nrepeat) + '-' + args.model + '-perturb-all'
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
if args.dataset == 'mnist':
    X, y = load_mnist()
elif args.dataset == 'cifar10':
    X, y = load_cifar10()
elif args.dataset == 'cifar100':
    X, y = load_cifar100()
else:
    log.write(' Undefined dataset\n')


log.write(' X.shape= {}, y.shape = {}\n'.format(X.shape, y.shape))
log.flush()


n = 50
# n = int(np.round(X.shape[0]*(1-args.test_split)))-1
nclass = len(set(y))

base_train_loss = np.zeros((args.nrepeat, ))
base_train_acc = np.zeros((args.nrepeat, ))
base_test_loss = np.zeros((args.nrepeat, ))
base_test_acc = np.zeros((args.nrepeat, ))

rashomon_prob = np.zeros((args.nrepeat, n, nclass, nclass))
rashomon_loss = np.zeros((args.nrepeat, n, nclass))
sample_train_cap = np.zeros((args.nrepeat, n))
y_train_all = np.zeros((args.nrepeat, n))

criterion = nn.CrossEntropyLoss()
for j in range(args.nrepeat):
    repeat_time = time.localtime()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split, random_state=j)

    ## build dataloaders
    trainloader = build_loader(X_train, y_train, args.train_batch_size)
    testloader = build_loader(X_test, y_test, args.test_batch_size)
    log.flush()

    base_training_time = time.localtime()
    log.write('Training base model...\n')
    log.flush()

    ## random seed
    np.random.seed(j)
    random.seed(j)
    torch.manual_seed(j)

    ## train
    base_model = train_model_torch(log, trainloader, args, nclass, criterion, device)

    ## evaluation
    base_train_acc[j], base_train_loss[j], _ = eval_model_torch(trainloader, base_model, criterion, device)
    base_test_acc[j], base_test_loss[j], _ = eval_model_torch(testloader, base_model, criterion, device)

    ## logging
    log.write(' Repeat: {:2d}/{:2d}'.format(j+1, args.nrepeat))
    log.write(' Base train loss {:.4f}, train acc {:.4f}'.format(base_train_loss[j], base_train_acc[j]))
    log.write(' Base test loss {:.4f}, test acc {:.4f}'.format(base_test_loss[j], base_test_acc[j]))
    log.write(' Base training time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(base_training_time))/60))
    log.flush()


    total_search_time = time.localtime()
    log.write('Start searching...\n')
    log.flush()
    for i in range(n):
        search_time = time.localtime()
        rashomon_prob[j, i, :, :], rashomon_loss[j, i, :] = perturb_all_weights_cv3(base_model, args, criterion, X_test, y_test, testloader, i, nclass, torch.scalar_tensor(args.prob_target), args.loss_tolerance, device)
        log.write(' Repeat {:2d}/{:2d}'.format(j+1, args.nrepeat))
        log.write(' Sample {:5d}/{:5d} time: {:.2f} mins\n'.format(i, n, (time.mktime(time.localtime()) - time.mktime(search_time))/60))
        log.flush()

    log.write(' Total searching time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(total_search_time))/60))
    log.flush()

    ## compute capacity
    # sample_train_cap = np.zeros((n, ))
    total_capacity_time = time.localtime()
    log.write('Computing capacity...\n')
    log.flush()
    for i in range(n):
        capacity_time = time.localtime()
        channel = rashomon_prob[j, i, :, :]
        sample_train_cap[j, i], _, _ = blahut_arimoto(channel)

        log.write(' Repeat {:2d}/{:2d}'.format(j+1, args.nrepeat))
        log.write(' Sample {:5d}'.format(i))
        log.write(' capacity {:.4f}'.format(2**sample_train_cap[j, i]))
        log.write(' time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(capacity_time))/60))
        log.flush()

    log.write('Total capacity time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(total_capacity_time))/60))
    log.flush() 

    y_train_all[j, :] = y_train[:n]

savename = datapath + filename + '-' + str(n) + '-' + str(args.loss_tolerance) + '.npz'
np.savez_compressed(savename,
                    base_train_acc=base_train_acc,
                    base_test_acc=base_test_acc,
                    base_train_loss=base_train_loss,
                    base_test_loss=base_test_loss,
                    rashomon_prob=rashomon_prob,
                    rashomon_loss=rashomon_loss,
                    sample_train_cap=sample_train_cap,
                    y_train_all=y_train_all)
log.write('finished!!!\n')
log.close()






















































