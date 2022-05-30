
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
parser.add_argument('--nrepeat', type=int, default=1) # 5
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--model', type=str, choices=['vgg', 'resnet18', 'alexnet'], default='vgg')
parser.add_argument('--nmodel', type=int, default=100) # 100
parser.add_argument('--test_split', type=float, default=0.1)
parser.add_argument('--train_batch_size', type=int, default=40)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--nepoch', type=int, default=4)
parser.add_argument('--trainlr', type=float, default=1e-3)
args = parser.parse_args()
configuration_dict = vars(args)

""" logging """
## Settings
start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filepath = './'
datapath = '../results/'
filename = args.dataset + '-' + str(args.nrepeat) + '-sample-all'
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

n = int(np.round(X.shape[0]*(1-args.test_split)))-1
nclass = len(set(y))
ntest = X.shape[0]-n

log.write(' X.shape = {}, y.shape = {}, nclass = {}\n'.format(X.shape, y.shape, nclass))
log.write(' X_train.shape = {}\n'.format(n))
log.flush()


train_acc = np.zeros((args.nrepeat, args.nmodel))
train_loss = np.zeros((args.nrepeat, args.nmodel))
test_acc = np.zeros((args.nrepeat, args.nmodel))
test_loss = np.zeros((args.nrepeat, args.nmodel))
test_likelihood = np.zeros((args.nrepeat, args.nmodel, ntest, nclass))
y_test_all = np.zeros((args.nrepeat, ntest))


## Training
log.write('Sampling...\n')
sample_time = time.localtime()
log.flush()
criterion = nn.CrossEntropyLoss()
for i in range(args.nrepeat):
    repeat_time = time.localtime()
    #Creation of Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ntest, random_state=i)
    y_test_all[i, :] = y_test

    ## build dataloaders
    trainloader = build_loader(X_train, y_train, args.train_batch_size)
    testloader = build_loader(X_test, y_test, args.test_batch_size)
    log.flush()

    for j in range(args.nmodel):
        model_time = time.localtime()

        ## random seed
        np.random.seed(1000*j+i)
        random.seed(1000*j+i)
        # torch.fix_global_seed(1000*j+i)
        torch.manual_seed(1000*j+i)
        torch.cuda.manual_seed(1000*j+i)
        torch.cuda.manual_seed_all(1000*j+i)



        

        ## train
        model = train_model_torch(log, trainloader, args, nclass, criterion, device)

        ## evaluation
        train_acc[i, j], train_loss[i, j], _ = eval_model_torch(trainloader, model, criterion, device)
        test_acc[i, j], test_loss[i, j], test_likelihood[i, j, :, :] = eval_model_torch(testloader, model, criterion, device)

        ## logging
        log.write(' Repeat: {:2d}/{:2d}'.format(i+1, args.nrepeat))
        log.write(' Model: {:3d}/{:3d}'.format(j+1, args.nmodel))
        log.write(' Train acc: {:.4f}, Test acc: {:.4f}'.format(train_acc[i, j], test_acc[i, j]))
        log.write(' Train loss: {:.4f}, Test loss: {:.4f}'.format(train_loss[i, j], test_loss[i, j]))
        log.write(' time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(model_time)) / 60))
        log.flush()

    log.write(' Training time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(repeat_time)) / 60))
    log.flush()

log.write('Total sampling time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(sample_time)) / 60))
log.flush()

savename = datapath + filename + '-' + str(n) + '.npz'
np.savez_compressed(savename,
                    nrepeat=args.nrepeat,
                    nmodel=args.nmodel,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    test_likelihood=test_likelihood,
                    y_test_all=y_test_all
                    )
log.write('finished!!!\n')
log.close()



























