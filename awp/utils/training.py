# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp

from time import localtime, strftime
import time

## scikit learn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score

## pytorch
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import models

## torchtext
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import copy


## Multi-layer preceptron with weight perturbation
class MLP(nn.Module):
    def __init__(self, nn_arch):
        super(MLP, self).__init__()
        self.nfeature, self.nclass, self.nneuron, self.nlayer = nn_arch

        self.read_in = nn.Linear(self.nfeature, self.nneuron)
        self.ff = nn.Linear(self.nneuron, self.nneuron)
        self.read_out = nn.Linear(self.nneuron, self.nclass)

    def forward(self, x):
        x = self.read_in(x)
        for i in range(self.nlayer):
            x = F.relu(self.ff(x))

        logits = self.read_out(x)
        return logits

## train model 
def train_model(model, X, y, epoch, optimizer, criterion, device):
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        outputs = model(torch.Tensor(X).to(device))
        loss = criterion(torch.squeeze(outputs), torch.Tensor(y).type(torch.LongTensor).to(device))
        loss.backward()  # Computes the gradient of the given tensor w.r.t. graph leaves
        optimizer.step()

    return model

def evaluate(model, X, y, criterion, device):
    model.eval()
    outputs = model(torch.Tensor(X).to(device))
    # likelihoods = F.softmax(outputs, dim=1).cpu().detach().numpy()
    loss = criterion(torch.squeeze(outputs), torch.Tensor(y).type(torch.LongTensor).to(device)).item()
    _, predicted = outputs.max(1)
    acc = predicted.eq(torch.Tensor(y).type(torch.LongTensor).to(device)).sum().item()/len(y)
    return loss, acc

def evaluate_all_loss(model, X, y, device):
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    outputs = model(torch.Tensor(X).to(device))
    # likelihoods = F.softmax(outputs, dim=1).cpu().detach().numpy()
    loss = criterion(torch.squeeze(outputs), torch.Tensor(y).type(torch.LongTensor).to(device)).cpu().detach().numpy()
    return loss


def perturb_all_weights3(base_model, nn_arch, args, criterion, X, y, Xt, yt, idx_target, nclass, likelihood_tol, loss_percentage, device):
    test_loss, _ = evaluate(base_model, Xt, yt, criterion, device)
    loss_tol = test_loss + loss_percentage
    nfeature = X.shape[1]

    # x_target = X[idx_target, :].reshape((1, nfeature))
    # y_target = y[idx_target]
    x_target = Xt[idx_target, :].reshape((1, nfeature))
    y_target = yt[idx_target]
    # X_red = np.delete(X, idx_target, axis=0)
    # y_red = np.delete(y, idx_target)

    total_prob = np.zeros((nclass, nclass))
    total_loss = np.zeros((nclass, ))

    print('base', test_loss)

    # classes = np.arange(nclass)
    # classes = np.delete(classes, y_target)

    for i in range(nclass):
        
        # target_class = classes[i]
        model = MLP(nn_arch).to(device)
        model.load_state_dict(copy.deepcopy(base_model.state_dict()))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.searchlr, momentum=args.moment)

        x_target_logit = model(torch.Tensor(x_target).to(device))
        x_target_prob = torch.squeeze(F.softmax(x_target_logit, dim=1))
        test_loss, _ = evaluate(model, Xt, yt, criterion, device)
        print('perturb', test_loss)
        
        
        ## perturb 
        store_loss = test_loss
        store_prob = x_target_prob.cpu().detach().numpy()
        cnt = 1
        while (x_target_prob[i] < likelihood_tol) and (test_loss < loss_tol):
            model.train()
            optimizer.zero_grad()
            x_target_logit = model(torch.Tensor(x_target).to(device))
            x_target_logit = torch.squeeze(x_target_logit)
            # x_target_prob = torch.squeeze(F.softmax(x_target_logit, dim=1))
            # loss = likelihood_tol - x_target_prob[i]
            loss = -x_target_logit[i]
            loss.backward()  
            optimizer.step()

            test_loss, _ = evaluate(model, Xt, yt, criterion, device)
            x_target_logit = model(torch.Tensor(x_target).to(device))
            x_target_prob = torch.squeeze(F.softmax(x_target_logit, dim=1))

            if test_loss < loss_tol:
                store_loss = test_loss
                store_prob = x_target_prob.cpu().detach().numpy()
            
            cnt += 1

        total_prob[i, :] = store_prob 
        total_loss[i] = store_loss
            # total_prob[i, :] = x_target_prob.cpu().detach().numpy()
            # total_loss[i] = test_loss
    print(cnt)
    return total_prob, total_loss


def perturb_all_weights_cv3(base_model, args, criterion, X, y, testloader, idx_target, nclass, likelihood_tol, loss_percentage, device):
    _, test_loss, _ = eval_model_torch(testloader, base_model, criterion, device)
    print(test_loss)
    loss_tol = test_loss + loss_percentage
    nfeature = X.shape[1:]

    x_target = X[idx_target].reshape((1, nfeature[0], nfeature[1], nfeature[2]))
    y_target = y[idx_target]
    X_red = np.delete(X, idx_target, axis=0)
    y_red = np.delete(y, idx_target)

    total_prob = np.zeros((nclass, nclass))
    total_loss = np.zeros((nclass, ))


    # classes = np.arange(nclass)
    # classes = np.delete(classes, y_target)

    for i in range(nclass):
        print('Class', i)
        class_time = time.localtime()
        # target_class = classes[i]
        build_time = time.localtime()
        if args.model == 'vgg':
            model = models.vgg16(pretrained = True)
            nlayer  = len(model.classifier)
            input_lastLayer = model.classifier[nlayer-1].in_features
            model.classifier[nlayer-1] = nn.Linear(input_lastLayer, nclass)
        elif args.model == 'resnet18':
            model = models.resnet18(pretrained = True)
            input_lastLayer = model.fc.in_features
            model.fc = nn.Linear(input_lastLayer, nclass)
        elif args.model == 'alexnet':
            model = models.alexnet(pretrained = True)
            nlayer = len(model.classifier)
            input_lastLayer = model.classifier[nlayer-1].in_features
            model.classifier[nlayer-1] = nn.Linear(input_lastLayer, nclass)
        else:
            log.write(' Undefined model!\n')
        print(' Build time: {:.2f} mins'.format((time.mktime(time.localtime()) - time.mktime(build_time))/60))

        clone_time = time.localtime()
        model = model.to(device)        
        model.load_state_dict(copy.deepcopy(base_model.state_dict()))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.searchlr, momentum=args.moment)
        print(' Clone time: {:.2f} mins'.format((time.mktime(time.localtime()) - time.mktime(clone_time))/60))

        x_target_logit = model(torch.Tensor(x_target).to(device))
        x_target_prob = torch.squeeze(F.softmax(x_target_logit, dim=1))
        _, test_loss, _ = eval_model_torch(testloader, model, criterion, device)

        
        
        ## perturb 
        store_loss = test_loss
        store_prob = x_target_prob.cpu().detach().numpy()
        cnt = 1
        while (x_target_prob[i] < likelihood_tol) and (test_loss < loss_tol):
            train_time = time.localtime()
            model.train()
            optimizer.zero_grad()
            x_target_logit = model(torch.Tensor(x_target).to(device))
            x_target_logit = torch.squeeze(x_target_logit)
            # x_target_prob = torch.squeeze(F.softmax(x_target_logit, dim=1))
            # loss = likelihood_tol - x_target_prob[i]
            loss = -x_target_logit[i]
            loss.backward()  
            optimizer.step()
            # print(' Train time: {:.2f} mins'.format((time.mktime(time.localtime()) - time.mktime(train_time))/60))
            
            # test_loss, _ = evaluate(model, Xt, yt, criterion, device)
            eval_time = time.localtime()
            _, test_loss, _ = eval_model_torch(testloader, model, criterion, device)
            # print(' Evaluation time: {:.2f} mins'.format((time.mktime(time.localtime()) - time.mktime(eval_time))/60))
            x_target_logit = model(torch.Tensor(x_target).to(device))
            x_target_prob = torch.squeeze(F.softmax(x_target_logit, dim=1))

            if test_loss < loss_tol:
                store_loss = test_loss
                store_prob = x_target_prob.cpu().detach().numpy()
                
                cnt += 1

        total_prob[i, :] = store_prob 
        total_loss[i] = store_loss
        print(cnt)
        # print(' Class time: {:.2f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(class_time))/60))
    return total_prob, total_loss

# ## train with neural networks
def build_loader(X, y, nbatch):
    X, y = torch.Tensor(X), torch.Tensor(y)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, shuffle=True, batch_size=nbatch)
    return loader

def train_model_torch(log, loader, args, nclass, criterion, device):
    if args.model == 'vgg':
        model = models.vgg16(pretrained = True)
        nlayer  = len(model.classifier)
        input_lastLayer = model.classifier[nlayer-1].in_features
        model.classifier[nlayer-1] = nn.Linear(input_lastLayer, nclass)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained = True)
        input_lastLayer = model.fc.in_features
        model.fc = nn.Linear(input_lastLayer, nclass)
    elif args.model == 'alexnet':
        model = models.alexnet(pretrained = True)
        nlayer = len(model.classifier)
        input_lastLayer = model.classifier[nlayer-1].in_features
        model.classifier[nlayer-1] = nn.Linear(input_lastLayer, nclass)
    else:
        log.write(' Undefined model!\n')

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.trainlr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.nepoch):
        model.train()  # prep model for training
        for batch_i, (inputs, targets) in enumerate(loader, start=0):
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model

def eval_model_torch(loader, model, criterion, device):
    model.eval()
    losses = 0
    correct = 0
    total = 0
    output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            output.append(F.softmax(outputs, dim=1))
            # print(F.softmax(outputs, dim=0).shape)
    losses = losses/(batch_idx+1)
    acc = correct/total
    output = torch.cat(output, dim=0).cpu().detach().numpy()
    # print(output.shape)
    return acc, losses, output


