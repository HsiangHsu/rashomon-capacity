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
    prob = F.softmax(outputs, dim=1).cpu().detach().numpy()
    return loss, acc, prob

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
