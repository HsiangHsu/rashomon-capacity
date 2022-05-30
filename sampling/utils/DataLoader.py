# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics as sm
import sys
from sklearn.preprocessing import MinMaxScaler

## torchvision
import torchvision
# from torchvision import datasets, transforms

## torchtext
import torchtext
# from torchtext import datasets
from torchtext.data.functional import to_map_style_dataset

## method for loading adult/compas datasets
def load_data(name='adult'):
    
    #% Processing for UCI-ADULT
    if name == 'adult':
        file = '../data/UCI-Adult/adult.data'
        fileTest = '../data/UCI-Adult/adult.test'
        
        df = pd.read_csv(file, header=None,sep=',\s+',engine='python')
        dfTest = pd.read_csv(fileTest,header=None,skiprows=1,sep=',\s+',engine='python') 
        
        
        columnNames = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        
        df.columns = columnNames
        dfTest.columns = columnNames
        
        df = df.append(dfTest)
        
        # drop columns that won't be used
        dropCol = ["fnlwgt","workclass","occupation"]
        df.drop(dropCol,inplace=True,axis=1)
        
        # keep only entries marked as ``White'' or ``Black''
        ix = df['race'].isin(['White','Black'])
        df = df.loc[ix,:]
        
        # binarize race
        # Black = 0; White = 1
        df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='White' else 0)
        
        # binarize gender
        # Female = 0; Male = 1
        df.loc[:,'gender'] = df['gender'].apply(lambda x: 1 if x=='Male' else 0)
        
        # binarize income
        # '>50k' = 1; '<=50k' = 0
        df.loc[:,'income'] = df['income'].apply(lambda x: 1 if x[0]=='>' else 0)
        
        
        # drop "education" and native-country (education already encoded in education-num)
        features_to_drop = ["education","native-country"]
        df.drop(features_to_drop,inplace=True,axis=1)
        
        
        
        # create one-hot encoding
        categorical_features = list(set(df)-set(df._get_numeric_data().columns))
        df = pd.concat([df,pd.get_dummies(df[categorical_features])],axis=1,sort=False)
        df.drop(categorical_features,inplace=True,axis=1)
        
        # reset index
        df.reset_index(inplace=True,drop=True)
        
    
    #% Processing for COMPAS
    if name == 'compas':
        file = '../data/COMPAS/compas-scores-two-years.csv'
        df = pd.read_csv(file,index_col=0)
        
        # select features for analysis
        df = df[['age', 'c_charge_degree', 'race',  'sex', 'priors_count', 
                    'days_b_screening_arrest',  'is_recid',  'c_jail_in', 'c_jail_out']]
        
        # drop missing/bad features (following ProPublica's analysis)
        # ix is the index of variables we want to keep.

        # Remove entries with inconsistent arrest information.
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix

        # remove entries entries where compas case could not be found.
        ix = (df['is_recid'] != -1) & ix

        # remove traffic offenses.
        ix = (df['c_charge_degree'] != "O") & ix


        # trim dataset
        df = df.loc[ix,:]

        # create new attribute "length of stay" with total jail time.
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)
        
        
        # drop 'c_jail_in' and 'c_jail_out'
        # drop columns that won't be used
        dropCol = ['c_jail_in', 'c_jail_out','days_b_screening_arrest']
        df.drop(dropCol,inplace=True,axis=1)
        
        # keep only African-American and Caucasian
        df = df.loc[df['race'].isin(['African-American','Caucasian']),:]
        
        # binarize race 
        # African-American: 0, Caucasian: 1
        df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='Caucasian' else 0)
        
        # binarize gender
        # Female: 1, Male: 0
        df.loc[:,'sex'] = df['sex'].apply(lambda x: 1 if x=='Male' else 0)
        
        # rename columns 'sex' to 'gender'
        df.rename(index=str, columns={"sex": "gender"},inplace=True)
        
        # binarize degree charged
        # Misd. = -1, Felony = 1
        df.loc[:,'c_charge_degree'] = df['c_charge_degree'].apply(lambda x: 1 if x=='F' else -1)
               
        # reset index
        df.reset_index(inplace=True,drop=True)
        
    return df

def load_hsls(file_path, filename, vars):
    ## group_feature can be either 'sexbin' or 'racebin'
    ## load csv
    df = pd.read_pickle(file_path+filename)

    ## if no variables specified, include all variables
    if vars != []:
        df = df[vars]

    ## Setting NaNs to out-of-range entries
    ## entries with values smaller than -7 are set as NaNs
    df[df <= -7] = np.nan

    ## Dropping all rows or columns with missing values
    ## this step significantly reduces the number of samples
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variables
    ## X1SEX: 1 -- Male, 2 -- Female, -9 -- NaN -> Preprocess it to: 0 -- Female, 1 -- Male, drop NaN
    ## X1RACE: 0 -- BHN, 1 -- WA
    df['gradebin'] = df['grade9thbin']
    df['racebin'] = np.logical_or(((df['studentrace']*7).astype(int)==7).values, ((df['studentrace']*7).astype(int)==1).values).astype(int)
    df['sexbin'] = df['studentgender'].astype(int)


    ## Dropping race and 12th grade data just to focus on the 9th grade prediction ##
    df = df.drop(columns=['studentgender', 'grade9thbin', 'grade12thbin', 'studentrace'])

    ## Scaling ##
    # scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    ## Balancing data to have roughly equal race=0 and race =1 ##
    # df = balance_data(df, group_feature)
    return df

## image datasets
# def load_mnist():
#     root = './mnist/data'
#     train_set = datasets.MNIST(root=root, train=True, download=True)
#     test_set = datasets.MNIST(root=root, train=False, download=True)
#
#     X = np.concatenate((np.expand_dims(train_set.data, axis=3), np.expand_dims(test_set.data, axis=3)), axis=0)
#     y = np.concatenate((np.array(train_set.targets), np.array(test_set.targets)), axis=0)
#     X = X.transpose((0, 3, 1, 2))
#     ## X.shape = (60000, 1, 28, 28)
#     ## y.shape = (60000,)
#     return X, y

def load_cifar10():
    root = './cifar10/data'
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    X = np.concatenate((train_set.data, test_set.data), axis=0)
    y = np.concatenate((np.array(train_set.targets), np.array(test_set.targets)), axis=0)
    X = X.transpose((0, 3, 1, 2))
    ## X.shape = (60000, 3, 32, 32)
    ## y.shape = (60000,)
    return X, y

def load_cifar100():
    root = './cifar100/data'
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True)

    X = np.concatenate((train_set.data, test_set.data), axis=0)
    y = np.concatenate((np.array(train_set.targets), np.array(test_set.targets)), axis=0)
    X = X.transpose((0, 3, 1, 2))
    ## X.shape = (60000, 3, 32, 32)
    ## y.shape = (60000,)
    return X, y

## text datasets
def load_agnews():
    root = '../data/agnews'
    trainset = torchtext.datasets.AG_NEWS(root=root, split='train')
    testset = torchtext.datasets.AG_NEWS(root=root, split='test')
    trainset = to_map_style_dataset(trainset)
    testset = to_map_style_dataset(testset)
    return trainset+testset

## 
def load_npz(filepath, filename):
    data = np.load(filepath+filename, allow_pickle=True)
    nrepeat = data['nrepeat']
    nmodel = data['nmodel']
    train_acc = data['train_acc']
    test_acc = data['test_acc']
    train_loss = data['train_loss']
    test_loss = data['test_loss']
    loss_percentage = ['loss_percentage']
    test_likelihood = data['test_likelihood']
    test_capacity = data['test_capacity']
    quantiale_list = data['quantiale_list']
    test_rashomon_size = data['test_rashomon_size']
    return nrepeat, nmodel, train_acc, test_acc, train_loss, test_loss, \
           loss_percentage, test_likelihood, test_capacity, quantiale_list, test_rashomon_size

def load_npz_sample(filepath, filename):
    data = np.load(filepath+filename, allow_pickle=True)
    base_train_acc = data['base_train_acc'],
    base_test_acc = data['base_test_acc'],
    base_train_loss = data['base_train_loss'],
    base_test_loss = data['base_test_loss'],
    base_test_loss_all=data['base_test_loss_all'],
    rashomon_prob = data['rashomon_prob'],
    rashomon_loss = data['rashomon_loss'],
    sample_train_cap = data['sample_train_cap'],
    y_train_all = data['y_train_all']

    return base_train_acc, base_test_acc, base_train_loss, base_test_loss, base_test_loss_all, rashomon_prob, rashomon_loss, sample_train_cap, y_train_all 