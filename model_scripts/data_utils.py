# Imports
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Data normalization
import albumentations as A # albumentations - a Python library for image augmentation
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm # tqdm - used for creating Progress Meters or Progress Bars
from sklearn import preprocessing

import logging

loggerupds = logging.getLogger('update')

# Classes
class CropDataset:
    def __init__(self, data_fname, test_size=0.2, validation_size=0.1):
        self.df = pd.read_csv(data_fname)
        print("self.df : {}".format(self.df))
        
        '''
        # sklearn normalizing using MinMaxScaler() method
        x = self.df.iloc[:, :-1]#.values #returns a numpy array
        # minmax scalar normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        #df = pd.DataFrame(x_scaled)
        normalized_x = pd.DataFrame(x_scaled)
        print("MinMaxScaler -> normalized_x : {}".format(normalized_x))
        '''
    
        # sklearn normalizing using Normalizer method
        x = self.df.iloc[:, :-1]#.values #returns a numpy array
        # minmax scalar normalization
        #min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler = preprocessing.Normalizer().fit(x)  # fit does nothing
        x_scaled = min_max_scaler.transform(x)
        #x_scaled = min_max_scaler.fit_transform(x)
        #df = pd.DataFrame(x_scaled)
        normalized_x = pd.DataFrame(x_scaled)
        print("Normalizer -> normalized_x : {}".format(normalized_x))
        # Normalization test
        self.dfX_train, self.dfX_test, self.dfy_train, self.dfy_test = train_test_split(normalized_x, self.df.iloc[:, -1], test_size=test_size)
        
        #self.dfX_train, self.dfX_test, self.dfy_train, self.dfy_test = train_test_split(self.df.iloc[:, :-1], self.df.iloc[:, -1], test_size=test_size)
        
        self.dfX_train, self.dfX_validation, self.dfy_train, self.dfy_validation = train_test_split(self.dfX_train, self.dfy_train, test_size=validation_size)
        
        print(np.unique(self.dfy_train, return_counts=True),np.unique(self.dfy_validation, return_counts=True), np.unique(self.dfy_test, return_counts=True) )
        
        #self.dfX_train, self.dfX_test, self.dfy_train, self.dfy_test = train_test_split(self.df.iloc[:, :-1], self.df.iloc[:, -1], test_size=test_size)
        #self.dfX_train, self.dfX_validation, self.dfy_train, self.dfy_validation = train_test_split(self.dfX_train, self.dfy_train, test_size=validation_size)
        
        print(f'Number of Samples [Total] : {self.df.shape[0]}')
        print(f'Number of Samples [Train] : {self.dfX_train.shape[0]}')
        print(f'Number of Samples [Validation] : {self.dfX_validation.shape[0]}')
        print(f'Number of Samples [Test] : {self.dfX_test.shape[0]}')
        loggerupds.info(f'Number of Samples [Total] : {self.df.shape[0]}')
        loggerupds.info(f'Number of Samples [Train] : {self.dfX_train.shape[0]}')
        loggerupds.info(f'Number of Samples [Validation] : {self.dfX_validation.shape[0]}')
        loggerupds.info(f'Number of Samples [Test] : {self.dfX_test.shape[0]}')

    def get_next_batch(self, use_data='train', batch_size=100):
                     
        if use_data == 'train':
            X = self.dfX_train.copy()
            y = self.dfy_train.copy()
        elif use_data == 'validation':
            X = self.dfX_validation.copy()
            y = self.dfy_validation.copy()
        elif use_data == 'test':
            X = self.dfX_test.copy()
            y = self.dfy_test.copy()
        else:
            raise ValueError(f'use_data can only be one of "train", "validation", "test". Cannot be "{use_data}"')

        data = pd.concat([X, y], axis=1)
        #print("get_next_batch -> data = {}".format(data))
        positives = data[data.label == 1].iloc[:, :-1]
        #print("get_next_batch -> positives = {}".format(positives))
        negatives = data[data.label == 0].iloc[:, :-1]
        #print("get_next_batch -> negatives = {}".format(negatives))
        positives = positives.sample(frac=1)
        #print("get_next_batch -> positives.sample(frac=1) : {}".format(positives))

        pos_batch_size = batch_size // 2
        print("pos_batch_size = {}".format(pos_batch_size))
        neg_batch_size = batch_size - pos_batch_size
        print("neg_batch_size = {}".format(neg_batch_size))

        start = 0
        num_positive_samples = positives.shape[0]
        print("num_positive_samples : {}".format(num_positive_samples))

        while start < num_positive_samples:
            end = start + pos_batch_size
            positive_batch = positives.iloc[start:end]
            # positive_batch['label'] = 1
            positive_batch = positive_batch.assign(label=1)

            if neg_batch_size > len(negatives):
                replace = True
            else:
                replace = False
            negative_batch = negatives.sample(n=neg_batch_size, replace=replace)
            # negative_batch['label'] = 0
            negative_batch = negative_batch.assign(label=0)

            batch = pd.concat([positive_batch, negative_batch], axis=0)
            batch = batch.sample(frac=1)
            batchX = batch.iloc[:, :-1]
            batchy = batch.iloc[:, -1]
            batchX, batchy = torch.tensor(batchX.values).float(), torch.tensor(batchy.values).long()
            batchX = batchX.view(-1, 2, batchX.shape[1]//2)
            # batchX = batchX.view(-1, 2, 15)
            batchy = batchy.view(-1, 1)
            start += batch_size
            yield batchX.transpose(2, 1), batchy
