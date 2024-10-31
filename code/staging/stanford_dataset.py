'''
Code for training model with 2 layers for predicting type
'''
import glob
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import plotly
import plotly.graph_objs as go
import torchvision
from torchvision import transforms
import random
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
import torchvision.models as models
import json
from torch.utils.data import Dataset
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import os
from torchmetrics import JaccardIndex
import sys
# In[4]:

n_attributes = 14
n_groups = 4
expt_num = sys.argv[1]
PATH = f'../../data/models/stanford_combined_pool_TiRADS_P_loss_pred_reproduce_sampled.pt'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")    
EPOCHS = 170
EPOCH_CLASSIFIER = 150
"""
def projection_simplex_sort(v, z=1):
    '''
    Used to nomalize the G matrix for mpping from attributes to groups
    '''

    n_features = v.size(1)
    u,_ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u,1) - z
    ind = torch.arange(n_features).type_as(v) + 1
    cond = u - cssv / ind > 0
    #rho = ind[cond][-1]
    rho,ind_rho = (ind*cond).max(1)
    #theta = cssv[cond][-1] / float(rho)
    theta = torch.gather(cssv,1,ind_rho[:,None]) / rho[:,None]
    w = torch.clamp(v - theta, min=0)
    return w
"""

class thyroidActualDataset(Dataset):
    def __init__(self, split):
        with open('../legacy_code/labels_stanford_' + split + '.txt') as f:
            self.cases = f.read().splitlines()
            self.samples = []
            self.types_count = [0,0]
            self.types = []
            for case in self.cases:
                
                case_num = case.split(',')[0]
                files = glob.glob('/home/ahana/Stanford_thyroid/thyroidultrasoundcineclip/images/' + case_num + '*')
                case_data = case.split(',')[1:]
                
               
                for file_name in files: #[:1]:
                    file_no_ext = file_name.split('.')[0]
                    case_frame = file_no_ext.split('/')[-1]
                    if int(case_frame.split('_')[-1]) % 10 == 0:
                        sample = case.replace(case_num, file_name)
                        self.samples.append(sample)
                        if int(case_data[-1]) <= 3:
                            self.types.append(0)
                            self.types_count[0] = self.types_count[0] + 1
                        else:
                            self.types.append(1)
                            self.types_count[1] = self.types_count[1] + 1
                    

        
        
        self.sample_weights = [1/self.types_count[label] for label in self.types]
        
        self.scores = [0, 1, 2,    0, 1, 2, 3,     0, 2, 3,    0, 1,2,3,   50]      
        self.cases = self.samples      
        

        

    def __len__(self):
        return len(self.samples)

    

    def translate(self, img, shift, direction):
        roll=True
        assert direction in [0, 1, 2, 3], 'Directions should be top|up|left|right'
        #print(np.shape(img))
        img = img.copy()
        if direction == 0:
            right_slice = img[:, -shift:].copy()
            img[:, shift:] = img[:, :-shift]
            if roll:
                img[:,:shift] = np.fliplr(right_slice)
        if direction == 1:
            left_slice = img[:, :shift].copy()
            img[:, :-shift] = img[:, shift:]
            if roll:
                img[:, -shift:] = left_slice
        if direction == 2:
            down_slice = img[-shift:, :].copy()
            img[shift:, :] = img[:-shift,:]
            if roll:
                img[:shift, :] = down_slice
        if direction == 3:
            upper_slice = img[:shift, :].copy()
            img[:-shift, :] = img[shift:, :]
            if roll:
                img[-shift:,:] = upper_slice
        return img



    def __getitem__(self, idx):
        
        data_list = self.samples[idx].split(', ')
        name = data_list[0]
        #print(name)

        labels = data_list[1:16]
        labels = [float(element) for element in labels]
        labels = np.array(labels)
        if int(data_list[21]) > 3:
             labels[14] = 1
        else:
             labels[14] = 0
        
        im1 = cv2.imread(str(name))
            
        mask = cv2.imread(str(name).replace('images','masks_new'))
            
        mask[mask==255] = 1
            
        im = np.zeros((256, 256, 3))
        mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        im1 = cv2.resize(im1, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        im[:220,:,:] = im1[:220,:,:]
           
       
        im = im[:,:,:1]
        mask = mask[:,:,:1]
        masked = im * mask
         # Adding data augmentation to avoid overfitting
        rand1 = np.random.randint(-20, 0)
        rand2 = np.random.randint(0, 4)
        im = self.translate(im, rand1, rand2)
        mask = self.translate(mask, rand1, rand2)
       
        if random.randint(1, 10) > 5:
            im = np.flipud(im)
            mask = np.flipud(mask)
            masked = np.flipud(masked)
        if random.randint(1, 10) > 5:
            im = np.fliplr(im)
            mask = np.fliplr(mask)
            masked = np.fliplr(masked)
        if random.randint(1, 10) > 5:
            for i in range(random.randint(1, 4)):
                im = np.rot90(im)
               
        im = np.ascontiguousarray(im)
        mask = np.ascontiguousarray(mask)
        masked = np.ascontiguousarray(masked)
        transforms = Compose([ToTensor()])
        im = transforms(im)
        masked = transforms(masked)
        mask = torch.from_numpy(mask).long().view(256,256)
        #im_masked = transforms(im_masked)
        #labels =  torch.from_numpy(labels)
        #labels = labels.type(torch.LongTensor)
        #labels = torch.unsqueeze(labels, 0)
        sample = {"images": im.float(), "mask": mask, "labels": torch.from_numpy(labels), "scores": torch.from_numpy(np.array(labels.dot(self.scores))), "types": self.types, "name": name}
        return sample





