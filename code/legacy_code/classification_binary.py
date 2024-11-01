#!/usr/bin/env python
# coding: utf-8

# https://scholar.google.com/scholar?start=0&hl=en&as_sdt=80005&sciodt=0,11&cites=3033757799749220519&scipsc=
# 
# https://arxiv.org/pdf/2006.05861.pdf
# 
# 
# http://cimalab.intec.co/applications/thyroid/
# 
# # self attention

# In[1]:

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from unet_models import UNet_encoder

use_cuda = torch.cuda.is_available()

class thyroidDataset(Dataset):
    def __init__(self, split):
        self.all_data = []
        self.compositions = {'Unknown':0, 'cystic':1,
                             'predominantly solid':2,
                             'solid':3, 'spongiform appareance':4}
        self.echogenicities = {'Unknown':0, 'hyperechogenecity':1,
                             'hypoechogenecity':2, 'isoechogenicity':3,
                             'marked hypoechogenecity':4}
        self.margins = {'Unknown':0, 'ill- defined':1, 'microlobulated':2,
                        'spiculated':3, 'well defined smooth':4}
        self.calcifications = {'macrocalcification':0, 'microcalcification':1, 'non':2}
        self.types ={'benign':0, 'malign':1}
        for t_type in ['benign', 'malign']:
            root_dir=Path('../data/' + split + '/' + t_type).expanduser().resolve().absolute() 
            files = list(root_dir.glob("*"))
            labels = [t_type] * len(files)
            data_list = list(zip(files, labels))
            self.all_data.extend(data_list)
        random.shuffle(self.all_data)
        self.cases, self.types = zip(*self.all_data)
        print("number of data items:" + str(len(self.cases)))
            

    def __len__(self):
        return len(self.cases)
  
    def __getitem__(self, idx):
        labels = np.zeros(15, dtype = float)
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).getroot()
        for x in xml_data:
            if x.tag=='composition' and x.text is not None:
                composition = x.text
                if self.compositions[composition] > 0:
                    labels[self.compositions[composition] - 1] += 1.0
            if x.tag=='echogenicity' and x.text is not None:
                echogenicity = x.text
                if self.echogenicities[echogenicity] > 0:
                    labels[self.echogenicities[echogenicity] + 3] += 1.0
            if x.tag=='margins' and x.text is not None:
                margin = x.text
                if self.margins[margin] > 0:
                    labels[self.margins[margin] + 7] += 1.0
            if x.tag=='calcifications' and x.text is not None:
                calcification = x.text
                labels[self.calcifications[calcification] + 12] += 1.0
        im_name = list(self.cases[idx].glob('*[0-9].jpg'))[0]
        im = cv2.imread(str(im_name))
        im = cv2.resize(im, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
        # Adding data augmentation to avoid overfitting
        if random.randint(1, 10) > 5:
            im = np.flipud(im)
        if random.randint(1, 10) > 5:
            im = np.fliplr(im)
        if random.randint(1, 10) > 5:
            for i in range(random.randint(1, 4)):
                im = np.rot90(im)
        im = np.ascontiguousarray(im)
        transforms = Compose([ToTensor()])
        im = transforms(im)
        sample = {"image": im, "labels": torch.from_numpy(labels)}
        return sample
    

class net(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256,3)
        #self.GAP = nn.AdaptiveAvgPool2d((1,1))
        
        #self.features = [self.conv1, self.conv2, self.conv3, self.conv4]
        
        self.fc1 = nn.Linear(12544, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)
        
        #self.G_ = torch.nn.Parameter(G)
        #self.W_ = torch.nn.Parameter(W)
        #self.CNN_ = torch.nn.Parameter(CNN)
        


    def forward(self, x):
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # Adding 1 more conv, a GAP and a final linear layer for CAM
        x = self.pool(F.relu(self.conv5(x)))
        #features = x
        #x = self.GAP(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        
        #x = torch.unsqueeze(x, 2)
        #print(x.size())
        
        #g = torch.matmul(self.G_, x) 
        
        #g = g.repeat((1, 1, k))
        
        #y = g * self.W_
        
        #y, _ = y.max(axis=2)
        
        #y = torch.transpose(y, 1, 0)
        
        #y = torch.sum(y, axis=0)
        
        #return (torch.sigmoid(y), x, features)
        return torch.sigmoid(y)

def train_model(architecture):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    parameters_train = {
        "batch_size": 8,
        "shuffle": True,
    }
    parameters_test = {
        "batch_size": 1,
        "shuffle": False,
    }
    training_set = thyroidDataset(split='train')
    training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)
    totiter = len(training_generator)
    if architecture == 0:
        print("Training VG16")
        model = models.vgg16(pretrained=False)
        model.classifier._modules['6'] = torch.nn.Linear(4096, 15)
    elif architecture == 1:
        print("Training Unet encoder")
        model = UNet_encoder(3, 15)
    else:
        print("Training custom model")
        model = net()

    rescale = torch.nn.Sigmoid()
    model = model.to(device)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(600):
        running_loss = 0.0
        model.train()
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            model.train(True)
            x_train = data["image"]
            # print(x_train.size())
            y_train = data["labels"]
            x_train, y_train = (
                x_train.to(device),
                y_train.to(device),
            )
            optimizer.zero_grad()

        # forward + backward + optimize
            output = model(x_train)
            output = rescale(output)
            loss = criterion(output.float(), y_train.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / totiter))
        
    print("Training complete")
    torch.save(model.state_dict(), f'../data/models/classification_binary_{architecture}.pt')
    return model

def perform_test(model):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    parameters_test = {
        "batch_size": 1,
        "shuffle": False,
    }
    rescale = torch.nn.Sigmoid()
    training_set = thyroidDataset(split='train')
    training_generator = torch.utils.data.DataLoader(training_set, **parameters_test)
    test_set = thyroidDataset(split='test')
    test_generator = torch.utils.data.DataLoader(test_set, **parameters_test)
    totiter = len(test_generator)
    total = 0
    correct = 0
    class_correctness = np.zeros(15, dtype=int)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_generator), total=totiter):
            x_test = data["image"]
            y_test = data["labels"]
            x_test, y_test = (
                x_test.to(device),
                y_test.to(device),
            )
            output = model(x_test)
            output = rescale(output)
            predicted = output.data
            predicted = predicted.detach().cpu().numpy()
            predicted = predicted.flatten()
            y_test = y_test.detach().cpu().numpy()
            y_test = y_test.flatten()
            predicted[predicted < 0.5] = 0.0
            predicted[predicted >= 0.5] = 1.0
            total += 15
            comparison = y_test == predicted
            print(comparison[:4].all(), comparison[4:8].all(), comparison[8:12].all(), comparison[12:].all())
            errors = np.ones(15, dtype = int)
            errors[predicted == y_test] = 0
            correct = correct + 15 - np.sum(errors)
            for i in range(15):
                class_correctness[i] = class_correctness[i] + 1 - errors[i]
            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Per class acuracy is: ')
    print((class_correctness / totiter) * 100)

    totiter = len(training_generator)
    total = 0
    correct = 0
    class_correctness = np.zeros(15, dtype=int)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            x_test = data["image"]
            y_test = data["labels"]
            x_test, y_test = (
                x_test.to(device),
                y_test.to(device),
            )
            output = model(x_test)
            output = rescale(output)
            predicted = output.data
            predicted = predicted.detach().cpu().numpy()
            predicted = predicted.flatten()
            y_test = y_test.detach().cpu().numpy()
            y_test = y_test.flatten()
            predicted[predicted < 0.5] = 0.0
            predicted[predicted >= 0.5] = 1.0
            total += 15
            errors = np.ones(15, dtype = int)
            errors[predicted == y_test] = 0
            correct = correct + 15 - np.sum(errors)
            for i in range(15):
                class_correctness[i] = class_correctness[i] + 1 - errors[i]

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Per class acuracy is: ')
    print((class_correctness / totiter) * 100)
if __name__ == "__main__":
    architecture = int(sys.argv[1])
    print(architecture)
    model = train_model(architecture)
    perform_test(model)
