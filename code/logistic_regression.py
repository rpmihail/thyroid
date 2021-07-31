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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

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
        self.cases, self.labels = zip(*self.all_data)
        print("number of data items:" + str(len(self.cases)))
            

    def __len__(self):
        return len(self.cases)
  
    def __getitem__(self, idx):
        composition = 'Unknown'
        echogenicity = 'Unknown'
        margin = 'Unknown'
        calcification = 'Unknown'
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).getroot()
        for x in xml_data:
            if x.tag=='composition' and x.text is not None:
                composition = x.text
            if x.tag=='echogenicity' and x.text is not None:
                echogenicity = x.text
            if x.tag=='margins' and x.text is not None:
                margin = x.text
            if x.tag=='calcifications' and x.text is not None:
                calcification = x.text
        features = np.zeros(4, dtype=np.float32)
        features[0] = self.compositions[composition]
        features[1] = self.echogenicities[echogenicity]
        features[2] = self.margins[margin]
        features[3] = self.calcifications[calcification]
        features = torch.from_numpy(features)
        sample = {"features": features,
                  "label": self.types[self.labels[idx]]}
        return sample

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.rescale = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.linear(x)
        output = self.rescale(output)
        return output
    
def train_model():
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
    input_dim = 4
    output_dim = 1
    model = LogisticRegression(input_dim, output_dim)
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(5000):
        running_loss = 0.0
        model.train()
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            model.train(True)
            x_train = data["features"]
            y_train = data["label"]
            x_train, y_train = (
                x_train.to(device),
                y_train.to(device),
            )
            y_train = y_train.reshape([y_train.shape[0], 1])
            optimizer.zero_grad()

        # forward + backward + optimize
            output = model(x_train)
            loss = criterion(output.float(), y_train.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / totiter))
        
    print("Training complete")
    torch.save(model.state_dict(), '../data/models/logistic.pt')
    return model

def perform_test(model):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    parameters_test = {
        "batch_size": 1,
        "shuffle": False,
    }
    test_set = thyroidDataset(split='test')
    test_generator = torch.utils.data.DataLoader(test_set, **parameters_test)
    totiter = len(test_generator)
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_generator), total=totiter):
            x_test = data["features"]
            y_test = data["label"]
            x_test, y_test = (
                x_test.to(device),
                y_test.to(device),
            )
            output = model(x_test)
            predicted = output.data
            predicted = predicted.detach().cpu().numpy()
            predicted[predicted < 0.5] = 0.0
            predicted[predicted >= 0.5] = 1.0
            total += 1
            predicted = predicted.flatten()
            print(predicted, y_test)
            for i in range(y_test.shape[0]):
                if predicted[i] == y_test[i]:
                    correct += 1
            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

if __name__ == "__main__":
    model = train_model()
    perform_test(model)
