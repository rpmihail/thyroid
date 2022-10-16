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
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
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
            labels = [self.types[t_type]] * len(files)
            data_list = list(zip(files, labels))
            self.all_data.extend(data_list)
        random.shuffle(self.all_data)
        self.cases, self.types = zip(*self.all_data)
        print("number of data items:" + str(len(self.cases)))
            

    def __len__(self):
        return len(self.cases)
  
    def __getitem__(self, idx):
        labels = np.zeros(16, dtype = float)
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).getroot()
        for x in xml_data:
            if x.tag=='composition' and x.text is not None:
                composition = x.text
                labels[self.compositions[composition] - 1] = 1.0
            if x.tag=='echogenicity' and x.text is not None:
                echogenicity = x.text
                labels[self.echogenicities[echogenicity] + 3] = 1.0
            if x.tag=='margins' and x.text is not None:
                margin = x.text
                labels[self.margins[margin] + 7] = 1.0
            if x.tag=='calcifications' and x.text is not None:
                calcification = x.text
                labels[self.calcifications[calcification] + 11] = 1.0
        
        labels[15] = list(self.types)[idx]
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
    
def projection_simplex_sort(v, z=1):

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

class End2EndModel(nn.Module):
    def __init__(self, architecture, num_attribs=15, num_groups=5, y_size=1):
        super(End2EndModel, self).__init__()
        self.num_attribs = num_attribs
        self.num_groups = num_groups
        self.y_size = y_size
        if architecture == 0:
            print("Training VG16")
            self.basenet = models.vgg16(pretrained=False)
            self.basenet.classifier._modules['6'] = torch.nn.Linear(4096, self.num_attribs)
        else:
            print("Training Unet encoder")
            self.basenet = UNet_encoder(3, self.num_attribs)
        self.rescale = torch.nn.Sigmoid()
        self.fc1 = nn.Linear(self.num_attribs, self.num_groups)
        self.fc2 = nn.Linear(self.num_groups, self.y_size)

    def forward(self, x):
        attrib_preds = self.rescale(self.basenet(x))
        groups = self.fc1(attrib_preds)
        output = self.rescale(self.fc2(groups))
        return torch.cat((attrib_preds, output), dim=1)


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
    
    model = End2EndModel(architecture, num_attribs=15, num_groups=5, y_size=1)
    model = model.to(device)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(100):
        running_loss = 0.0
        model.train()
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            model.train(True)
            x_train = data["image"]
            y_train = data["labels"]
            x_train, y_train = (
                x_train.to(device),
                y_train.to(device),
            )
            optimizer.zero_grad()

        # forward + backward + optimize
            output = model(x_train)
            loss = criterion(output.float(), y_train.float())
            loss.backward()
            optimizer.step()
            model.fc1.weight.data = projection_simplex_sort(model.fc1.weight.data)
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
    test_set = thyroidDataset(split='test')
    test_generator = torch.utils.data.DataLoader(test_set, **parameters_test)
    totiter = len(test_generator)
    total = 0
    correct = 0
    class_correctness = np.zeros(16, dtype=int)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_generator), total=totiter):
            x_test = data["image"]
            y_test = data["labels"]
            x_test, y_test = (
                x_test.to(device),
                y_test.to(device),
            )
            output = model(x_test)
            predicted = output.data
            predicted = predicted.detach().cpu().numpy()
            predicted = predicted.flatten()
            y_test = y_test.detach().cpu().numpy()
            y_test = y_test.flatten()
            predicted[predicted < 0.5] = 0.0
            predicted[predicted >= 0.5] = 1.0
            total += 16
            errors = np.ones(16, dtype = int)
            errors[predicted == y_test] = 0
            correct = correct + 16 - np.sum(errors)
            for i in range(16):
                class_correctness[i] = class_correctness[i] + 1 - errors[i]
            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Per class acuracy is: ')
    print((class_correctness / totiter) * 100)
if __name__ == "__main__":
    architecture = int(sys.argv[1])
    print(architecture)
    model = train_model(architecture)
    perform_test(model)
