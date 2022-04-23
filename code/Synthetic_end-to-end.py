#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import plotly.graph_objs as go
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
import json
from torch.utils.data import Dataset
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import os

# In[4]:

n_attributes = 15
n_groups = 192
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim + 15 + 1,  fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 3 * 3 * fc2_input_dim),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(fc2_input_dim, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(fc2_input_dim, 128, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,  padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


# In[5]:


torch.manual_seed(0)

### Initialize the two networks
d = 16

decoder = Decoder(encoded_space_dim=d,fc2_input_dim=512)
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

decoder.to(device)


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


# In[2]:

decoder.load_state_dict(torch.load("decoder.dict"))  


    
# In[3]:

import torchvision
from torchvision import transforms
import random



def generate_label():
    
    label = np.zeros((1, 16))
    
    with open('combinations.txt') as file:
        lines = [line.rstrip() for line in file]
        
    train_data = [line.split(',') for line in lines]
    #print(train_data)
    
    selected_example = np.random.randint(len(train_data))
    comp = int(train_data[selected_example][0])
    echo = int(train_data[selected_example][1])
    margins = int(train_data[selected_example][2])
    calc = int(train_data[selected_example][3])
    t_type = int(train_data[selected_example][4])
    #comp = np.random.randint(5)
    #echo = np.random.randint(5)
    #margins = np.random.randint(5)
    #calc = np.random.randint(3)
    
    if comp > 0:
        label[0,comp-1] = 1
    if echo > 0:
        label[0,3+echo] = 1
    if margins > 0:
        label[0,7+margins] = 1
    label[0,12+calc] = 1
    label[0, 15] = t_type
    return label

# 0 1 2 3 |4 5 6 7 | 8 9 10 11 |12 13 14


print(generate_label())
    


# dataset definition
class thyroidDataset(Dataset):
    def __init__(self):
        self.num_cases = 512
    
    def __len__(self):
        return self.num_cases
  
    def __getitem__(self, idx):
        labels = torch.FloatTensor(generate_label()).to(device)
        latent = torch.FloatTensor(np.random.randn(1, d)).to(device)
        decoder_input = torch.cat((latent, labels), dim = 1)
        img_recon = torch.unsqueeze(torch.squeeze(decoder(decoder_input)), 0)
        
        
        sample = {"image": img_recon.detach().cpu().numpy(), "labels": labels[:, :15], "type" : labels[:, 15]}
        return sample




class thyroidActualDataset(Dataset):
    def __init__(self, split):
        with open('labels_list_' + split + '_gen.txt') as f:
            self.samples = f.read().splitlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_list = self.samples[idx].split(', ')
        name = data_list[0]
        labels = data_list[1:]
        labels = [float(element) for element in labels]
        labels = np.array(labels)
        im_name = os.path.join('/home/ahana/thyroid/code/generated_images1', name + '.jpg')
        im = cv2.imread(str(im_name))
        im = cv2.resize(im, dsize=(252, 252), interpolation=cv2.INTER_CUBIC)
        im = im[:,:,0]
        # Adding data augmentation to avoid overfitting
        #if random.randint(1, 10) > 5:
        #    im = np.flipud(im)
        #if random.randint(1, 10) > 5:
        #    im = np.fliplr(im)
        #if random.randint(1, 10) > 5:
        #    for i in range(random.randint(1, 4)):
        #        im = np.rot90(im)
        #im = np.ascontiguousarray(im)
        transforms = Compose([ToTensor()])
        im = transforms(im)
        labels =  torch.from_numpy(labels)
        sample = {"image": im, "labels": labels[:15], "type": labels[15]}
        return sample

# Dataset creation
training_set = thyroidDataset()
parameters_train = {
    "batch_size": 32,
    #"shuffle": True,
}

training_set = thyroidDataset()
training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)



# In[6]:

#for data in training_generator: # with "_" we just ignore the labels (the second element of the dataloader tuple)
#     image_batch = data["image"]
#     print(np.shape(image_batch))



################################################################################################################################

class net(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = models.vgg16(pretrained=False) # pretrained=False just for debug reasons
        model_layers = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        model_layers.extend(list(self.model.features))
        self.model.features= nn.Sequential(*model_layers)
        self.model.classifier._modules['6'] = torch.nn.Linear(4096, 15)
        self.rescale = torch.nn.Sigmoid()
        self.fc1 = nn.Linear(n_attributes, n_groups, bias = False)
        self.fc2 = nn.Linear(n_groups, 1, bias = False)
        self.fc1.weight.data = self.fc1.weight.data.relu() / (
            self.fc1.weight.data.relu().sum(1, keepdim=True))
        self.fc2.weight.data.fill_(0)

        #type_classifier_layers = [nn.Linear(n_attributes, n_groups, bias = False),
        #        nn.Linear(n_groups, 1, bias = False)]
        #type_classifier = nn.Sequential(*type_classifier_layers) 
        #self.G_ = torch.nn.Parameter(G)
        #self.W_ = torch.nn.Parameter(W)
        #self.CNN_ = torch.nn.Parameter(CNN)
        


    def forward(self, x):
            
        attributes = self.rescale(self.model(x))
        y = self.fc1(attributes)
        y = self.fc2(y)
        
        #x = torch.unsqueeze(x, 2)
        #print(x.size())
        
        #g = torch.matmul(self.G_, x) 
        
        #g = g.repeat((1, 1, k))
        
        #y = g * self.W_
        
        #y, _ = y.max(axis=2)
        
        #y = torch.transpose(y, 1, 0)
        
        #y = torch.sum(y, axis=0)
        
        #return (torch.sigmoid(y), x, features)
        return attributes, torch.sigmoid(y)

################################################################################################################################


# In[7]:
import torchvision.models as models
from unet_models import UNet_encoder


use_cuda = torch.cuda.is_available()    

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
    training_set = thyroidDataset()
    training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)
    totiter = len(training_generator)
    print("Training custom model")
    model = net()

    model = model.to(device)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(200):
        running_loss = 0.0
        loss1_sum = 0.0
        loss2_sum = 0.0
        model.train()
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            model.train(True)
            x_train = data["image"]
            y_train = data["labels"]
            y_type = data["type"]
            x_train, y_train, y_type = (
                x_train.to(device),
                y_train.to(device),
                y_type.to(device)
            )
            optimizer.zero_grad()

        # forward + backward + optimize
            attributes, target = model(x_train)
            attributes_pred = torch.unsqueeze(attributes, 1)
            loss1 = criterion(attributes_pred.float(), y_train.float())
            loss2 = criterion(target.float(), y_type.float())
            loss = loss1 + loss2 
            loss.backward()
            optimizer.step()
            model.fc1.weight.data = projection_simplex_sort(model.fc1.weight.data)
            running_loss += loss.item()
            loss1_sum += loss1.item()
            loss2_sum += loss2.item()
        print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / totiter))
        print(loss1_sum, loss2_sum, running_loss)
        
    print("Training complete")
    torch.save(model.state_dict(), f'../data/models/end_to_end.pt')
    return model

def perform_test(model, dataset):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    parameters_test = {
        "batch_size": 1,
        "shuffle": False,
    }
    
    test_set = dataset
    test_generator = torch.utils.data.DataLoader(test_set, **parameters_test)
    
    totiter = len(test_generator)
    total = 0
    correct = 0
    type_total = 0
    type_correct = 0
    class_correctness = np.zeros(15, dtype=int)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_generator), total=totiter):
            x_test = data["image"]
            y_test = data["labels"]
            y_type = data["type"]
            x_test, y_test, y_type = (
                x_test.to(device),
                y_test.to(device),
                y_type.to(device)
            )
            output, pred_type = model(x_test)
            predicted = output.data
            predicted = predicted.detach().cpu().numpy()
            predicted = predicted.flatten()
            pred_type = pred_type.detach().cpu().numpy()
            pred_type = pred_type.flatten()
            y_type = y_type.detach().cpu().numpy()
            y_type = y_type.flatten()
            y_test = y_test.detach().cpu().numpy()
            y_test = y_test.flatten()
            predicted[predicted < 0.5] = 0.0
            predicted[predicted >= 0.5] = 1.0
            pred_type[pred_type < 0.5] = 0.0
            pred_type[pred_type >= 0.5] = 1.0
            total += 15
            type_total += 1
            pred_type = pred_type.astype(int)
            y_type = y_type.astype(int)
            y_test = y_test.astype(int)
            predicted = predicted.astype(int)
            errors = np.ones(15, dtype = int)
            errors[predicted == y_test] = 0
            correct = correct + 15 - np.sum(errors)
            if y_type[0] == pred_type[0]:
                type_correct += 1
            for i in range(15):
                class_correctness[i] = class_correctness[i] + 1 - errors[i]
            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Per class acuracy is: ')
    print((class_correctness / totiter) * 100)
    print('Type accuracy: %d %%' %(100 * type_correct / type_total))

model = train_model()
perform_test(model, thyroidDataset())
perform_test(model, thyroidActualDataset('test'))
perform_test(model, thyroidActualDataset('training'))



