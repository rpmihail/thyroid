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


# In[4]:

    
    
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
        label[0,4+echo] = 1
    if margins > 0:
        label[0,8+margins] = 1
    label[0,12+calc] = 1
    label[0, 15] = t_type
    return label

# 0 1 2 3 |4 5 6 7 | 10 11 12 13 | 15 16


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
        
        
        

        
        sample = {"image": img_recon.detach().cpu().numpy(), "labels": labels[:, :15], "types" : labels[:, 15]}
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
        
        self.conv1 = nn.Conv2d(1, 16, 3)
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

################################################################################################################################


# In[7]:
import torchvision.models as models
from unet_models import UNet_encoder


use_cuda = torch.cuda.is_available()    

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
    training_set = thyroidDataset()
    training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)
    totiter = len(training_generator)
    if architecture == 0:
        print("Training VG16")
        model = models.vgg16(pretrained=False)
        model.classifier._modules['6'] = torch.nn.Linear(4096, 15)
    elif architecture == 1:
        print("Training Unet encoder")
        model = UNet_encoder(1, 3)
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
    training_set = thyroidDataset()
    training_generator = torch.utils.data.DataLoader(training_set, **parameters_test)
    
    test_set = thyroidDataset()
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


architecture = 3
print(architecture)
model = train_model(architecture)
perform_test(model)




