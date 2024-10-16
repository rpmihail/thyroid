#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:58:54 2024

@author: mihail
"""

import torch
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import torch.nn.functional as F
import matplotlib.pyplot as plt



from thyroid_dataset import *
from patchify import patchify
    
batch_size = 32
    
# Dataset creation
parameters_train = {
    "batch_size": batch_size,
    #"shuffle": True,
}

parameters_test = {
    "batch_size": 1,
    "shuffle": False,
}

training_set = thyroidDataset(split='train')
training_generator = torch.utils.data.DataLoader(training_set, **parameters_train, sampler=torch.utils.data.WeightedRandomSampler(training_set.sample_weights, len(training_set.cases), replacement=True))

testing_set = thyroidDataset(split='test')
testing_generator = torch.utils.data.DataLoader(testing_set, **parameters_test, sampler=torch.utils.data.WeightedRandomSampler(testing_set.sample_weights, len(testing_set.cases), replacement=True))

# %%

## test dataloader
patch_nr = 0
for item in training_generator:
    continue
    break
# %%


# %%



step = 16
device = "cuda"

import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        
        # input, image
        self.conv1 = nn.Conv2d(1, 64, 3, 2)
        self.relu1 = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, 2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, 2)
        self.relu3 = nn.LeakyReLU(0.2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.relu4 = nn.LeakyReLU(0.2)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, 2)
        self.relu5 = nn.LeakyReLU(0.2)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, 2)
        self.relu5 = nn.LeakyReLU(0.2)
        self.bn5 = nn.BatchNorm2d(64)

        
        self.fc4 = nn.Linear(3136, 256) # shrink this
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 2)
        
        
        

    def forward(self, im):
        x = self.conv1(im); x = self.relu1(x); x = self.bn1(x)
        x = self.conv2(x); x = self.relu2(x); x = self.bn2(x)
        x = self.conv3(x); x = self.relu3(x); x = self.bn3(x)        
        x = self.conv4(x); x = self.relu4(x); x = self.bn4(x)        
        x = self.conv5(x); x = self.relu5(x); x = self.bn5(x)        
        
        x = x.view(np.shape(x)[0], np.shape(x)[1]*7*7)
        
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
       
        return x


net = Net()

learning_rate = 0.001

import torch.optim as optim


    

#criterion = nn.L1Loss(reduction='batchmean') 


optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

losses = []



# %%67

model = Net().to(device)

x = torch.randn(1, 1, 256, 256).to(device) # Dummy images
print(np.shape(model(x))) # torch.Size([7, 49, 16])

N_EPOCHS = 100000
LR = 0.0005

# %%

#checkpoint = torch.load("../../data/models/transformer_v10.pt")
#model.load_state_dict(checkpoint)


#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#torch.load(model.state_dict(), "../../data/models/transformer_v1.pt")

# %%
optimizer = Adam(model.parameters(), lr=LR)

# Training loop

model.train()

for epoch in range(N_EPOCHS):
    train_loss = 0.0
    for batch in training_generator:
        
        x, y = batch["images"], batch["labels"]
        x, y = x.to(device), y.to(device)
        
        y_hat = model(x)
        loss = torch.sum(torch.abs(torch.sum(torch.pow(y_hat, 2)) - torch.pow(batch["scores"].to(device), 0.5)))

            

        train_loss += loss.detach().cpu().item() / len(training_generator)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.10f}", end="\n")
        


torch.save(model.state_dict(), "../../data/models/stager.pt")

# %% test set
test_loss = 0

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

model.eval()

coords = np.zeros((0, 2))
scores = np.zeros((0))


for batch in training_generator:
    total += 1
    x, y = batch["images"], batch["scores"]
    x, y = x.to(device), y.to(device)
       
    y_hat = model(x)
    
    coords = np.concatenate((coords, y_hat.detach().cpu().numpy()))
    scores = np.concatenate((scores, y.detach().cpu().numpy()))
    #print(y_hat)
    

cols = (scores>50) * 1
plt.scatter(coords[:, 0], coords[:, 1], c = cols)


