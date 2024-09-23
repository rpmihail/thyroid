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

import matplotlib.pyplot as plt



from thyroid_dataset import *
from patchify import patchify
    
    
# Dataset creation
parameters_train = {
    "batch_size": 32,
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
fig, ax = plt.subplots(16, 16, figsize=(10, 10))
patch_nr = 0
for item in training_set:
  for i in range(16):
      for j in range(16):
          ax[i, j].imshow(np.reshape(item["patches"][patch_nr, :], (16, 16)), cmap="gray")
          ax[i, j].xaxis.set_ticks([])
          ax[i, j].yaxis.set_ticks([])
          #ax[i, j].yaxis.set_visible("false")
          patch_nr += 1
  #ax[1].imshow(item["image"][0, :, :] + item["mask"][0, :, :], cmap="gray")
  break
# %%


# %%


step = 16

for item in training_generator:
    im = item["patches"]
    #patches = patchify(im, (16, 16), 16)
    break


#fig, ax = plt.subplots(16, 16, figsize=(10, 10))
#for i in range(16):
#    for j in range(16):
#        ax[i, j].imshow(patches[i, j, :, :])
        
#tensor_patches = np.reshape(patches, (16*16, step*step) )





device = "cuda"

import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result



class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
  def __init__(self):
    # Super constructor
    super(MyViT, self).__init__()

    # Attributes
    self.n_patches = 256
    
    self.hidden_d = 16
    
    self.n_blocks = 16
    self.n_heads = 16
    
    self.out_d = 2

    #assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    #assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    
    #self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
    self.patch_size = (16, 16)

    # 1) Linear mapper
    self.input_d = 256
    
    
    self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

    # 2) Learnable classifiation token ### FIXME MAYBE
    self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
    
    
    # 3) Positional embedding
    self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches + 1, self.hidden_d)))
    self.pos_embed.requires_grad = False
    
    
    self.blocks = nn.ModuleList([MyViTBlock(self.hidden_d, self.n_heads) for _ in range(self.n_blocks)])
    
    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self.hidden_d, self.out_d),
        nn.Softmax(dim=-1)
    )


  def forward(self, images):
    #patches = patchify(images, self.n_patches)
    tokens = self.linear_mapper(images)
    
    # Adding classification token to the tokens
    tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
    
    # Adding positional embedding
    pos_embed = self.pos_embed.repeat(np.shape(tokens)[0], 1, 1)

    out = tokens + pos_embed
    
    # Transformer Blocks
    for block in self.blocks:
        out = block(out)
        
    # Getting the classification token only
    out = out[:, 0]
    
    
    return self.mlp(out)



# %%

model = MyViT( ).to(device)

#x = torch.randn(7, 256, 256) # Dummy images
#print(model(x)) # torch.Size([7, 49, 16])

N_EPOCHS = 100000
LR = 0.0001

# %%

    # Training loop
optimizer = Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
for epoch in range(N_EPOCHS):
    train_loss = 0.0
    for batch in training_generator:
        
        x, y = batch["patches"], batch["labels"]
        x, y = x.to(device), y.to(device)
        
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(training_generator)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}", end="\n")
        if(epoch % 5 == 0):
            print(np.concatenate((y_hat.detach().cpu().numpy(),  np.expand_dims(y.detach().cpu().numpy(), -1)), axis=1))


# %%

torch.save(model.state_dict(), "../../data/models/transformer_v1.pt")

# %% test set
test_loss = 0

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
correct = 0
total = 0
for batch in testing_generator:
    total += 1
    x, y = batch["patches"], batch["labels"]
    x, y = x.to(device), y.to(device)
    
    y_hat = model(x)
    loss = criterion(y_hat, y)

    if y[0] == y_hat.argmax():
        correct += 1

    test_loss = loss.detach().cpu().item() #/ len(training_generator)


    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()

    x = np.concatenate((y_hat.detach().cpu().numpy(),  np.expand_dims(y.detach().cpu().numpy(), -1)), axis=1)
    
    print(x)
    print(f"Loss: {test_loss}\n")

print("Accuract:", correct / total)


