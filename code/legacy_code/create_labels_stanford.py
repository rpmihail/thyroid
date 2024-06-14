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
import pandas as pd

all_data = []
file1 = open('/home/ahana/Stanford_thyroid/thyroidultrasoundcineclip/metadata.csv', 'r')
all_data = file1.readlines()
all_data = all_data[1:]
random.shuffle(all_data)
cases = all_data
print("number of data items:" + str(len(cases)))
all_labels = []
tot_count = 0
count = np.zeros((14, 14), dtype=int)
for idx in range(len(cases)):
    labels = np.zeros(21, dtype = int)
    data = cases[idx].split(',')
    case_num = data[0]
    labels[int(data[7])] = 1
    labels[15] = int(data[7])
    labels[int(data[8]) + 3] = 1
    labels[16] = int(data[8])
    if int(data[10]) == 0:
        labels[int(data[10]) + 7] = 1
        labels[17] = int(data[10])
    elif int(data[10]) > 0:
        labels[int(data[10]) + 6] = 1
        labels[17] = int(data[10]) - 1
    labels[int(data[11]) + 10] = 1
    labels[18] = int(data[11])
    labels[14] = int(data[13])
    labels[20] = int(data[12])
    if int(data[9]) == 3:
        labels[19] = 1
    elif int(data[9]) == 0:
        labels[19] = 0
    
    for i in range(0, 14):
        if labels[i] > 0:
            for j in range(0, 14):
                if labels[j] > 0 and i!=j:
                    count[i][j] += 1
    print(case_num + ', ' +(str(labels.tolist()).replace('[','')).replace(']',''))
    tot_count += 1

    all_labels.append(labels.tolist())
#count = np.round(count /390, 2)


dataset = pd.DataFrame({'cystic': count[:, 0], 'mixed': count[:, 1], 'solid': count[:, 2], 'anechoic':  count[:, 3], 'hyper/iso': count[:, 4], 'hypo': count[:, 5], 'very hypo': count[:, 6], 'ill-defined/smooth': count[:, 7], 'lobulated': count[:, 8], 'e-T extension': count[:, 9],'non': count[:, 10],  'macro': count[:, 11], 'rim': count[:, 12], 'micro': count[:, 13]})
dataset.insert(0, "attribs", ['cystic', 'mixed', 'solid',  'anechoic', 'hyper',  'hypo', 'marked hypo', 'ill-defined/smooth', 'lobulated', 'E-T extension', 'non', 'macro', 'rim', 'micro'], True)
print(dataset.to_string())
print("valid data items=" + str(tot_count))
       
def find_conditional_probability(dep_var_index):
    cond_var = []
    num_classes = [3, 4, 3, 4]
    for i in range(15, 19):
        if i!= dep_var_index:
            cond_var.append(i)
    
    for j in range(num_classes[cond_var[0] - 15]):
        for k in range(num_classes[cond_var[1] - 15]):
            for l in range(num_classes[cond_var[2] - 15]):
                deno = 0
                joint_ps = []
                for i in range(num_classes[dep_var_index - 15]):    
                    joint_p = 0
                    for labels in all_labels:
                        if labels[cond_var[0]] == j and labels[cond_var[1]] == k and  labels[cond_var[2]] == l and  labels[dep_var_index] == i:
                            joint_p += 1
                            #print(labels, joint_p)
                            deno += 1
                    joint_ps.append(joint_p)
                    #if joint_p > 0:
                    #    print("dep:" + str(i), "cond:" + str(j) + " " + str(k) + " " +str(l) + " " +str(joint_p))

                if deno > 0:
                    print("cond:" + str(j) + " " + str(k) + " " +str(l) + " " +str([str(joint_ps[i] / deno) for i in range(len(joint_ps))]))

print("dep_var: comp")
find_conditional_probability(15)
print("dep_var: echo")
find_conditional_probability(16)
print("dep_var: margin")
find_conditional_probability(17)
print("dep_var: calc")
find_conditional_probability(18)

    



'''        
#im_name = list(self.cases[idx].glob('*[0-9].jpg'))[0]
        #im = cv2.imread(str(im_name))
        #im = cv2.resize(im, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
        # Adding data augmentation to avoid overfitting
       # if random.randint(1, 10) > 5:
       #     im = np.flipud(im)
        #if random.randint(1, 10) > 5:
        #    im = np.fliplr(im)
        #if random.randint(1, 10) > 5:
        #    for i in range(random.randint(1, 4)):
        #        im = np.rot90(im)
        #im = np.ascontiguousarray(im)
        #transforms = Compose([ToTensor()])
        #im = transforms(im)
        #sample = {"image": im, "labels": torch.from_numpy(labels)}
        #return sample


'''
