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
compositions = {'Unknown':0, 'cystic':1, 'predominantly cystic':0,
        'predominantly solid':0, 'dense':0,
                'solid':3, 'spongiform appareance':1, 'spongiform': 1}
echogenicities = {'Unknown':0, 'hyperechogenicity':2, 'hyperechogenecity':2,
        'hypoechogenicity':3, 'hypoechogenecity':3, 'isoechogenicity':2,
        'marked hypoechogenicity':4, 'marked hypoechogenecity':4, 'anechoic': 1}
margins = {'Unknown':0, 'ill- defined':1, 'ill defined':1, 'microlobulated':2,
        'spiculated':2, 'well defined smooth':1, 'well defined': 1, 'macrolobulated': 2, 'extra-thyroidal extension': 3}
calcifications = {'macrocalcification':1, 'macrocalcifications':1, 'microcalcification':3, 'microcalcifications':3, 'non':0, 'rim-calcifications': 2}
types ={'benign':0, 'malign':1}
root_dir=Path('../../data/thyroid')
files = list(root_dir.glob("*.xml"))
data_list = list(files)
all_data.extend(data_list)
random.shuffle(all_data)
cases = all_data
print("number of data items:" + str(len(cases)))
all_labels = []
tot_count = 0
weights = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0]
weights = np.array(weights)
tirads = None
count = np.zeros((14, 14), dtype=int)
for idx in range(len(cases)):
    labels = np.zeros(19, dtype = int)
    xml_data = ET.parse(cases[idx]).getroot()
    any_none = False
    for x in xml_data:
        if x.tag=='composition': 
            if x.text is not None:
                composition = x.text
                if compositions[composition] > 0:
                    labels[compositions[composition] - 1] += 1.0
                    labels[15] = compositions[composition] - 1
                else:
                    any_none = True
            else:
                any_none = True
        if x.tag=='echogenicity':
            if x.text is not None:
                echogenicity = x.text
                if echogenicities[echogenicity] > 0:
                    labels[echogenicities[echogenicity] + 2] += 1.0
                    labels[16] = echogenicities[echogenicity] - 1
                else:
                    any_none = True

            else:
                any_none = True
        if x.tag=='margins':
            if x.text is not None:
                margin = x.text
                if margins[margin] > 0:
                    labels[margins[margin] + 6] += 1.0
                    labels[17] = margins[margin]  - 1
                else:
                    any_none = True
            else:
                any_none = True
        if x.tag=='calcifications':
            if x.text is not None:
                calcification = x.text
                labels[calcifications[calcification] + 10] += 1.0
                labels[18] = calcifications[calcification] 
            else:
                any_none = True
        if x.tag=='tirads':
            if x.text is not None:
                if int(x.text[0]) > 3:
                    labels[14] = 1
                tirads = x.text
            else:
                any_none = True
            #print(cases[idx], x.text)
    case_num = (str(cases[idx]).split('/')[-1]).split('.')[0]
    #if not any_none:
        #weighted = np.multiply(weights, labels)
        #print(weighted, np.sum(weighted), tirads)
        #if int(tirads[0]) > 3 and np.sum(weighted) <= 3.0:
            #print(weighted, np.sum(weighted), tirads)
            #print(case_num, 'sum too low', weighted, np.sum(weighted), tirads)
        #if int(tirads[0]) <= 3 and np.sum(weighted) > 3.0:
            #print(weighted, np.sum(weighted), tirads)
            #print(case_num, 'sum too high', weighted, np.sum(weighted), tirads)
    
    if not any_none:
        for i in range(0, 14):
            if labels[i] > 0:
                for j in range(0, 14):
                    if labels[j] > 0 and i!=j:
                        count[i][j] += 1
    if not any_none:
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
