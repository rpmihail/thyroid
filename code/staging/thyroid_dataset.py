#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:29:59 2024

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
import json
import cv2
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from patchify import patchify





# dataset definition
class thyroidDataset(Dataset):
    def __init__(self, split):
        self.all_data = []
        self.compositions = {'Unknown':0, 'cystic':1,
                             'predominantly solid':2,
                             'solid': 3, 'spongiform appareance': 4}
        
        self.echogenicities = {'Unknown':0, 'hyperechogenecity':1,
                             'hypoechogenecity':2, 'isoechogenicity':3,
                             'marked hypoechogenecity': 4}
        
        self.margins = {'Unknown':0, 'ill- defined':1, 'microlobulated':2,
                        'spiculated':3, 'well defined smooth':4}
        
        self.calcifications = {'macrocalcification':0, 'microcalcification':1, 'non':2}
        
        self.types = {'benign':0, 'malign':1}
        
        self.im_size = (256, 256)
        
        self.patch_size = 16
        
        self.scores = [0, 1, 1, 2, 0,    0, 1, 2, 1, 3,    0, 0, 2, 2, 3,    1, 3, 2, 3,   50]
        
        self.types_count = []
        
        for t_type in ['benign', 'malign']:
            root_dir=Path('../../data/' + split + '/' + t_type).expanduser().resolve().absolute() 
            
            files = list(root_dir.glob("*"))
            labels = [self.types[t_type]] * len(files)
            self.types_count.append(len(files))
            data_list = list(zip(files, labels))
            self.all_data.extend(data_list)
        

        #print(self.all_data)        
        random.shuffle(self.all_data)
        
        
        
        self.cases, self.types = zip(*self.all_data)
        #print("number of data items:" + str(len(self.cases)))
        self.sample_weights = [1/self.types_count[label] for label in self.types]
    def __len__(self):
        return len(self.cases)
    
    
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
        labels = np.zeros(20, dtype = float)
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).getroot()
        for x in xml_data:
            if x.tag=='composition' and x.text is not None:
                composition = x.text
                labels[self.compositions[composition] ] = 1.0
            if x.tag=='echogenicity' and x.text is not None:
                echogenicity = x.text
                labels[self.echogenicities[echogenicity] + 5] = 1.0
            if x.tag=='margins' and x.text is not None:
                margin = x.text
                labels[self.margins[margin] + 10] = 1.0
            if x.tag=='calcifications' and x.text is not None:
                calcification = x.text
                labels[self.calcifications[calcification] + 15] = 1.0
        
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).find("mark")
        
        for x in xml_data:
            if(x.tag=='svg'):
                encoded = str(x.text)
                poly_data = json.loads(x.text)
        
        if list(self.types)[idx] == 1:
            labels[19] = 1
            
        
        
        
        im_name = list(self.cases[idx].glob('*[0-9].jpg'))[0]
        im = cv2.imread(str(im_name))[:, :, 0]
        mask = np.zeros(np.shape(im))
        im = cv2.resize(im, dsize=self.im_size, interpolation=cv2.INTER_CUBIC) / 256.0
        
        # add mask 
        for polygon in poly_data:
            xs = []
            ys = []
            for point in polygon["points"]:
                xs.append(point["x"])
                ys.append(point["y"])
            contour = np.concatenate((np.expand_dims(xs, 1), np.expand_dims(ys, 1)), axis=1)
            cv2.fillPoly(mask, pts = [contour], color =(1, 1, 1))
        
        #mask = cv2.resize(mask, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
        
        mask = cv2.resize(mask, dsize=self.im_size, interpolation=cv2.INTER_LINEAR)

        #im = im * mask

        
        im = self.translate(im, np.random.randint(-20, 0), np.random.randint(0, 4))

        
        
        # Adding data augmentation to avoid overfitting
        
        if random.randint(1, 10) > 5:
            im = np.flipud(im)
        if random.randint(1, 10) > 5:
            im = np.fliplr(im)
        if random.randint(1, 10) > 5:
            for i in range(random.randint(1, 4)):
                im = np.rot90(im)
        im = np.ascontiguousarray(im)

        #plt.figure()
        #plt.imshow(im)
        
        
        #patches = patchify(im, (self.patch_size, self.patch_size), self.patch_size)
        #tensor_patches = np.reshape(patches, (self.patch_size**2, self.patch_size*self.patch_size) )
        
        #tensor_patches = torch.FloatTensor(tensor_patches)

        transforms = Compose([ToTensor()])
        mask = transforms(mask)
        im = transforms(im)
        
        #print(torch.from_numpy(labels.dot(self.scores)))
        
        im = im.type(torch.FloatTensor)
        #sample = {"image": im, "mask": mask, "patches": tensor_patches, "labels": torch.from_numpy(labels), "types" : self.types, "name": str(im_name)}
        sample = {"images": im, "mask": mask, "labels": torch.from_numpy(labels), "scores": torch.from_numpy(np.array(labels.dot(self.scores))), "types" : self.types, "name": str(im_name)}
        #sample = {"images": tensor_patches, "labels": torch.from_numpy(np.array(int(labels[15])))}
        return sample
    