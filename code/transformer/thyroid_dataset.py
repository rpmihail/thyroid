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
                             'solid':3, 'spongiform appareance':4}
        
        self.echogenicities = {'Unknown':0, 'hyperechogenecity':1,
                             'hypoechogenecity':2, 'isoechogenicity':3,
                             'marked hypoechogenecity':4}
        
        self.margins = {'Unknown':0, 'ill- defined':1, 'microlobulated':2,
                        'spiculated':3, 'well defined smooth':4}
        
        self.calcifications = {'macrocalcification':0, 'microcalcification':1, 'non':2}
        
        self.types = {'benign':0, 'malign':1}
        
        self.im_size = (256, 256)
        
        self.patch_size = 16
        
        
        self.types_count = []
        
        for t_type in ['benign', 'malign']:
            root_dir=Path('../../data/' + split + '/' + t_type).expanduser().resolve().absolute() 
            
            files = list(root_dir.glob("*"))
            labels = [self.types[t_type]] * len(files)
            self.types_count.append(len(files))
            data_list = list(zip(files, labels))
            self.all_data.extend(data_list)
        

        print(self.all_data)        
        random.shuffle(self.all_data)
        
        
        
        self.cases, self.types = zip(*self.all_data)
        print("number of data items:" + str(len(self.cases)))
        self.sample_weights = [1/self.types_count[label] for label in self.types]
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
        
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).find("mark")
        
        for x in xml_data:
            if(x.tag=='svg'):
                encoded = str(x.text)
                poly_data = json.loads(x.text)
        
        labels[15] = list(self.types)[idx]
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
        

        im = im * mask
        
        
        # Adding data augmentation to avoid overfitting
        #if random.randint(1, 10) > 5:
        #    im = np.flipud(im)
        #if random.randint(1, 10) > 5:
        #    im = np.fliplr(im)
        #if random.randint(1, 10) > 5:
        #    for i in range(random.randint(1, 4)):
        #        im = np.rot90(im)
        #im = np.ascontiguousarray(im)

        #plt.figure()
        #plt.imshow(im)
        
        
        patches = patchify(im, (self.patch_size, self.patch_size), self.patch_size)
        tensor_patches = np.reshape(patches, (self.patch_size**2, self.patch_size*self.patch_size) )
        
        tensor_patches = torch.FloatTensor(tensor_patches)

        transforms = Compose([ToTensor()])
        mask = transforms(mask)
        im = transforms(im)
        
        
        im = im.type(torch.FloatTensor)
        #sample = {"image": im, "mask": mask, "patches": tensor_patches, "labels": torch.from_numpy(labels), "types" : self.types, "name": str(im_name)}
        sample = {"patches": tensor_patches, "labels": torch.from_numpy(np.array(int(labels[15])))}
        return sample
    