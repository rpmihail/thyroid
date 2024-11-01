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
import plotly
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
import sys

# In[4]:

n_attributes = 15
expt_num = sys.argv[1]
    
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
        self.split = split
        with open('labels_list_' + split + '_gen1.txt') as f:
            self.samples = f.read().splitlines()
        if len(self.samples) > 73:
            self.samples = self.samples[:73]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_list = self.samples[idx].split(', ')
        name = data_list[0]
        labels = data_list[1:]
        labels = [float(element) for element in labels]
        labels = np.array(labels)
        im_name = os.path.join('/home/ahana/thyroid/code/generated_images1', name + '.jpg')
        im_masked = cv2.imread(str(im_name))
        #im_masked = cv2.resize(im_masked, dsize=(252, 252), interpolation=cv2.INTER_CUBIC)
        im_masked = im_masked[:,:,0]
        if int(name) < 100:
            im_name = os.path.join('/home/ahana/thyroid/data/orig', name + '.jpg')
            im1 = cv2.imread(str(im_name))
            #im = np.zeros((400, 400, 3))
            #im[38:353,:,:] = im1[:, 81:481,:]
            im = np.zeros((252, 252, 3))
            #im[122:437,:,:] = im1[:,:,:]
            #im[122:437,60:500,:] = im1[:,60:500,:]
            im1 = cv2.resize(im1, dsize=(252, 252), interpolation=cv2.INTER_CUBIC)
            im[:,40:210,:] = im1[:,40:210,:]
        else:
            im = np.zeros((252, 252, 3))
        
        im[:, :, 1] = im_masked
        cv2.imwrite(f'heatmaps/{name}_1.png', im)
        
        im = im[:,:,:2]
        # Adding data augmentation to avoid overfitting
        aug = True
        if self.split != "test" and aug == True:
            if random.randint(1, 10) > 5:
                if random.randint(1, 10) > 5:
                    im = np.flipud(im)
                else: #elif random.randint(1, 10) > 5:
                    im = np.fliplr(im)
                #elif random.randint(1, 10) > 5:
                #    for i in range(random.randint(1, 4)):
                #        im = np.rot90(im)
                #else:
                #    axis = random.randint(0, 1)
                #    direction = random.randint(0, 1)
                #    if direction == 0:
                #        direction = -1
                #    shift = random.randint(100, im.shape[0] // 2) * direction
                    #print(f'Random shift: {shift} {axis} {name}')
                #    im = np.roll(im, shift, axis=axis)
                    #cv2.imwrite(f'heatmaps/{name}.png', im)
            im = np.ascontiguousarray(im)
        transforms = Compose([ToTensor()])
        im = transforms(im)
        im_masked = transforms(im_masked)
        labels =  torch.from_numpy(labels)
        labels = torch.unsqueeze(labels, 0)
        #for i in range(0,15):
        #    if labels[0][i] == 1:
        #        print((i % 4) + 1, end=',')
        #print()
        #cv2.imwrite(f'heatmaps/{name}.png', im.numpy()[0,:,:])
        im = im.float()
        #cv2.imwrite(f'heatmaps/{name}.png', im.numpy()[0,:,:])
        sample = {"image": im.float(), "im_masked": im_masked, "labels": labels[:,:15], "type": labels[:,15], "filename": name}
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
        
        ###self.resnet = models.resnet50(pretrained=True) # pretrained=False just for debug reasons
        
        self.model = models.vgg16(pretrained=True) # pretrained=False just for debug reasons
        '''
        self.conv_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.model = nn.Sequential(self.conv_layer, self.resnet)
        '''
        model_layers = [nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        model_layers.extend(list(self.model.features))
        self.model.features= nn.Sequential(*model_layers)
        #self.model.classifier._modules['6'] = torch.nn.Linear(4096, 15)
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = torch.nn.Sequential(torch.nn.Linear(512, 15))
        print(self.model)
        '''
        num_ftrs = self.model._modules['1'].fc.in_features
        self.model._modules['1'].fc = nn.Linear(num_ftrs, 15)
        print(self.model)
        '''
        self.rescale = torch.nn.Sigmoid()
        self.fc1 = nn.Linear(n_attributes, 1, bias = False)

        self.fc1.weight.data = self.fc1.weight.data.relu() / (
                            self.fc1.weight.data.relu().sum(1, keepdim=True))

        '''
        self.fc2 = nn.Linear(n_groups, 1, bias = False)
        self.fc1.weight.data = self.fc1.weight.data.relu() / (
            self.fc1.weight.data.relu().sum(1, keepdim=True))
        self.fc2.weight.data.fill_(0)
        '''
        #type_classifier_layers = [nn.Linear(n_attributes, n_groups, bias = False),
        #        nn.Linear(n_groups, 1, bias = False)]
        #type_classifier = nn.Sequential(*type_classifier_layers) 
        #self.G_ = torch.nn.Parameter(G)
        #self.W_ = torch.nn.Parameter(W)
        #self.CNN_ = torch.nn.Parameter(CNN)
        


    def forward(self, x):
            
        attributes = self.rescale(self.model(x))
        #data = torch.cat((attributes, torch.squeeze(y, 1)), 1).float()
        #attrib_thres = self.threshold(attributes)
        y = self.fc1(attributes)
        '''
        y = self.fc2(y)
        '''
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


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

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
    training_set = thyroidActualDataset('training')
    #training_set = thyroidDataset()
    training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)
    totiter = len(training_generator)
    print("Training")
    model = net()

    model = model.to(device)
    print(model)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(1):
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
            add_factor = model.fc1.weight.data.min()
            model.fc1.weight.data = model.fc1.weight.data - add_factor
            model.fc1.weight.data = model.fc1.weight.data / model.fc1.weight.data.sum()
            
            running_loss += loss.item()
            loss1_sum += loss1.item()
            loss2_sum += loss2.item()
        print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / totiter))
        print(loss1_sum, loss2_sum, running_loss)
        
    print("Training complete")
    torch.save(model.state_dict(), f'../data/models/end_to_end.pt')
    G = model.fc1.weight.data
    G = G.cpu().numpy()
    a = n_attributes
    z = 1
    source = [i % a for i in range(z*a)]
    target = [(i // a) + a for i in range(z*a)]
    '''
    '''
    value = G.flatten().tolist()
    color_node = [
              #'#808080', '#808080', '#808080', '#808080', '#808080',
              #'#808080', '#808080', '#808080', '#808080', '#808080',
              #'#808080', '#808080', '#808080', '#808080', '#808080',
              '#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#FF00FF',
              '#00CED1', '#FF8C00', '#BDB76B', '#2F4F4F', '#B8860B',
              '#149697', '#00AAC0', '#8080CC', '#333388', '#77CCCC'
              ]
    color_link = []
    link_pos_colors = ['#149697', '#00AAC0', '#8080CC', '#333388', '#77CCCC', '#149697', '#00AAC0', '#8080CC', '#333388', '#77CCCC']
    link_neg_colors = ['#E65662', '#FF8066', '#894E3F', '#E79598', '#FDA45C', '#E65662', '#FF8066', '#894E3F', '#E79598', '#FDA45C']

    color_link = ['#5d8aa8','#ffbf00','#e52b50','#9966cc','#4b5320','#0d98ba','#480607','#cc5500','#702963','#008b8b','#a52a2a','#004b49','#b8860b','#ff4040','#00009c']
    label = []
    #for i in range(len(source)):
    #    if value[i] < 0:
    #        color_link.append(link_neg_colors[i%10])
    #    else:
    #        color_link.append(link_pos_colors[i%10])

    value = np.abs(value).tolist()
    attrib_names = ["cystic composition", "mostly solid composition", "solid composition", "spongiform composition",
            "hyperechogenicity", "hypoechogenicity", "isoechogenicity", "marked hypoechogenicity",
            "ill-defined margin", "micro margin", "spiculated margin", "smooth margin",
            "macro calcification", "micro calcification", "no calcification"]
    label = []
    for i in range(len(source)):
        label.append(f'{value[i]:.2f} | {attrib_names[i]}')
    label.append("benign/ malignant")

    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          color = color_node,
          label = label
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          color = color_link
      ))])
    fig.update_layout(title_text="Basic Sankey Diagram", font_size=18)
    fig.show()
    plotly.offline.plot(fig, filename=f'plot_1fc_CAM_{expt_num}.html')
    #fig.write_image('plot1.png')
    #fig.show()
    #plt.savefig("plot1.png")
    #fig.show()
    #plt.savefig('plot2.png')
    #fig.write_image('plot2.png')
    return model

def returnCAM(feature_conv, weight_linear, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_linear[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        if (np.max(cam)) != 0:
            cam_img = cam / (np.max(cam))
        else:
            cam_img = cam
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam

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
    FEATS = []
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    model.model.features.register_forward_hook(get_features('feats'))
    ###model.resnet.layer4.register_forward_hook(get_features('feats'))
    weight_linear = np.squeeze(model.model.classifier[0].weight.detach().cpu().numpy())
    ###weight_linear = np.squeeze(model.resnet.fc.weight.detach().cpu().numpy())
    attrib_sum_weights = [0, 2, 2, 0, 1, 2, 1, 3, 0, 2, 2, 0, 1, 3, 0]
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_generator), total=totiter):
            x_test = data["image"]
            y_test = data["labels"]
            y_type = data["type"]
            file_name = data["filename"][0]
            x_test, y_test, y_type = (
                x_test.to(device),
                y_test.to(device),
                y_type.to(device)
            )
            output, pred_type = model(x_test)
            FEATS.append(features['feats'].cpu().numpy())
            attrib_list = [i for i in range(0, n_attributes)]
            # saving the CAM image with heatmaps
            CAMs = returnCAM(FEATS[-1], weight_linear, attrib_list)
            img = np.zeros((252, 252, 3))
            gray_img = data["image"].cpu().numpy()
            #gray_img = gray_img.reshape((252, 252, 2))
            gray_img = gray_img[0,0,:,:]
            #cv2.imwrite(f'heatmaps/{file_name}.png', gray_img)
            #gray_img = gray_img * 255
            #gray_img = cv2.resize(gray_img, dsize=(252, 252), interpolation=cv2.INTER_CUBIC)
            img[:, :, 0] = gray_img
            img[:, :, 1] = gray_img
            img[:, :, 2] = gray_img
            #img = img * 255
            height, width, c = img.shape
            #for i in range(0, n_attributes):
            #    if 
            #    heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)
            #    result = heatmap * 0.5 + img
            #    cv2.imwrite(f'heatmaps/{file_name}_{i}.png', result)
            predicted = output.data
            predicted = predicted.detach().cpu().numpy()
            predicted = predicted.flatten()
            pred_type = pred_type.detach().cpu().numpy()
            pred_type = pred_type.flatten()
            y_type = y_type.detach().cpu().numpy()
            y_type = y_type.flatten()
            y_test = y_test.detach().cpu().numpy()
            y_test = y_test.flatten()
            #print(predicted, pred_type, y_test, y_type)
            prob_pred_sum = 0
            for i in range(n_attributes):
                prob_pred_sum += predicted[i] * attrib_sum_weights[i]
            print(predicted, y_test)
            predicted[predicted < 0.5] = 0.0
            predicted[predicted >= 0.5] = 1.0
            pred_type[pred_type < 0.5] = 0.0
            pred_type[pred_type >= 0.5] = 1.0
            predicted_sum = 0
            actual_sum = 0
            for i in range(n_attributes):
                predicted_sum += predicted[i] * attrib_sum_weights[i]
                actual_sum += y_test[i] * attrib_sum_weights[i]

            print(f'Predicted probability: {prob_pred_sum} Predicted sum  {predicted_sum} Actual sum {actual_sum} Actual y {y_type} Predicted y {pred_type}')
            print(predicted, y_test)
            total += 15
            type_total += 1
            pred_type = pred_type.astype(int)
            y_type = y_type.astype(int)
            y_test = y_test.astype(int)
            predicted = predicted.astype(int)
            errors = np.ones(15, dtype = int)
            errors[predicted == y_test] = 0
            correct = correct + 15 - np.sum(errors)
            for i in range(0, n_attributes):
                if predicted[i] != y_test[i] or y_test[i] != 1:
                    continue
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)
                result = img + heatmap * 0.4
                img_m = np.zeros((252, 252, 3))
                gray_img_m = data["im_masked"].cpu().numpy()
                img_m[:, :, 0] = gray_img_m
                img_m[:, :, 1] = gray_img_m
                img_m[:, :, 2] = gray_img_m
                img_m = img_m * 255
                result = cv2.hconcat([result, img_m])
                h_file_name = f'heatmaps{expt_num}/{file_name}_{i}.png'
                cv2.imwrite(h_file_name, result)
            if y_type[0] == pred_type[0]:
                type_correct += 1
            for i in range(15):
                class_correctness[i] = class_correctness[i] + 1 - errors[i]
            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    accuracy = 100 * correct / total
    print('Per class acuracy is: ')
    print((class_correctness / totiter) * 100)
    print('Type accuracy: %d %%' %(100 * type_correct / type_total))
    type_acc = 100 * type_correct / type_total
    FEATS = np.concatenate(FEATS)
    print('- feats shape:', FEATS.shape)
    with open("results_unmasked_fc1.txt", "a") as file1:
        file1.write(str(type_acc) + '  ' + str(accuracy) + '\n')

model = train_model()
# perform_test(model, thyroidDataset())
perform_test(model, thyroidActualDataset('test'))
#perform_test(model, thyroidActualDataset('training'))



