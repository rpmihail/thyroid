'''
Code for training model with 2 layers for predicting type
'''

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
import torchvision
from torchvision import transforms
import random
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
import torchvision.models as models
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
n_groups = 4
expt_num = sys.argv[1]

    

def projection_simplex_sort(v, z=1):
    '''
    Used to nomalize the G matrix for mpping from attributes to groups
    '''

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


class thyroidActualDataset(Dataset):
    def __init__(self, split):
        with open('../labels_list_' + split + '_gen1.txt') as f:
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
        # Reading maske dimages
        im_name = os.path.join('/home/ahana/thyroid/code/generated_images1', name + '.jpg')
        im_masked = cv2.imread(str(im_name))
        im_masked = cv2.resize(im_masked, dsize=(252, 252), interpolation=cv2.INTER_CUBIC)
        im_masked = im_masked[:,:,0]
        # Reads actual images. If-else logic is leagacy and was used for cases where images were genereated.
        # Only the if code without condition is necessary now.
        if int(name) < 100:
            im_name = os.path.join('/home/ahana/thyroid/data/orig', name + '.jpg')
            im1 = cv2.imread(str(im_name))
            im = np.zeros((252, 252, 3))
            im1 = cv2.resize(im1, dsize=(252, 252), interpolation=cv2.INTER_CUBIC)
            im[:,40:210,:] = im1[:,40:210,:]
        else:
            im = np.zeros((252, 252, 3))
        
        # Creating 2 channel input
        im[:, :, 1] = im_masked
        im = im[:,:,:2]

        transforms = Compose([ToTensor()])
        im = transforms(im)
        im_masked = transforms(im_masked)
        labels =  torch.from_numpy(labels)
        labels = torch.unsqueeze(labels, 0)
        sample = {"image": im.float(), "im_masked": im_masked, "labels": labels[:,:15], "type": labels[:,15], "filename": name}
        return sample




class net(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = models.vgg16(pretrained=True) # pretrained=False just for debug reasons
        model_layers = [nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        model_layers.extend(list(self.model.features))
        self.model.features= nn.Sequential(*model_layers)
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # modifying VGG-16 for CAM
        self.model.classifier = torch.nn.Sequential(torch.nn.Linear(512, 15))
        self.rescale = torch.nn.Sigmoid()
        # Adding 2 layers for type prediction
        self.fc1 = nn.Linear(n_attributes, n_groups, bias = False)
        self.fc2 = nn.Linear(n_groups, 1, bias = False)
        self.fc1.weight.data = self.fc1.weight.data.relu() / (
            self.fc1.weight.data.relu().sum(1, keepdim=True))
        self.fc2.weight.data.fill_(0)
        


    def forward(self, x):
           
        attributes = self.rescale(self.model(x))
        y = self.fc1(attributes)
        y = self.fc2(y)
        return attributes, torch.sigmoid(y)





use_cuda = torch.cuda.is_available()    


#def get_features(name):
#    '''
#    Hook for fetching features from model for CAM use
#    '''
#    def hook(model, input, output):
#        features[name] = output.detach()
#    return hook

def train_model():
    '''
    Function for training the model
    '''
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
        loss1_sum = 0.0 #attributes loss
        loss2_sum = 0.0 # type loss
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
    torch.save(model.state_dict(), f'../../data/models/end_to_end.pt')

    # Sankey plot preparation
    G = model.fc1.weight.data
    G = G.cpu().numpy()
    W = model.fc2.weight.data
    a = n_attributes
    z = n_groups
    source = [i % a for i in range(z*a)]
    target = [(i // a) + a for i in range(z*a)]
    # Dropping very small weights
    G[G < 0.05] = 0.0
    value = G.flatten().tolist()
    color_node = [
              '#808080', '#808080', '#808080', '#808080', '#808080',
              '#808080', '#808080', '#808080', '#808080', '#808080',
              '#808080', '#808080', '#808080', '#808080', '#808080',
              #'#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#FF00FF',
              #'#00CED1', '#FF8C00', '#BDB76B', '#2F4F4F', '#B8860B'
              ]
    color_link = []
    link_colors = ['#F08080', '#FFFACD', '#98FB98', '#87CEFA', '#FF0000', '#00CED1', '#FF8C00', '#BDB76B', '#2F4F4F', '#B8860B']
    link_colors = link_colors * ((n_groups // len(link_colors)) + 1)
    for i in range(a*z):
        color_link.extend([link_colors[i % len(link_colors)]])
    
    group_label = []
    for i in range(1, n_groups + 1):
        group_label.append("G" + str(i))
    label = ["cystic_c", "mostly solid_c", "solid_c", "spongiform_c",
            "hyper_e", "hypo_e", "iso_e", "marked_e",
            "ill-defined_m", "micro_m", "spiculated_m", "smooth_m",
            "macro_ca", "micro_ca", "non_ca"]
    label.extend(group_label)

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
    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.show()
    plotly.offline.plot(fig, filename=f'plot1_{n_groups}_CAM_{expt_num}.html')

    # Second Sankey plot from groups to type prediction
    source = [i % z for i in range(z*1)]
    target = [(i // z) + z for i in range(z*1)] 
    W = W.cpu().numpy()
    value = W.flatten()

    color_node = ['#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#FF00FF', '#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#FF00FF']
    color_node = color_node * ((n_groups // len(color_node))+ 1)
    color_link = []
    link_pos_colors = ['#149697', '#00AAC0', '#8080CC', '#333388', '#77CCCC', '#149697', '#00AAC0', '#8080CC', '#333388', '#77CCCC']
    link_neg_colors = ['#E65662', '#FF8066', '#894E3F', '#E79598', '#FDA45C', '#E65662', '#FF8066', '#894E3F', '#E79598', '#FDA45C']
    
    link_colors = []
    label = []
    # Coloring links according to +ve or negative and plotting the absolute values
    for i in range(len(source)):
        if value[i] < 0:
            link_colors.append(link_neg_colors[i%10])
        else:
            link_colors.append(link_pos_colors[i%10])
        label.append(str(value[i]) + "| G" + str(i+1))

    label.append("benign/ malignant")
    value = np.abs(value).tolist()
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          color = color_node,
          label = label,
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          color = link_colors
      ))])
    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.show()
    plotly.offline.plot(fig, filename=f'plot2_{n_groups}_CAM_{expt_num}.html')
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
    '''
    Function to perfrom test
    '''
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

    # Code for extracting features for CAM
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    model.model.features.register_forward_hook(get_features('feats'))
    weight_linear = np.squeeze(model.model.classifier[0].weight.detach().cpu().numpy())
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
            # Fetching CAM
            CAMs = returnCAM(FEATS[-1], weight_linear, attrib_list)
            img = np.zeros((252, 252, 3))
            gray_img = data["image"].cpu().numpy()
            gray_img = gray_img[0,0,:,:]
            img[:, :, 0] = gray_img
            img[:, :, 1] = gray_img
            img[:, :, 2] = gray_img
            # img = img * 255
            height, width, c = img.shape
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
            # CAM image is saved only if presence of attribute is correctly predicted
            for i in range(0, n_attributes):
                if predicted[i] != y_test[i] or y_test[i] != 1:
                    continue
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.4 + img
                img_m = np.zeros((252, 252, 3))
                gray_img_m = data["im_masked"].cpu().numpy()
                img_m[:, :, 0] = gray_img_m
                img_m[:, :, 1] = gray_img_m
                img_m[:, :, 2] = gray_img_m
                img_m = img_m * 255
                result = cv2.hconcat([result, img_m])
                cv2.imwrite(f'heatmaps{expt_num}/{file_name}_{i}.png', result)
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
    #with open("results_unmasked.txt", "a") as file1:
    #    file1.write(str(n_groups) + '   ' + str(type_acc) + '  ' + str(accuracy) + '\n')


if __name__ == "__main__":
    model = train_model()
    perform_test(model, thyroidActualDataset('test'))



