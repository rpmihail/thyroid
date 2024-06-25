'''
Code for training model with 2 layers for predicting type
'''
import glob
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
from torchmetrics import JaccardIndex
import sys
from VGG16 import VGG16
from SegNet import SegNet
# In[4]:

n_attributes = 14
n_groups = 4
expt_num = sys.argv[1]
PATH = f'../../data/models/segment_stanford_running_loss_iou_SegNet_reproduce_sampled.pt'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")    
"""
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
"""

class thyroidActualDataset(Dataset):
    def __init__(self, split):
        with open('../legacy_code/labels_stanford_' + split + '.txt') as f:
            self.cases = f.read().splitlines()
            self.samples = []
            for case in self.cases:
                case_num = case.split(',')[0]
                files = glob.glob('/home/aroychoudhury/Stanford_thyroid/thyroidultrasoundcineclip/images/' + case_num + '*')
                case_data = case.split(',')[1:]
                for file_name in files: #[:1]:
                    file_no_ext = file_name.split('.')[0]
                    case_frame = file_no_ext.split('/')[-1]
                    if split == 'training' and int(case_frame.split('_')[-1]) % 10 == 0:
                        sample = case.replace(case_num, file_name)
                        self.samples.append(sample)
                    elif split != 'training': #and int(case_frame.split('_')[-1]) % 10 == 0:
                        sample = case.replace(case_num, file_name)
                        self.samples.append(sample)
                    

        self.split = split
        #if len(self.samples) > 73:
        #    self.samples = self.samples[:73]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_list = self.samples[idx].split(', ')
        name = data_list[0]
        #print(name)
        labels = data_list[15:20]
        labels = [float(element) for element in labels]
        labels = np.array(labels)
        #xml_data = ET.parse('/home/ahana/thyroid/data/thyroid/'+ name + '.xml').find("mark")
        #for x in xml_data:
        #    if(x.tag=='svg'):
        #        encoded = str(x.text)
        #        poly_data = json.loads(x.text)
        # Reading maske dimages
        #im_name = os.path.join('/home/ahana/thyroid/data/thyroid', name + '.jpg')
        #im_masked = cv2.imread(str(im_name))
        #im_masked = cv2.resize(im_masked, dsize=(252, 252), interpolation=cv2.INTER_CUBIC)
        #im_masked = im_masked[:,:,0]
        # Reads actual images. If-else logic is leagacy and was used for cases where images were genereated.
        # Only the if code without condition is necessary now.
        if True: #int(name) < 100:
            #im_name = os.path.join('/home/ahana/thyroid/data/thyroid', name + '_1.jpg')
            im1 = cv2.imread(str(name))
            #mask = np.zeros(np.shape(im1))
            #for polygon in poly_data:
            #    xs = []
            #    ys = []
            #    for point in polygon["points"]:
            #        xs.append(point["x"])
            #        ys.append(point["y"])
            #    contour = np.concatenate((np.expand_dims(xs, 1), np.expand_dims(ys, 1)), axis=1)
            #    cv2.fillPoly(mask, pts = [contour], color =(1, 1, 1))
            mask = cv2.imread(str(name).replace('images','masks_new'))
            #print(str(name).replace('images','masks'))
            #masked_im = mask * im1
            #masked_im = cv2.resize(masked_im, dsize=(252, 252), interpolation=cv2.INTER_LINEAR)
            #cv2.imwrite('images_check/' + name + '.png', masked_im)
            #masked_im = masked_im[:,:,0]
            mask[mask==255] = 1
            #print(np.unique(mask))
            im = np.zeros((256, 256, 3))
            mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            im1 = cv2.resize(im1, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            im[:220,:,:] = im1[:220,:,:]
           
            #cv2.imwrite('/home/ahana/test_stanford_images/' + name.split('/')[-1], im)
            #cv2.imwrite('images_check/' + name + '_orig.png', im)
        #else:
        #    im = np.zeros((252, 252, 3))
        
        # Creating 2 channel input
        #im[:, :, 1] = masked_im
        #im = im[:,:,:2]
        im = im[:,:,:1]
        mask = mask[:,:,:1]
         # Adding data augmentation to avoid overfitting
        aug = True
        if self.split != "test" and self.split != "val" and aug == True:
            if random.randint(1, 10) > 5:
                if random.randint(1, 10) > 5:
                    im = np.flipud(im)
                    mask = np.flipud(mask)
                else: #elif random.randint(1, 10) > 5:
                    im = np.fliplr(im)
                    mask = np.fliplr(mask)
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
            mask = np.ascontiguousarray(mask)
        transforms = Compose([ToTensor()])
        im = transforms(im)
        mask = torch.from_numpy(mask).long().view(256,256)
        #im_masked = transforms(im_masked)
        labels =  torch.from_numpy(labels)
        labels = labels.type(torch.LongTensor)
        #labels = torch.unsqueeze(labels, 0)
        sample = {"image": im.float(), "mask": mask, "comp": labels[1], "echo": labels[2], "margin": labels[3], "calc": labels[4], "type": torch.unsqueeze(labels[0], 0), "filename": name}
        return sample



class net(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = models.vgg16(pretrained=True) # pretrained=False just for debug reasons
        model_layers = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        model_layers.extend(list(self.model.features))
        self.model.features= nn.Sequential(*model_layers)
        #self.model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.model.avgpool = torch.nn.Identity()
        # modifying VGG-16 for CAM
        self.model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 128))
        self.dropout = nn.Dropout(p=0.2)
        self.rescale = torch.nn.Softmax()
        # Adding 2 layers for type prediction

        self.fc_comp1 = nn.Linear(128, 64, bias = False)
        self.fc_comp2 = nn.Linear(64, 3, bias = False)
        self.fc_echo1 = nn.Linear(128, 64, bias = False)
        self.fc_echo2 = nn.Linear(64, 4, bias = False)
        self.fc_margin1 = nn.Linear(128, 64, bias = False)
        self.fc_margin2 = nn.Linear(64, 3, bias = False)
        self.fc_calc1 = nn.Linear(128, 64, bias = False)
        self.fc_calc2 = nn.Linear(64, 4, bias = False)
        #self.fc1 = nn.Linear(n_attributes, 1, bias = False)
        #self.fc2 = nn.Linear(n_groups, 1, bias = False)
        #self.fc1.weight.data = self.fc1.weight.data.relu() / (
        #    self.fc1.weight.data.relu().sum(1, keepdim=True))
        #self.fc2.weight.data.fill_(0)
        


    def forward(self, x):
           
        features = self.dropout(F.relu(self.model(x)))
        comp = F.relu(self.fc_comp1(features))
        comp_raw = self.fc_comp2(comp)
        comp = self.rescale(comp_raw)
        echo = F.relu(self.fc_echo1(features))
        echo_raw = self.fc_echo2(echo)
        echo = self.rescale(echo_raw)

        margin = F.relu(self.fc_margin1(features))
        margin_raw = self.fc_margin2(margin)
        margin =  self.rescale(margin_raw)
        calc = F.relu(self.fc_calc1(features))
        calc_raw = self.fc_calc2(calc)
        calc = self.rescale(calc_raw)
        #attributes = torch.cat((comp, echo, margin, calc), dim=1)

        #y = self.fc1(attributes)
        #y = self.fc2(y)
        return comp_raw, echo_raw, margin_raw, calc_raw





#use_cuda = torch.cuda.is_available()    


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
    #device = torch.device("cuda:0" if use_cuda else "cpu")
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
    #model = net()
    model = SegNet()
    model = model.to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    #criterion1 = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    #best_loss1 = float('inf')
    best_iou = 0.0
    best_epoch = -1

    for epoch in range(200):    
        running_loss = 0.0
        #loss1_sum = 0.0 #attributes loss
        #loss2_sum = 0.0 # type loss
        model.train()
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            model.train(True)
            x_train = data["image"]
            y_mask = data["mask"]
            y_comp = data["comp"]
            y_echo = data["echo"]
            y_margin = data["margin"]
            y_calc = data["calc"]
            #y_type = data["type"]
            x_train, y_mask = (x_train.to(device), y_mask.to(device))
            '''
            x_train, y_comp, y_echo, y_margin, y_calc = (
                x_train.to(device),
                y_comp.to(device),
                y_echo.to(device),
                y_margin.to(device),
                y_calc.to(device),
            )
            '''
            optimizer.zero_grad()

            pred_mask = model(x_train)
            #print(y_mask.size())
            #pred_comp = torch.unsqueeze(pred_comp, 1)
            #pred_echo = torch.unsqueeze(pred_echo, 1)
            #pred_margin = torch.unsqueeze(pred_margin, 1)
            #pred_calc = torch.unsqueeze(pred_calc, 1)
            #loss_co = criterion(pred_comp, y_comp)
            #loss_e = criterion(pred_echo, y_echo)
            #loss_m = criterion(pred_margin, y_margin)
            #loss_ca = criterion(pred_calc, y_calc)
            #loss2 = criterion1(target.float(), y_type.float())
            #loss = loss_co + loss_e + loss_m + loss_ca
            loss = criterion(pred_mask, y_mask)
            loss.backward()
            optimizer.step()
            #model.fc1.weight.data = projection_simplex_sort(model.fc1.weight.data)
            running_loss += loss.item()
            #loss1_sum = loss1_sum + loss_co.item() + loss_e.item() + loss_m.item() + loss_ca.item()
            #loss2_sum += loss2.item()
            
        val_iou = perform_test(model, thyroidActualDataset('val'), display=False)


        if val_iou > best_iou:
            torch.save(model.state_dict(), PATH.replace("_stanford", "_stanford_running_loss"))
            best_iou = val_iou
            best_epoch = epoch

        print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / totiter))
        #print(loss1_sum, loss2_sum, running_loss)
        
    print("Training complete")
    torch.save(model.state_dict(), PATH)
    # Creating the Sankey plot
    """"
    G = model.fc1.weight.data
    G = G.cpu().numpy()
    a = n_attributes
    z = 1
    source = [i % a for i in range(z*a)]
    target = [(i // a) + a for i in range(z*a)]
    '''
    '''
    value = G.flatten().tolist()
    print(value)
    color_node = [
            #'#808080', '#808080', '#808080', '#808080', '#808080',
            #'#808080', '#808080', '#808080', '#808080', '#808080',
            #'#808080', '#808080', '#808080', '#808080', '#808080',
            '#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#FF00FF',
            '#00CED1', '#FF8C00', '#BDB76B', '#2F4F4F', '#B8860B',
            '#149697', '#00AAC0', '#8080CC', '#333388', '#77CCCC'
            ]
    color_link = []

    color_link = ['#5d8aa8','#ffbf00','#e52b50','#9966cc',
                  '#4b5320','#0d98ba','#480607','#cc5500',
                  '#702963','#008b8b','#a52a2a','#004b49',
                  '#b8860b','#ff4040','#00009c']
    label = []

    attrib_names = ["cystic composition", "mostly solid composition", "solid composition", "spongiform composition",
                    "hyperechogenicity", "hypoechogenicity", "isoechogenicity", "marked hypoechogenicity",
                    "ill-defined margin", "microlobulated margin", "spiculated margin", "smooth margin",
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

    """

    '''
    # Sankey plot preparation
    G = model.fc1.weight.data
    G = G.cpu().numpy()
    #W = model.fc2.weight.data
    a = n_attributes
    z = 1
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
    '''
    return model, best_epoch
'''
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
'''
def perform_test(model, dataset, display=False):
    '''
    Function to perfrom test
    '''
    model.eval()
    #print(model)
    #print(list(model.parameters()))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    parameters_test = {
        "batch_size": 1,
        "shuffle": False,
    }
    '''
    G = model.fc1.weight.data
    G = G.cpu().numpy()
    value = G.flatten().tolist()
    print(value)
    '''
    test_set = dataset
    test_generator = torch.utils.data.DataLoader(test_set, **parameters_test) 
    totiter = len(test_generator)
    comp_correct = 0
    total = 0
    echo_correct = 0
    margin_correct = 0
    calc_correct = 0
    metric = JaccardIndex(task='multiclass', num_classes=2, ignore_index=None)
    #type_correct = 0
    FEATS = []
    features = {}
    accuracy = []
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    #criterion1 = torch.nn.BCELoss(reduction="mean")
    # Code for extracting features for CAM
    '''
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    '''
    #model.model.features.register_forward_hook(get_features('feats'))
    #weight_linear = np.squeeze(model.model.classifier[0].weight.detach().cpu().numpy())
    running_loss = 0.0
    #loss1_sum = 0.0 #attributes loss
    #loss2_sum = 0.0
    #rescale = torch.nn.Softmax()
    with torch.no_grad():
        print(model.training)
        for batch_idx, data in tqdm(enumerate(test_generator), total=totiter):
            x_test = data["image"]
            y_mask = data["mask"]
            y_comp = data["comp"]
            y_echo = data["echo"]
            y_margin = data["margin"]
            y_calc = data["calc"]
            #y_type = data["type"]
            '''
            x_test, y_comp, y_echo, y_margin, y_calc = (
                x_test.to(device),
                y_comp.to(device),
                y_echo.to(device),
                y_margin.to(device),
                y_calc.to(device)
            )
            '''
            x_test, y_mask = (x_test.to(device), y_mask.to(device))
            #print(x_test)
            pred_mask = model(x_test)
            #FEATS.append(features['feats'].cpu().numpy())
            #attrib_list = [i for i in range(0, n_attributes)]
            # Fetching CAM
            #loss_co = criterion(pred_comp, y_comp)
            #loss_e = criterion(pred_echo, y_echo)
            #loss_m = criterion(pred_margin, y_margin)
            #loss_ca = criterion(pred_calc, y_calc)
            #loss2 = criterion1(pred_type.float(), y_type.float())
            #loss = loss_co + loss_e + loss_m + loss_ca
            loss = criterion(pred_mask, y_mask)
            running_loss += loss.item()
            pred_class = torch.argmax(pred_mask, dim=1)
            pred_class = pred_class.view((256,256))
            pred_class[pred_class==1]=255
            name = data["filename"][0]
            cv2.imwrite(name.replace("images", "pred_masks"), pred_class.cpu().numpy())

            accuracy.append(metric(pred_mask.cpu(), y_mask.cpu()))
            #loss1_sum = loss1_sum + loss_co.item() + loss_e.item() + loss_m.item() + loss_ca.item()
            #loss2_sum += loss2.item()
            '''
            CAMs = returnCAM(FEATS[-1], weight_linear, attrib_list)
            img = np.zeros((252, 252, 3))
            gray_img = data["image"].cpu().numpy()
            gray_img = gray_img[0,0,:,:]
            img[:, :, 0] = gray_img
            img[:, :, 1] = gray_img
            img[:, :, 2] = gray_img
            # img = img * 255
            height, width, c = img.shape
            '''
            '''
            comp_probs = rescale(pred_comp).detach().cpu().numpy()
            echo_probs = rescale(pred_echo).detach().cpu().numpy()
            margin_probs = rescale(pred_margin).detach().cpu().numpy()
            calc_probs = rescale(pred_calc).detach().cpu().numpy()
            pred_comp = pred_comp.detach().cpu().numpy()
            pred_comp = pred_comp.flatten()
            pred_echo = pred_echo.detach().cpu().numpy()
            pred_echo = pred_echo.flatten()
            pred_margin = pred_margin.detach().cpu().numpy()
            pred_margin = pred_margin.flatten()
            pred_calc = pred_calc.detach().cpu().numpy()
            pred_calc = pred_calc.flatten()
            #pred_type = pred_type.detach().cpu().numpy()
            #pred_type = pred_type.flatten()
            pred_comp_class = np.argmax(pred_comp)
            pred_echo_class = np.argmax(pred_echo)
            pred_margin_class = np.argmax(pred_margin)
            pred_calc_class = np.argmax(pred_calc)
            #print(np.multiply(predicted,value))
            #print(np.sum(np.multiply(predicted,value)))
            #y_type = y_type.detach().cpu().numpy()
            #y_type = y_type.flatten()
            y_comp = y_comp.detach().cpu().numpy()
            y_comp = y_comp.flatten()
            y_echo = y_echo.detach().cpu().numpy()
            y_echo = y_echo.flatten()
            y_margin = y_margin.detach().cpu().numpy()
            y_margin = y_margin.flatten()
            y_calc = y_calc.detach().cpu().numpy()
            y_calc = y_calc.flatten()
            '''
            #predicted[predicted < 0.5] = 0.0
            #predicted[predicted >= 0.5] = 1.0
            #pred_type[pred_type < 0.5] = 0.0
            #pred_type[pred_type >= 0.5] = 1.0
            #total += 15
            total += 1
            #pred_type = pred_type.astype(int)
            #y_type = y_type.astype(int)
            '''
            if pred_comp_class == y_comp[0]:
                comp_correct += 1
            if pred_echo_class == y_echo[0]:
                echo_correct += 1
            if pred_margin_class == y_margin[0]:
                margin_correct += 1
            if pred_calc_class == y_calc[0]:
                calc_correct += 1
            class_correctness = [comp_correct, echo_correct, margin_correct, calc_correct]
            '''
            #y_test = y_test.astype(int)
            #predicted = predicted.astype(int)
            #errors = np.ones(15, dtype = int)
            #errors[predicted == y_test] = 0
            #correct = correct + 15 - np.sum(errors)
            # CAM image is saved only if presence of attribute is correctly predicted
            '''
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
            '''
            '''
            if display==True:
                #print(y_type[0], pred_type[0])
                print(pred_comp_class, pred_echo_class, pred_margin_class, pred_calc_class)
                print(y_comp, y_echo, y_margin, y_calc)
            #if y_type[0] == pred_type[0]:
            #    type_correct += 1
            rescale = torch.nn.Softmax()
            #print(data["filename"], pred_comp_class, pred_echo_class, pred_margin_class, pred_calc_class, comp_probs, echo_probs, margin_probs, calc_probs)
            #for i in range(15):
            #    class_correctness[i] = class_correctness[i] + 1 - errors[i]
            '''
    #print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    #accuracy = 100 * correct / total
    print(running_loss)
    print('Per class acuracy is: ')
    IoUs = np.array(accuracy)
    Per_class_IoU = np.mean(IoUs, axis=0)
    #miou = np.mean(Per_class_IoU, axis=0)
    print(Per_class_IoU)
    #class_correctness = [comp_correct/total, echo_correct/total, margin_correct/total, calc_correct/total]
    #print(class_correctness)
    #print('Type accuracy: %d %%' %(100 * type_correct / total))
    return Per_class_IoU
    #type_acc = 100 * type_correct / type_total
    #FEATS = np.concatenate(FEATS)
    #print('- feats shape:', FEATS.shape)
    #with open("results_unmasked.txt", "a") as file1:
    #    file1.write(str(n_groups) + '   ' + str(type_acc) + '  ' + str(accuracy) + '\n')


if __name__ == "__main__":
    #torch.use_deterministic_algorithms(True)
    #model, best_epoch = train_model()
    #print(best_epoch)
    #torch.use_deterministic_algorithms(True)
    model = SegNet()

    model = model.to(device)
    model.load_state_dict(torch.load(PATH))
    #perform_test(model, thyroidActualDataset('test'), display=False)
    #perform_test(model, thyroidActualDataset('training'), display=False)
    perform_test(model, thyroidActualDataset('val'), display=False)



