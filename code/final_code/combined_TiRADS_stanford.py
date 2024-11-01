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
from torch import FloatTensor
import sys
from VGG16 import VGG16
from SegNet import SegNet
from CombNet import CombNet
from CombNet_pool import CombNet_pool
from CombNet_pool_type import CombNet_pool_type
from CombNet_type import CombNet_type

# In[4]:

n_attributes = 14
n_groups = 4
expt_num = sys.argv[1]
PATH = f'../../data/models/segment_stanford_type_acc_combined_pool_reproduce_sampled_64_7.pt'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")    
EPOCHS = 170
EPOCH_CLASSIFIER = 150
TYPE_EPOCHS = 50
    
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
        if int(data_list[21]) > 3:
             labels[0] = 1
        else:
             labels[0] = 0
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
        masked = im * mask
         # Adding data augmentation to avoid overfitting
        aug = True
        if self.split != "test" and self.split != "val" and aug == True:
            if random.randint(1, 10) > 5:
                if random.randint(1, 10) > 5:
                    im = np.flipud(im)
                    mask = np.flipud(mask)
                    masked = np.flipud(masked)
                else: #elif random.randint(1, 10) > 5:
                    im = np.fliplr(im)
                    mask = np.fliplr(mask)
                    masked = np.fliplr(masked)
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
            masked = np.ascontiguousarray(masked)
        transforms = Compose([ToTensor()])
        im = transforms(im)
        masked = transforms(masked)
        mask = torch.from_numpy(mask).long().view(256,256)
        #im_masked = transforms(im_masked)
        labels =  torch.from_numpy(labels)
        labels = labels.type(torch.LongTensor)
        #labels = torch.unsqueeze(labels, 0)
        sample = {"image": im.float(), "mask": mask, "masked": masked.float(), "comp": labels[1], "echo": labels[2], "margin": labels[3], "calc": labels[4], "type": torch.unsqueeze(labels[0], 0), "filename": name}
        return sample


#use_cuda = torch.cuda.is_available()    


#def get_features(name):
#    '''
#    Hook for fetching features from model for CAM use
#    '''
#    def hook(model, input, output):
#        features[name] = output.detach()
#    return hook
parameters_train = {
        "batch_size": 8,
        "shuffle": True,
    }
training_set = thyroidActualDataset('training')
training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)
totiter = len(training_generator)
def train_model():
    '''
    Function for training the model
    '''
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    parameters_test = {
        "batch_size": 1,
        "shuffle": False,
    }
    print("Training")
    #model = net()
    model = CombNet_pool_type()
    model = model.to(device)
    model.fc1.weight.requires_grad = False 
    print(model)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    criterion2 = torch.nn.MSELoss()
    #criterion1 = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    #best_loss1 = float('inf')
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(EPOCHS):    
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
            x_masked = data["masked"]
            y_type = data["type"]
            x_train, y_mask, x_masked = (x_train.to(device), y_mask.to(device), x_masked.to(device))
            y_comp, y_echo, y_margin, y_calc, y_type = (
                y_comp.to(device),
                y_echo.to(device),
                y_margin.to(device),
                y_calc.to(device),
                y_type.to(device)
            )
            optimizer.zero_grad()
            masked_results = model(x_masked)
            #masked_results = masked_results.detach()
            linear_masked_features = masked_results[-2]
            linear_masked_features = linear_masked_features.detach()
            pred_mask, pred_comp, pred_echo, pred_margin, pred_calc, features, linear_features, pred_type = model(x_train)
            #pred_class = torch.argmax(pred_mask, dim=1)
            #pred_class = torch.unsqueeze(pred_class.detach(), 1)
            #pred_class = pred_class.view((256,256))
            #x_pred_masked = x_train * pred_class
            #masked_results = model(x_pred_masked)
            #masked_results = masked_results.detach()
            #linear_masked_features = masked_results[-1]
            #linear_masked_features = linear_masked_features.detach()
            #print(y_mask.size())
            if epoch > EPOCH_CLASSIFIER:
            	#pred_comp = torch.unsqueeze(pred_comp, 1)
            	#pred_echo = torch.unsqueeze(pred_echo, 1)
            	#pred_margin = torch.unsqueeze(pred_margin, 1)
            	#pred_calc = torch.unsqueeze(pred_calc, 1)
                loss_co = criterion(pred_comp, y_comp)
                loss_e = criterion(pred_echo, y_echo)
                loss_m = criterion(pred_margin, y_margin)
                loss_ca = criterion(pred_calc, y_calc)
                loss_p = criterion2(linear_masked_features,linear_features)
                #loss2 = criterion1(pred_type.float(), y_type.float())
                loss_a = loss_co + loss_e + loss_m + loss_ca + loss_p
            
            loss_b = criterion(pred_mask, y_mask)
            
            if epoch > EPOCH_CLASSIFIER:
                loss = loss_a + loss_b
            else:
                loss = loss_b

            loss.backward()
            optimizer.step()
            #model.fc1.weight.data = projection_simplex_sort(model.fc1.weight.data)
            running_loss += loss.item()
            #loss1_sum = loss1_sum + loss_co.item() + loss_e.item() + loss_m.item() + loss_ca.item()
            #loss2_sum += loss2.item()
            
        val_acc = perform_test(model, thyroidActualDataset('val'), display=False)


        if val_acc > best_acc and epoch > EPOCH_CLASSIFIER:
            torch.save(model.state_dict(), PATH.replace("stanford", "stanford_running_loss"))
            best_acc = val_acc
            best_epoch = epoch

        print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / totiter))
        
        if epoch > EPOCH_CLASSIFIER:
            print(loss_a, loss_b, loss_p)
        #print(loss1_sum, loss2_sum, running_loss)
        
    print("Training complete")
    torch.save(model.state_dict(), PATH)
    # Creating the Sankey plot
    return model, best_epoch

def train_type(model):
    for param in model.parameters():
        param.requires_grad = False

    model.fc1.weight.requires_grad = True
    print(model.fc1.weight)
    weight = torch.Tensor([0.52]).to(device)
    criterion1 = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=weight)
    #criterion1 = FocalLoss(weights=class_weights,gamma=1, alpha=1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(TYPE_EPOCHS):
        running_loss = 0.0
        #loss1_sum = 0.0 #attributes loss
        #loss2_sum = 0.0 # type loss
        model.train()
        model.bn11.eval()
        model.bn12.eval()
        model.bn21.eval()
        model.bn22.eval()
        model.bn31.eval()
        model.bn32.eval()
        model.bn33.eval()
        model.bn41.eval()
        model.bn42.eval()
        model.bn43.eval()
        model.bn51.eval()
        model.bn52.eval()
        model.bn53.eval()
        model.bn12d.eval()
        model.bn21d.eval()
        model.bn22d.eval()
        model.bn31d.eval()
        model.bn32d.eval()
        model.bn33d.eval()
        model.bn41d.eval()
        model.bn42d.eval()
        model.bn43d.eval()
        model.bn51d.eval()
        model.bn52d.eval()
        model.bn53d.eval()
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            
            model.train(True)
            model.bn11.eval()
            model.bn12.eval()
            model.bn21.eval()
            model.bn22.eval()
            model.bn31.eval()
            model.bn32.eval()
            model.bn33.eval()
            model.bn41.eval()
            model.bn42.eval()
            model.bn43.eval()
            model.bn51.eval()
            model.bn52.eval()
            model.bn53.eval()
            model.bn12d.eval()
            model.bn21d.eval()
            model.bn22d.eval()
            model.bn31d.eval()
            model.bn32d.eval()
            model.bn33d.eval()
            model.bn41d.eval()
            model.bn42d.eval()
            model.bn43d.eval()
            model.bn51d.eval()
            model.bn52d.eval()
            model.bn53d.eval() 
           
            x_train = data["image"]
            y_mask = data["mask"]
            y_comp = data["comp"]
            y_echo = data["echo"]
            y_margin = data["margin"]
            y_calc = data["calc"]
            x_masked = data["masked"]
            y_type = data["type"]
            x_train, y_mask, x_masked = (x_train.to(device), y_mask.to(device), x_masked.to(device))
            y_comp, y_echo, y_margin, y_calc, y_type = (
                y_comp.to(device),
                y_echo.to(device),
                y_margin.to(device),
                y_calc.to(device),
                y_type.to(device)
            )
            optimizer.zero_grad()
            pred_mask, pred_comp, pred_echo, pred_margin, pred_calc, features, linear_features, pred_type = model(x_train)
            loss2 = criterion1(pred_type.float(), y_type.float())
            loss2.backward()
            optimizer.step()
            #model.fc1.weight.data = projection_simplex_sort(model.fc1.weight.data)
            running_loss += loss2.item()
            #loss1_sum = loss1_sum + loss_co.item() + loss_e.item() + loss_m.item() + loss_ca.item()
            #loss2_sum += loss2.item()

        val_acc = perform_test(model, thyroidActualDataset('val'), TiRADS=True,display=False)


        if val_acc > best_acc:
            torch.save(model.state_dict(), PATH.replace("stanford", "stanford_type_acc"))
            best_acc = val_acc
            best_epoch = epoch

        print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / totiter))

        #print(loss1_sum, loss2_sum, running_loss)

    print("Training complete")
    print(model.fc1.weight)
    torch.save(model.state_dict(), PATH.replace("stanford", "stanford_type"))
    return best_epoch, model
    
def perform_test(model, dataset, TiRADS=False, display=False):
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
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    echo_correct = 0
    margin_correct = 0
    calc_correct = 0
    metric = JaccardIndex(task='multiclass', num_classes=2, ignore_index=None)
    type_correct = 0
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
    rescale = torch.nn.Softmax()
    with torch.no_grad():
        print(model.training)
        all_0 = True
        all_1 = True
        for batch_idx, data in tqdm(enumerate(test_generator), total=totiter):
            x_test = data["image"]
            y_mask = data["mask"]
            y_comp = data["comp"]
            y_echo = data["echo"]
            y_margin = data["margin"]
            y_calc = data["calc"]
            y_type = data["type"]
            y_comp, y_echo, y_margin, y_calc, y_type = (
                y_comp.to(device),
                y_echo.to(device),
                y_margin.to(device),
                y_calc.to(device),
                y_type.to(device)
            )
            x_test, y_mask = (x_test.to(device), y_mask.to(device))
            #print(x_test)
            pred_mask, pred_comp, pred_echo, pred_margin, pred_calc, features, linear_features, pred_type = model(x_test)
            #FEATS.append(features['feats'].cpu().numpy())
            #attrib_list = [i for i in range(0, n_attributes)]
            # Fetching CAM
            pred_type = torch.sigmoid(pred_type)
            loss_co = criterion(pred_comp, y_comp)
            loss_e = criterion(pred_echo, y_echo)
            loss_m = criterion(pred_margin, y_margin)
            loss_ca = criterion(pred_calc, y_calc)
            #loss2 = criterion1(pred_type.float(), y_type.float())
            loss_a = loss_co + loss_e + loss_m + loss_ca
            loss_b = criterion(pred_mask, y_mask)
            loss = loss_a + loss_b
            running_loss += loss.item()
            pred_class = torch.argmax(pred_mask, dim=1)
            pred_class = pred_class.view((256,256))
            pred_class[pred_class==1]=255
            #name = data["filename"][0]
            #cv2.imwrite(name.replace("images", "pred_masks"), pred_class.cpu().numpy())

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
            pred_type = pred_type.detach().cpu().numpy()
            pred_type = pred_type.flatten()
            pred_comp_class = np.argmax(pred_comp)
            pred_echo_class = np.argmax(pred_echo)
            pred_margin_class = np.argmax(pred_margin)
            pred_calc_class = np.argmax(pred_calc)
            #print(np.multiply(predicted,value))
            #print(np.sum(np.multiply(predicted,value)))
            y_type = y_type.detach().cpu().numpy()
            y_type = y_type.flatten()
            y_comp = y_comp.detach().cpu().numpy()
            y_comp = y_comp.flatten()
            y_echo = y_echo.detach().cpu().numpy()
            y_echo = y_echo.flatten()
            y_margin = y_margin.detach().cpu().numpy()
            y_margin = y_margin.flatten()
            y_calc = y_calc.detach().cpu().numpy()
            y_calc = y_calc.flatten()
            #predicted[predicted < 0.5] = 0.0
            #predicted[predicted >= 0.5] = 1.0
            pred_type[pred_type < 0.5] = 0.0
            pred_type[pred_type >= 0.5] = 1.0
            #total += 15
            total += 1
            pred_type = pred_type.astype(int)
            y_type = y_type.astype(int)
            if pred_comp_class == y_comp[0]:
                comp_correct += 1
            if pred_echo_class == y_echo[0]:
                echo_correct += 1
            if pred_margin_class == y_margin[0]:
                margin_correct += 1
            if pred_calc_class == y_calc[0]:
                calc_correct += 1
            class_correctness = [comp_correct, echo_correct, margin_correct, calc_correct]
            if y_type[0] == pred_type[0]:
                type_correct += 1
            if y_type == 1 and pred_type == 1:
                tp += 1
            if y_type == 0 and pred_type == 0:
                tn += 1
            if y_type == 1 and pred_type == 0:
                fn += 1
            if y_type == 0 and pred_type == 1:
                fp += 1
            if pred_type[0] == 0:
                all_1 = False
            if pred_type[0] == 1:
                all_0 = False
            if TiRADS == True:
                print(data["filename"],y_type[0], pred_type[0])

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
    class_correctness = [comp_correct/total, echo_correct/total, margin_correct/total, calc_correct/total]
    print(class_correctness)
    print('Type Accuracy:')
    print(type_correct / total)
    print(tp/(tp+fp))
    print(tp/(tp+fn))
    #print('Type accuracy: %d %%' %(100 * type_correct / total))
    if TiRADS == False:
    	return Per_class_IoU + 0.25 * sum(class_correctness)
    else:
        if not all_0 and not all_1:
            return type_correct / total
        else:
            return 0.0
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
    #model = CombNet_pool_type()
    
    #model = model.to(device)
    #model.load_state_dict(torch.load(PATH.replace("stanford","stanford_running_loss")), strict=False)
    #best_epoch , model = train_type(model)
    #print(best_epoch) 
    model = CombNet_pool_type()

    model = model.to(device)
    model.load_state_dict(torch.load(PATH))
    perform_test(model, thyroidActualDataset('test'), TiRADS=True, display=False)
    #perform_test(model, thyroidActualDataset('training'), display=False)
    #perform_test(model, thyroidActualDataset('val'), display=False)



