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
import plotly.graph_objs as go
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
import json
from torch.utils.data import Dataset
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


im_size = (252, 252)


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
        self.types ={'benign':0, 'malign':1}
        self.types_count = []
        for t_type in ['benign', 'malign']:
            root_dir=Path('../data/' + split + '/' + t_type).expanduser().resolve().absolute() 
            print(root_dir)
            files = list(root_dir.glob("*"))
            labels = [self.types[t_type]] * len(files)
            self.types_count.append(len(files))
            data_list = list(zip(files, labels))
            self.all_data.extend(data_list)
        random.shuffle(self.all_data)
        self.cases, self.types = zip(*self.all_data)
        print("number of data items:" + str(len(self.cases)))
        self.sample_weights = [1/self.types_count[label] for label in self.types]
    def __len__(self):
        return len(self.cases)
  
    def __getitem__(self, idx):
        labels = np.zeros(15, dtype = float)
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).getroot()
        for x in xml_data:
            if x.tag=='composition' and x.text is not None:
                composition = x.text
                if self.compositions[composition] > 0:
                    labels[self.compositions[composition] - 1] = 1.0
            if x.tag=='echogenicity' and x.text is not None:
                echogenicity = x.text
                if self.echogenicities[echogenicity] > 0:
                    labels[self.echogenicities[echogenicity] + 3] = 1.0
            if x.tag=='margins' and x.text is not None:
                margin = x.text
                if self.margins[margin] > 0:
                    labels[self.margins[margin] + 7] = 1.0
            if x.tag=='calcifications' and x.text is not None:
                calcification = x.text
                labels[self.calcifications[calcification] + 12] = 1.0
        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).find("mark")
        for x in xml_data:
            if(x.tag=='svg'):
                encoded = str(x.text)
                poly_data = json.loads(x.text)
        
        #labels[15] = list(self.types)[idx]
        im_name = list(self.cases[idx].glob('*[0-9].jpg'))[0]
        im = cv2.imread(str(im_name))[:, :, 0]
        mask = np.zeros(np.shape(im))
        im = cv2.resize(im, dsize=im_size, interpolation=cv2.INTER_CUBIC)
        
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
        mask = cv2.resize(mask, dsize=im_size, interpolation=cv2.INTER_LINEAR)
        
        
        
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

        transforms = Compose([ToTensor()])
        mask = transforms(mask)
        im = transforms(im)
        
        im = im * mask
        
        im = im.type(torch.FloatTensor)
        
        sample = {"image": im, "labels": torch.from_numpy(labels), "types" : self.types[idx], "name": str(im_name)}
        return sample


# In[3]:


# Dataset creation
training_set = thyroidDataset(split='train')
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

training_generator1 = torch.utils.data.DataLoader(training_set, **parameters_test, sampler=torch.utils.data.WeightedRandomSampler(training_set.sample_weights, len(training_set.cases), replacement=True))


testing_set = thyroidDataset(split='test')
testing_generator = torch.utils.data.DataLoader(testing_set, **parameters_test, sampler=torch.utils.data.WeightedRandomSampler(testing_set.sample_weights, len(testing_set.cases), replacement=True))


import torch.distributions


# In[4]:


class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        self.N = torch.distributions.Normal(0, 1)

        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0


        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.ReLU(True)

        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_fc = nn.Sequential(
            nn.Linear(2304 + 15 + 1, fc2_input_dim * 4),
            nn.ReLU(True),
            nn.Linear(4* fc2_input_dim, 2 * fc2_input_dim)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(2 * fc2_input_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )
        self.encoder_lin1 = nn.Sequential(
            nn.Linear(2 * fc2_input_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )
        
    def forward(self, x, attributes):
        #print("Encoder input: ", np.shape(x))
        x = self.encoder_cnn(x)
        
        x = self.flatten(x)
        #print("Encoder flattened output: ", np.shape(x))
        #x = self.encoder_lin(x)
        x = torch.cat((x, attributes), dim=1)
        x = self.encoder_fc(x.float())
        mu =  self.encoder_lin(x)
        sigma = torch.exp(self.encoder_lin1(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z
    
    
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim + 15 + 1 , fc2_input_dim),
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


### Define the loss function
loss_fn = torch.nn.MSELoss()
loss_latent = torch.nn.L1Loss()



### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 16

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=512)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=512)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)


# In[6]:


### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for data in training_generator: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = data["image"]
        image_batch = image_batch.to(device)
        
        types = torch.unsqueeze(data["types"], 1)
        
        
        #print("Random data shape", np.shape(random_data), "; data[labels] ", np.shape(data["labels"]))
        latent = data["labels"]
        latent = latent.to(device)
        
        latent = torch.cat((latent, types.to(device)), dim=1).to(device)
        
        #print(latent)
        
        # Encode data
        encoded_data = encoder(image_batch, latent)
        
        # Decode data
        
        decoder_input = torch.cat((encoded_data, latent), dim = 1)
        
        decoded_data = decoder(decoder_input.float())
        # Evaluate loss

        #print("Encoder.kl : ", encoder.kl)

        d = ((image_batch - decoded_data)**2).sum()
        loss = d + 0.5*encoder.kl

        #loss = loss_fn(decoded_data, image_batch) + 0.2*loss_latent(encoded_data, latent.float())
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        #train_loss.append(loss.detach().cpu().numpy())

    return loss


# In[ ]:





# In[7]:


num_epochs = 10000
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
    train_loss = train_epoch(encoder,decoder,device,training_generator,loss_fn,optim)
    print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs,train_loss))
    diz_loss['train_loss'].append(train_loss)


# In[22]:
    
torch.save(decoder.state_dict(), "decoder.dict")
# In[23]:


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
        label[0,4+echo] = 1
    if margins > 0:
        label[0,8+margins] = 1
    label[0,12+calc] = 1
    print(label.shape) 
    label[0, 15] = t_type
    return label

# 0 1 2 3 |4 5 6 7 | 10 11 12 13 | 15 16


#print(generate_label())


# In[38]:


#fig, ax = plt.subplots(100, 5, figsize=(10, 40))
#fig.set_tight_layout(True)
from PIL import Image

fp = open("labels_list_gen.txt", "w")
for c in range(100): # with "_" we just ignore the labels (the second element of the dataloader tuple)
    # Move tensor to the proper device
    
    labels = torch.FloatTensor(generate_label()).to(device)
    
    #print("Random data shape", np.shape(random_data), "; data[labels] ", np.shape(data["labels"]))
    for i in range(5):
        save_data = []
        latent = torch.FloatTensor(np.random.randn(1, d)).to(device)
        decoder_input = torch.cat((latent, labels), dim = 1)
        img_recon = decoder(decoder_input)
        img_recon = img_recon.detach().cpu()
        img_recon = torch.squeeze(img_recon).numpy()
        img_recon = img_recon * 255
        img_recon = img_recon.astype(np.uint8)
        img_recon = Image.fromarray(img_recon)
        img_recon.save(f"generated_images1/{c*5+i+100}.jpg")
        save_data.append(c*5+i+100)
        save_labels = labels.cpu().numpy().tolist()
        save_data.extend(save_labels)
        fp.write("%s\n" % save_data)
        #ax[c, i].imshow(img_recon[0, 0, :, :])
    
fp.close()  

#plt.tight_layout()


# In[ ]:




