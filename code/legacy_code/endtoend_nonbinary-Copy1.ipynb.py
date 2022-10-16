{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from scipy.optimize import minimize\n",
    "import matplotlib.pyplot as plt\n",
    "from itertools import product\n",
    "from sklearn.model_selection import train_test_split\n",
    "from pathlib import Path\n",
    "import random\n",
    "from tqdm import tqdm\n",
    "import xml.etree.ElementTree as ET\n",
    "import cv2\n",
    "import plotly.graph_objs as go\n",
    "from torchvision.transforms import Compose\n",
    "from torchvision.transforms import ToTensor\n",
    "import json\n",
    "from torch.utils.data import Dataset\n",
    "import torch\n",
    "import cv2\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "device\n",
    "\n",
    "# dataset definition\n",
    "class thyroidDataset(Dataset):\n",
    "    def __init__(self, split):\n",
    "        self.all_data = []\n",
    "        self.split = split\n",
    "        \n",
    "        self.compositions = {'Unknown':0, 'cystic':1,\n",
    "                             'predominantly solid':2,\n",
    "                             'solid':3, 'spongiform appareance':4}\n",
    "        self.echogenicities = {'Unknown':0, 'hyperechogenecity':1,\n",
    "                             'hypoechogenecity':2, 'isoechogenicity':3,\n",
    "                             'marked hypoechogenecity':4}\n",
    "        self.margins = {'Unknown':0, 'ill- defined':1, 'microlobulated':2,\n",
    "                        'spiculated':3, 'well defined smooth':4}\n",
    "        self.calcifications = {'macrocalcification':0, 'microcalcification':1, 'non':2}\n",
    "        self.types ={'benign':0, 'malign':1}\n",
    "        self.types_count = []\n",
    "        for t_type in ['benign', 'malign']:\n",
    "            root_dir=Path('../data/' + split + '/' + t_type).expanduser().resolve().absolute() \n",
    "            print(root_dir)\n",
    "            files = list(root_dir.glob(\"*\"))\n",
    "            labels = [self.types[t_type]] * len(files)\n",
    "            self.types_count.append(len(files))\n",
    "            data_list = list(zip(files, labels))\n",
    "            self.all_data.extend(data_list)\n",
    "        random.shuffle(self.all_data)\n",
    "        self.cases, self.types = zip(*self.all_data)\n",
    "        print(\"number of data items:\" + str(len(self.cases)))\n",
    "        # self.sample_weights = [1/self.types_count[label] for label in self.types]\n",
    "    def __len__(self):\n",
    "        return len(self.cases)\n",
    "  \n",
    "    def __getitem__(self, idx):\n",
    "        labels = np.zeros(16, dtype = float)\n",
    "        composition = 'Unknown'\n",
    "        echogenicity = 'Unknown'\n",
    "        margin = 'Unknown'\n",
    "        calcification = 'Unknown'\n",
    "        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).getroot()\n",
    "        for x in xml_data:\n",
    "            if x.tag=='composition' and x.text is not None:\n",
    "                composition = x.text\n",
    "                #self.compositions[composition]\n",
    "                # labels[self.compositions[composition] - 1] = 1.0\n",
    "            if x.tag=='echogenicity' and x.text is not None:\n",
    "                echogenicity = x.text\n",
    "                # labels[self.echogenicities[echogenicity] + 3] = 1.0\n",
    "            if x.tag=='margins' and x.text is not None:\n",
    "                margin = x.text\n",
    "                # labels[self.margins[margin] + 7] = 1.0\n",
    "            if x.tag=='calcifications' and x.text is not None:\n",
    "                calcification = x.text\n",
    "                # labels[self.calcifications[calcification] + 11] = 1.0\n",
    "        xml_data = ET.parse(list(self.cases[idx].glob('*[0-9].xml'))[0]).find(\"mark\")\n",
    "        for x in xml_data:\n",
    "            if(x.tag=='svg'):\n",
    "                encoded = str(x.text)\n",
    "                poly_data = json.loads(x.text)\n",
    "        \n",
    "        # labels[15] = list(self.types)[idx]\n",
    "        im_name = list(self.cases[idx].glob('*[0-9].jpg'))[0]\n",
    "        im = cv2.imread(str(im_name))\n",
    "        mask = np.zeros(np.shape(im))\n",
    "        im = cv2.resize(im, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)\n",
    "        \n",
    "        # add mask \n",
    "        for polygon in poly_data:\n",
    "            xs = []\n",
    "            ys = []\n",
    "            for point in polygon[\"points\"]:\n",
    "                xs.append(point[\"x\"])\n",
    "                ys.append(point[\"y\"])\n",
    "            contour = np.concatenate((np.expand_dims(xs, 1), np.expand_dims(ys, 1)), axis=1)\n",
    "            cv2.fillPoly(mask, pts = [contour], color =(1, 1, 1))\n",
    "        \n",
    "        mask = cv2.resize(mask, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)\n",
    "        \n",
    "        #im = im * mask\n",
    "        \n",
    "        # Adding data augmentation to avoid overfitting\n",
    "        if self.split == 'train':\n",
    "            if random.randint(1, 10) > 5:\n",
    "                im = np.flipud(im)\n",
    "                mask = np.flipud(mask)\n",
    "            if random.randint(1, 10) > 5:\n",
    "                im = np.fliplr(im)\n",
    "                mask = np.fliplr(mask)\n",
    "            if random.randint(1, 10) > 5:\n",
    "                for i in range(random.randint(1, 4)):\n",
    "                    im = np.rot90(im)\n",
    "                    mask = np.rot90(mask)\n",
    "        im = np.ascontiguousarray(im)\n",
    "        mask = np.ascontiguousarray(mask)\n",
    "        #plt.figure()\n",
    "        #plt.imshow(im)\n",
    "\n",
    "        transforms = Compose([ToTensor()])\n",
    "        mask = transforms(mask)\n",
    "        im = transforms(im)\n",
    "        mask = mask[0]\n",
    "        mask = torch.unsqueeze(mask,0)\n",
    "        #print(mask.shape)\n",
    "        masked_im = torch.cat((im, mask), 0)\n",
    "        # masked_im = im * ((mask / 2) + 0.5)\n",
    "        \n",
    "        #print(masked_im)\n",
    "        \n",
    "        masked_im = masked_im.type(torch.cuda.FloatTensor)\n",
    "        \n",
    "        sample = {\"image\": masked_im, \"comp\": self.compositions[composition],\n",
    "                  \"echo\": self.echogenicities[echogenicity],\n",
    "                  \"margin\": self.margins[margin],\n",
    "                  \"calc\": self.calcifications[calcification],\n",
    "                  \"types\" : self.types[idx], \"name\": str(im_name)}\n",
    "        return sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/ahana/thyroid/data/train/benign\n",
      "/home/ahana/thyroid/data/train/malign\n",
      "number of data items:73\n",
      "/home/ahana/thyroid/data/train/benign\n",
      "/home/ahana/thyroid/data/train/malign\n",
      "number of data items:73\n"
     ]
    }
   ],
   "source": [
    "# Dataset creation\n",
    "training_set = thyroidDataset(split='train')\n",
    "parameters_train = {\n",
    "    \"batch_size\": 8,\n",
    "    \"shuffle\": True,\n",
    "}\n",
    "parameters_test = {\n",
    "    \"batch_size\": 1,\n",
    "    \"shuffle\": False,\n",
    "}\n",
    "training_set = thyroidDataset(split='train')\n",
    "#training_generator = torch.utils.data.DataLoader(training_set, **parameters_train, sampler=torch.utils.data.WeightedRandomSampler(training_set.sample_weights, len(training_set.cases), replacement=True))\n",
    "training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)\n",
    "totiter = len(training_generator)\n",
    "train_data = iter(training_generator).next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model definition and creation\n",
    "\n",
    "z = 7 # groups\n",
    "a = 18\n",
    "k = 5 # top k \n",
    "num_labels = 1\n",
    "num_channels = 4\n",
    "\n",
    "G = np.random.randn(z, a)\n",
    "W = np.random.randn(z, num_labels)\n",
    "\n",
    "# class MutualExclusivityLoss(nn.Module):\n",
    "#     def __init__(self, num_classes, num_attrs):\n",
    "#         super(MutualExclusivityLoss, self).__init__()\n",
    "#         self.num_attrs = num_attrs\n",
    "#         self.num_classes = num_classes\n",
    "#     def forward(self, target, result):\n",
    "#         target = 1.0 - target\n",
    "#         me_loss = target.mul(result)\n",
    "#         return torch.mean(me_loss)\n",
    "        \n",
    "   \n",
    "\n",
    "class AttribNet(torch.nn.Module):\n",
    "    \n",
    "    def __init__(self, num_classes):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.conv1 = nn.Conv2d(num_channels, 16, 5)\n",
    "        self.pool = nn.MaxPool2d(2, 2)\n",
    "        self.conv2 = nn.Conv2d(16, 32, 5)\n",
    "        self.conv3 = nn.Conv2d(32, 64, 5)\n",
    "        self.conv4 = nn.Conv2d(64, 128, 5)\n",
    "        self.conv5 = nn.Conv2d(128, 256,3)\n",
    "        self.GAP = nn.AdaptiveAvgPool2d((1,1))\n",
    "        self.fc3 = nn.Linear(256, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "            \n",
    "        x = self.pool(F.tanh(self.conv1(x)))\n",
    "        x = self.pool(F.tanh(self.conv2(x)))\n",
    "        x = self.pool(F.tanh(self.conv3(x)))\n",
    "        x = self.pool(F.tanh(self.conv4(x)))\n",
    "        # Adding 1 more conv, a GAP and a final linear layer for CAM\n",
    "        x = F.tanh(self.conv5(x))\n",
    "        features = x\n",
    "        x = self.GAP(x)\n",
    "        \n",
    "        x = torch.flatten(x, 1) # flatten all dimensions except batch\n",
    "        x = self.fc3(x)\n",
    "        # x = torch.unsqueeze(x, 2)\n",
    "        \n",
    "        return (x, features)\n",
    "    \n",
    "class TypeNet(torch.nn.Module):\n",
    "    \n",
    "    def __init__(self, G, W):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.comp_model = AttribNet(5)\n",
    "        self.echo_model = AttribNet(5)\n",
    "        self.margin_model = AttribNet(5)\n",
    "        self.calc_model = AttribNet(3)\n",
    "        \n",
    "        self.G_ = torch.nn.Parameter(G)\n",
    "        self.W_ = torch.nn.Parameter(W)\n",
    "        #self.CNN_ = torch.nn.Parameter(CNN)\n",
    "        \n",
    "\n",
    "\n",
    "    def forward(self, x):\n",
    "            \n",
    "        y_comp, comp_features = self.comp_model(x)\n",
    "        y_echo, echo_features = self.echo_model(x)\n",
    "        y_margin, margin_features = self.margin_model(x)\n",
    "        y_calc, calc_features = self.calc_model(x)\n",
    "        y_comp_s = F.softmax(y_comp)\n",
    "        y_echo_s = F.softmax(y_echo)\n",
    "        y_margin_s = F.softmax(y_margin)\n",
    "        y_calc_s = F.softmax(y_calc)\n",
    "        \n",
    "        attribs = torch.cat((y_comp_s, y_echo_s, y_margin_s, y_calc_s), 1)\n",
    "        #print(attribs.shape)\n",
    "        features = torch.cat((comp_features, echo_features, margin_features, calc_features), 1)\n",
    "        g = torch.matmul(self.G_, attribs.reshape((-1, a, 1))) \n",
    "        #print(g.shape, self.G_.shape)\n",
    "        g = g.repeat((1, 1, num_labels))\n",
    "        \n",
    "        y = g * self.W_\n",
    "        #print(y.shape)\n",
    "        #y, _ = y.max(axis=2)\n",
    "        \n",
    "        y = torch.transpose(y, 1, 0)\n",
    "        \n",
    "        y = torch.sum(y, axis=0)\n",
    "        return (y_comp, y_echo, y_margin, y_calc, torch.sigmoid(y), features)\n",
    "\n",
    "# Model creation and definition of losses\n",
    "model = TypeNet(torch.FloatTensor(G), torch.FloatTensor(W))\n",
    "model.to(device)\n",
    "\n",
    "def projection_simplex_sort(v, z=1):\n",
    "\n",
    "    n_features = v.size(1)\n",
    "    u,_ = torch.sort(v, descending=True)\n",
    "    cssv = torch.cumsum(u,1) - z\n",
    "    ind = torch.arange(n_features).type_as(v) + 1\n",
    "    cond = u - cssv / ind > 0\n",
    "    #rho = ind[cond][-1]\n",
    "    rho,ind_rho = (ind*cond).max(1)\n",
    "    #theta = cssv[cond][-1] / float(rho)\n",
    "    theta = torch.gather(cssv,1,ind_rho[:,None]) / rho[:,None]\n",
    "    w = torch.clamp(v - theta, min=0)\n",
    "    return w\n",
    "\n",
    "criterion = torch.nn.CrossEntropyLoss(reduction='mean')\n",
    "\n",
    "criterion1 = torch.nn.L1Loss(reduction='sum')\n",
    "\n",
    "#criterion2 = MutualExclusivityLoss(15, 5)\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)\n",
    "\n",
    "#y_tr = torch.from_numpy(y_train).to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "Error(s) in loading state_dict for TypeNet:\n\tMissing key(s) in state_dict: \"comp_model.conv1.weight\", \"comp_model.conv1.bias\", \"comp_model.conv2.weight\", \"comp_model.conv2.bias\", \"comp_model.conv3.weight\", \"comp_model.conv3.bias\", \"comp_model.conv4.weight\", \"comp_model.conv4.bias\", \"comp_model.conv5.weight\", \"comp_model.conv5.bias\", \"comp_model.fc3.weight\", \"comp_model.fc3.bias\", \"echo_model.conv1.weight\", \"echo_model.conv1.bias\", \"echo_model.conv2.weight\", \"echo_model.conv2.bias\", \"echo_model.conv3.weight\", \"echo_model.conv3.bias\", \"echo_model.conv4.weight\", \"echo_model.conv4.bias\", \"echo_model.conv5.weight\", \"echo_model.conv5.bias\", \"echo_model.fc3.weight\", \"echo_model.fc3.bias\", \"calc_model.conv1.weight\", \"calc_model.conv1.bias\", \"calc_model.conv2.weight\", \"calc_model.conv2.bias\", \"calc_model.conv3.weight\", \"calc_model.conv3.bias\", \"calc_model.conv4.weight\", \"calc_model.conv4.bias\", \"calc_model.conv5.weight\", \"calc_model.conv5.bias\", \"calc_model.fc3.weight\", \"calc_model.fc3.bias\", \"margin_model.conv1.weight\", \"margin_model.conv1.bias\", \"margin_model.conv2.weight\", \"margin_model.conv2.bias\", \"margin_model.conv3.weight\", \"margin_model.conv3.bias\", \"margin_model.conv4.weight\", \"margin_model.conv4.bias\", \"margin_model.conv5.weight\", \"margin_model.conv5.bias\", \"margin_model.fc3.weight\", \"margin_model.fc3.bias\". \n\tUnexpected key(s) in state_dict: \"conv1.weight\", \"conv1.bias\", \"conv2.weight\", \"conv2.bias\", \"conv3.weight\", \"conv3.bias\", \"conv4.weight\", \"conv4.bias\", \"conv5.weight\", \"conv5.bias\", \"fc3.weight\", \"fc3.bias\". ",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_2800/2705133856.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Load existing model\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload_state_dict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf'../data/models/end_to_end_v2_CAM_3_weighted_mask.pt'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36mload_state_dict\u001b[0;34m(self, state_dict, strict)\u001b[0m\n\u001b[1;32m   1405\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0merror_msgs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1406\u001b[0m             raise RuntimeError('Error(s) in loading state_dict for {}:\\n\\t{}'.format(\n\u001b[0;32m-> 1407\u001b[0;31m                                self.__class__.__name__, \"\\n\\t\".join(error_msgs)))\n\u001b[0m\u001b[1;32m   1408\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0m_IncompatibleKeys\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmissing_keys\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0munexpected_keys\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1409\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mRuntimeError\u001b[0m: Error(s) in loading state_dict for TypeNet:\n\tMissing key(s) in state_dict: \"comp_model.conv1.weight\", \"comp_model.conv1.bias\", \"comp_model.conv2.weight\", \"comp_model.conv2.bias\", \"comp_model.conv3.weight\", \"comp_model.conv3.bias\", \"comp_model.conv4.weight\", \"comp_model.conv4.bias\", \"comp_model.conv5.weight\", \"comp_model.conv5.bias\", \"comp_model.fc3.weight\", \"comp_model.fc3.bias\", \"echo_model.conv1.weight\", \"echo_model.conv1.bias\", \"echo_model.conv2.weight\", \"echo_model.conv2.bias\", \"echo_model.conv3.weight\", \"echo_model.conv3.bias\", \"echo_model.conv4.weight\", \"echo_model.conv4.bias\", \"echo_model.conv5.weight\", \"echo_model.conv5.bias\", \"echo_model.fc3.weight\", \"echo_model.fc3.bias\", \"calc_model.conv1.weight\", \"calc_model.conv1.bias\", \"calc_model.conv2.weight\", \"calc_model.conv2.bias\", \"calc_model.conv3.weight\", \"calc_model.conv3.bias\", \"calc_model.conv4.weight\", \"calc_model.conv4.bias\", \"calc_model.conv5.weight\", \"calc_model.conv5.bias\", \"calc_model.fc3.weight\", \"calc_model.fc3.bias\", \"margin_model.conv1.weight\", \"margin_model.conv1.bias\", \"margin_model.conv2.weight\", \"margin_model.conv2.bias\", \"margin_model.conv3.weight\", \"margin_model.conv3.bias\", \"margin_model.conv4.weight\", \"margin_model.conv4.bias\", \"margin_model.conv5.weight\", \"margin_model.conv5.bias\", \"margin_model.fc3.weight\", \"margin_model.fc3.bias\". \n\tUnexpected key(s) in state_dict: \"conv1.weight\", \"conv1.bias\", \"conv2.weight\", \"conv2.bias\", \"conv3.weight\", \"conv3.bias\", \"conv4.weight\", \"conv4.bias\", \"conv5.weight\", \"conv5.bias\", \"fc3.weight\", \"fc3.bias\". "
     ]
    }
   ],
   "source": [
    "# Load existing model\n",
    "model.load_state_dict(torch.load(f'../data/models/end_to_end_v2_CAM_3_weighted_mask.pt'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/nn/functional.py:1794: UserWarning:\n",
      "\n",
      "nn.functional.tanh is deprecated. Use torch.tanh instead.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/nn/functional.py:718: UserWarning:\n",
      "\n",
      "Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448265233/work/c10/core/TensorImpl.h:1156.)\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:77: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:78: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:79: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:80: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/nn/modules/loss.py:97: UserWarning:\n",
      "\n",
      "Using a target size (torch.Size([8])) that is different to the input size (torch.Size([8, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch:  1 ; running-loss:  313.63643062114716\n",
      "Epoch:  2 ; running-loss:  306.28165459632874\n",
      "Epoch:  3 ; running-loss:  305.6020607948303\n",
      "Epoch:  4 ; running-loss:  304.4895740747452\n",
      "Epoch:  5 ; running-loss:  303.49196696281433\n",
      "Epoch:  6 ; running-loss:  305.03385305404663\n",
      "Epoch:  7 ; running-loss:  303.0276321172714\n",
      "Epoch:  8 ; running-loss:  300.97748613357544\n",
      "Epoch:  9 ; running-loss:  293.2420791387558\n",
      "Epoch:  10 ; running-loss:  294.8185197710991\n",
      "Epoch:  11 ; running-loss:  291.76898670196533\n",
      "Epoch:  12 ; running-loss:  289.33151906728745\n",
      "Epoch:  13 ; running-loss:  288.28163224458694\n",
      "Epoch:  14 ; running-loss:  286.7565311193466\n",
      "Epoch:  15 ; running-loss:  280.23563945293427\n",
      "Epoch:  16 ; running-loss:  281.01919159293175\n",
      "Epoch:  17 ; running-loss:  279.6402821838856\n",
      "Epoch:  18 ; running-loss:  285.3217976093292\n",
      "Epoch:  19 ; running-loss:  285.5874158143997\n",
      "Epoch:  20 ; running-loss:  285.5446127951145\n",
      "Epoch:  21 ; running-loss:  284.7681179046631\n",
      "Epoch:  22 ; running-loss:  279.61713540554047\n",
      "Epoch:  23 ; running-loss:  285.14750295877457\n",
      "Epoch:  24 ; running-loss:  279.273932993412\n",
      "Epoch:  25 ; running-loss:  285.14516058564186\n",
      "Epoch:  26 ; running-loss:  284.8618184328079\n",
      "Epoch:  27 ; running-loss:  279.1842626929283\n",
      "Epoch:  28 ; running-loss:  284.8910418152809\n",
      "Epoch:  29 ; running-loss:  284.92888271808624\n",
      "Epoch:  30 ; running-loss:  285.2514527440071\n",
      "Epoch:  31 ; running-loss:  285.29127210378647\n",
      "Epoch:  32 ; running-loss:  279.28736144304276\n",
      "Epoch:  33 ; running-loss:  284.5935949087143\n",
      "Epoch:  34 ; running-loss:  284.9188523888588\n",
      "Epoch:  35 ; running-loss:  284.40832233428955\n",
      "Epoch:  36 ; running-loss:  284.83557587862015\n",
      "Epoch:  37 ; running-loss:  278.8809372782707\n",
      "Epoch:  38 ; running-loss:  285.047570258379\n",
      "Epoch:  39 ; running-loss:  284.36133563518524\n",
      "Epoch:  40 ; running-loss:  284.5443112254143\n",
      "Epoch:  41 ; running-loss:  284.52034944295883\n",
      "Epoch:  42 ; running-loss:  278.7804820537567\n",
      "Epoch:  43 ; running-loss:  278.6784416437149\n",
      "Epoch:  44 ; running-loss:  278.8244639635086\n",
      "Epoch:  45 ; running-loss:  284.22152003645897\n",
      "Epoch:  46 ; running-loss:  278.56432417035103\n",
      "Epoch:  47 ; running-loss:  284.0965005159378\n",
      "Epoch:  48 ; running-loss:  278.786285340786\n",
      "Epoch:  49 ; running-loss:  283.9922715127468\n",
      "Epoch:  50 ; running-loss:  277.96547973155975\n",
      "Epoch:  51 ; running-loss:  278.3475948572159\n",
      "Epoch:  52 ; running-loss:  284.5098825991154\n",
      "Epoch:  53 ; running-loss:  284.4110255241394\n",
      "Epoch:  54 ; running-loss:  284.17454862594604\n",
      "Epoch:  55 ; running-loss:  284.87610536813736\n",
      "Epoch:  56 ; running-loss:  284.5668717622757\n",
      "Epoch:  57 ; running-loss:  284.41114434599876\n",
      "Epoch:  58 ; running-loss:  278.28323489427567\n",
      "Epoch:  59 ; running-loss:  283.9731987118721\n",
      "Epoch:  60 ; running-loss:  283.9414030313492\n",
      "Epoch:  61 ; running-loss:  277.8904777765274\n",
      "Epoch:  62 ; running-loss:  278.2780021429062\n",
      "Epoch:  63 ; running-loss:  283.8893843591213\n",
      "Epoch:  64 ; running-loss:  278.4540938735008\n",
      "Epoch:  65 ; running-loss:  283.7273026108742\n",
      "Epoch:  66 ; running-loss:  284.2182812988758\n",
      "Epoch:  67 ; running-loss:  284.05337911844254\n",
      "Epoch:  68 ; running-loss:  283.9646450281143\n",
      "Epoch:  69 ; running-loss:  284.2227168083191\n",
      "Epoch:  70 ; running-loss:  278.0428321957588\n",
      "Epoch:  71 ; running-loss:  284.1500133872032\n",
      "Epoch:  72 ; running-loss:  283.60367542505264\n",
      "Epoch:  73 ; running-loss:  277.7971004843712\n",
      "Epoch:  74 ; running-loss:  277.6794507801533\n",
      "Epoch:  75 ; running-loss:  277.81762900948524\n",
      "Epoch:  76 ; running-loss:  283.7727304697037\n",
      "Epoch:  77 ; running-loss:  284.0530762374401\n",
      "Epoch:  78 ; running-loss:  278.07606875896454\n",
      "Epoch:  79 ; running-loss:  283.8480424284935\n",
      "Epoch:  80 ; running-loss:  278.40562784671783\n",
      "Epoch:  81 ; running-loss:  277.74868816137314\n",
      "Epoch:  82 ; running-loss:  283.9340367913246\n",
      "Epoch:  83 ; running-loss:  278.0094087123871\n",
      "Epoch:  84 ; running-loss:  283.65441703796387\n",
      "Epoch:  85 ; running-loss:  277.8263699412346\n",
      "Epoch:  86 ; running-loss:  283.83920443058014\n",
      "Epoch:  87 ; running-loss:  277.4250248670578\n",
      "Epoch:  88 ; running-loss:  283.38869431614876\n",
      "Epoch:  89 ; running-loss:  283.5601850450039\n",
      "Epoch:  90 ; running-loss:  282.95024812221527\n",
      "Epoch:  91 ; running-loss:  283.5739048719406\n",
      "Epoch:  92 ; running-loss:  282.7632224559784\n",
      "Epoch:  93 ; running-loss:  283.7104807496071\n",
      "Epoch:  94 ; running-loss:  283.39028733968735\n",
      "Epoch:  95 ; running-loss:  283.0839975774288\n",
      "Epoch:  96 ; running-loss:  283.3581894040108\n",
      "Epoch:  97 ; running-loss:  277.31524389982224\n",
      "Epoch:  98 ; running-loss:  278.20262384414673\n",
      "Epoch:  99 ; running-loss:  277.3551627099514\n",
      "Epoch:  100 ; running-loss:  283.41737735271454\n",
      "Epoch:  101 ; running-loss:  277.7005639374256\n",
      "Epoch:  102 ; running-loss:  283.1263470053673\n",
      "Epoch:  103 ; running-loss:  283.1350765824318\n",
      "Epoch:  104 ; running-loss:  282.9922740459442\n",
      "Epoch:  105 ; running-loss:  282.9749284982681\n",
      "Epoch:  106 ; running-loss:  277.24730625748634\n",
      "Epoch:  107 ; running-loss:  282.4977744817734\n",
      "Epoch:  108 ; running-loss:  282.45854410529137\n",
      "Epoch:  109 ; running-loss:  283.10817566514015\n",
      "Epoch:  110 ; running-loss:  277.47514498233795\n",
      "Epoch:  111 ; running-loss:  283.42489659786224\n",
      "Epoch:  112 ; running-loss:  283.13137516379356\n",
      "Epoch:  113 ; running-loss:  277.35705721378326\n",
      "Epoch:  114 ; running-loss:  282.7798072248697\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_11495/537684392.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0mrunning_loss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0.0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0;31m#model.train()\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m     \u001b[0;32mfor\u001b[0m \u001b[0mdata\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mtraining_generator\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m         \u001b[0;31m#model.train(True)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m         \u001b[0mx_im_train\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"image\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/utils/data/dataloader.py\u001b[0m in \u001b[0;36m__next__\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    519\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sampler_iter\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    520\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_reset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 521\u001b[0;31m             \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_next_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    522\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_num_yielded\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    523\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_dataset_kind\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0m_DatasetKind\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mIterable\u001b[0m \u001b[0;32mand\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/utils/data/dataloader.py\u001b[0m in \u001b[0;36m_next_data\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    559\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_next_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    560\u001b[0m         \u001b[0mindex\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_next_index\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# may raise StopIteration\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 561\u001b[0;31m         \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_dataset_fetcher\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfetch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindex\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# may raise StopIteration\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    562\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_pin_memory\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    563\u001b[0m             \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_utils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpin_memory\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpin_memory\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py\u001b[0m in \u001b[0;36mfetch\u001b[0;34m(self, possibly_batched_index)\u001b[0m\n\u001b[1;32m     42\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mfetch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpossibly_batched_index\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     43\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mauto_collation\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 44\u001b[0;31m             \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0midx\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0midx\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mpossibly_batched_index\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     45\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     46\u001b[0m             \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mpossibly_batched_index\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/dl_env/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m     42\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mfetch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpossibly_batched_index\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     43\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mauto_collation\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 44\u001b[0;31m             \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0midx\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0midx\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mpossibly_batched_index\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     45\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     46\u001b[0m             \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mpossibly_batched_index\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/tmp/ipykernel_11495/2629120100.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, idx)\u001b[0m\n\u001b[1;32m     63\u001b[0m         \u001b[0;31m# labels[15] = list(self.types)[idx]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     64\u001b[0m         \u001b[0mim_name\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcases\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0midx\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mglob\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'*[0-9].jpg'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 65\u001b[0;31m         \u001b[0mim\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mim_name\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     66\u001b[0m         \u001b[0mmask\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mzeros\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mim\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     67\u001b[0m         \u001b[0mim\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mim\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m300\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m300\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minterpolation\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mINTER_CUBIC\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# perform training\n",
    "\n",
    "epoch = 0\n",
    "for epoch in range(20000):\n",
    "    running_loss = 0.0\n",
    "    #model.train()\n",
    "    for data in training_generator:\n",
    "        #model.train(True)\n",
    "        x_im_train = data[\"image\"]\n",
    "        if(np.shape(x_im_train)[0]==1):\n",
    "            continue\n",
    "        y_im_train = data[\"types\"].to(device)\n",
    "        x_im_train = x_im_train.to(device)\n",
    "        \n",
    "        y_comp = data[\"comp\"].to(device)\n",
    "        y_echo = data[\"echo\"].to(device)\n",
    "        y_margin = data[\"margin\"].to(device)\n",
    "        y_calc = data[\"calc\"].to(device)\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        (pred_comp, pred_echo, pred_margin, pred_calc, y_pred, features) = model(x_im_train)\n",
    "        # this needs work \n",
    "        loss = criterion1(y_pred.to(float), y_im_train.to(float)) + criterion(pred_comp, y_comp) + criterion(pred_echo, y_echo) + criterion(pred_margin, y_margin) + criterion(pred_calc, y_calc) \n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        model.G_.data = projection_simplex_sort(model.G_.data)\n",
    "        running_loss += loss.item()\n",
    "    epoch = epoch + 1\n",
    "    print(\"Epoch: \", epoch, \"; running-loss: \", running_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# save the trainined model\n",
    "\n",
    "torch.save(model.state_dict(), f'../data/models/end_to_end_v2_CAM_3_weighted_mask_new_separate_CNN.pt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/ahana/thyroid/data/test/benign\n",
      "/home/ahana/thyroid/data/test/malign\n",
      "number of data items:25\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:77: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:78: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:79: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n",
      "/home/ahana/anaconda3/envs/dl_env/lib/python3.7/site-packages/ipykernel_launcher.py:80: UserWarning:\n",
      "\n",
      "Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "## Run tests and save CAM images in folder \"heatmaps\" on test data\n",
    "def returnCAM(feature_conv, weight_linear, class_idx):\n",
    "    # generate the class activation maps upsample to 256x256\n",
    "    size_upsample = (256, 256)\n",
    "    bz, nc, h, w = feature_conv.shape\n",
    "    output_cam = []\n",
    "    for idx in class_idx:\n",
    "        cam = weight_linear[idx].dot(feature_conv.reshape((nc, h*w)))\n",
    "        cam = cam.reshape(h, w)\n",
    "        cam = cam - np.min(cam)\n",
    "        if (np.max(cam)) != 0:\n",
    "            cam_img = cam / (np.max(cam))\n",
    "        else:\n",
    "            cam_img = cam\n",
    "        cam_img = np.uint8(255 * cam_img)\n",
    "        output_cam.append(cv2.resize(cam_img, size_upsample))\n",
    "        \n",
    "    return output_cam\n",
    "\n",
    "parameters_test = {\n",
    "    \"batch_size\": 1,\n",
    "    \"shuffle\": False,\n",
    "}\n",
    "\n",
    "comp_params = list(model.comp_model.parameters())\n",
    "echo_params = list(model.echo_model.parameters())\n",
    "margin_params = list(model.margin_model.parameters())\n",
    "calc_params = list(model.calc_model.parameters())\n",
    "\n",
    "weight_comp_linear = np.squeeze(comp_params[-2].data.cpu().numpy())\n",
    "weight_echo_linear = np.squeeze(echo_params[-2].data.cpu().numpy())\n",
    "weight_margin_linear = np.squeeze(margin_params[-2].data.cpu().numpy())\n",
    "weight_calc_linear = np.squeeze(calc_params[-2].data.cpu().numpy())\n",
    "\n",
    "test_set = thyroidDataset(split='test')\n",
    "test_generator = torch.utils.data.DataLoader(test_set, **parameters_test)\n",
    "\n",
    "predicted = []\n",
    "ground_truth = []\n",
    "\n",
    "attr_pred = []\n",
    "attr_gt = []\n",
    "\n",
    "count = 0\n",
    "for data in test_generator:\n",
    "    y_comp = data[\"comp\"].to(device)\n",
    "    y_echo = data[\"echo\"].to(device)\n",
    "    y_calc = data[\"calc\"].to(device)\n",
    "    y_margin = data[\"margin\"].to(device)\n",
    "    x_im_test = data[\"image\"]\n",
    "    y_im_test = data[\"types\"].to(device)\n",
    "    x_im_test = x_im_test.to(device)\n",
    "    \n",
    "    (pred_comp, pred_echo, pred_margin, pred_calc, y_pred, features) = model(x_im_test)\n",
    "    features_comp = features[:,:256,:,:]\n",
    "    features_echo = features[:,256:512,:,:]\n",
    "    features_margin = features[:,512:768,:,:]\n",
    "    features_calc = features[:,768:1024,:,:]\n",
    "    pred_comp = pred_comp.detach().cpu().numpy()\n",
    "    pred_comp = np.argmax(pred_comp, axis=1)\n",
    "    pred_echo = pred_echo.detach().cpu().numpy()\n",
    "    pred_echo = np.argmax(pred_echo, axis=1)\n",
    "    pred_calc = pred_calc.detach().cpu().numpy()\n",
    "    pred_calc = np.argmax(pred_calc, axis=1)\n",
    "    pred_margin = pred_margin.detach().cpu().numpy()\n",
    "    pred_margin = np.argmax(pred_margin, axis=1)\n",
    "    attributes_pred = [pred_comp, pred_echo, pred_margin, pred_calc]\n",
    "    attr_pred.append(np.asarray(attributes_pred))\n",
    "    attributes = [y_comp.detach().cpu().numpy(), y_echo.detach().cpu().numpy(), y_margin.detach().cpu().numpy(), y_calc.detach().cpu().numpy()]\n",
    "    attr_gt.append(np.asarray(attributes))\n",
    "    \n",
    "    #attr_gt.append(data[\"types\"].detach().cpu().numpy())  \n",
    "    attrib_list = [i for i in range(0, 5)]\n",
    "    # saving the CAM image with heatmaps\n",
    "    CAMs = returnCAM(features_comp.detach().cpu().numpy(), weight_comp_linear, attrib_list)\n",
    "    img = data[\"image\"][0][:3,:,:].cpu().permute(1, 2, 0).numpy()\n",
    "    #img = data[\"image\"][0].cpu().permute(1, 2, 0).numpy()\n",
    "    img = img * 255\n",
    "    height, width, c = img.shape\n",
    "    for i in range(0, 5):\n",
    "        heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)\n",
    "        result = heatmap * 0.2 + img\n",
    "        file_name = data[\"name\"][0].split('/')[-1]\n",
    "        file_name = file_name.split('.')[0]\n",
    "        cv2.imwrite(f'heatmaps/{file_name}_comp_{i}.png', result)\n",
    "        \n",
    "    attrib_list = [i for i in range(0, 5)]\n",
    "    # saving the CAM image with heatmaps\n",
    "    CAMs = returnCAM(features_echo.detach().cpu().numpy(), weight_echo_linear, attrib_list)\n",
    "    img = data[\"image\"][0][:3,:,:].cpu().permute(1, 2, 0).numpy()\n",
    "    #img = data[\"image\"][0].cpu().permute(1, 2, 0).numpy()\n",
    "    img = img * 255\n",
    "    height, width, c = img.shape\n",
    "    for i in range(0, 5):\n",
    "        heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)\n",
    "        result = heatmap * 0.2 + img\n",
    "        file_name = data[\"name\"][0].split('/')[-1]\n",
    "        file_name = file_name.split('.')[0]\n",
    "        cv2.imwrite(f'heatmaps/{file_name}_echo_{i}.png', result)\n",
    "        \n",
    "    attrib_list = [i for i in range(0, 3)]\n",
    "    # saving the CAM image with heatmaps\n",
    "    CAMs = returnCAM(features_calc.detach().cpu().numpy(), weight_calc_linear, attrib_list)\n",
    "    img = data[\"image\"][0][:3,:,:].cpu().permute(1, 2, 0).numpy()\n",
    "    #img = data[\"image\"][0].cpu().permute(1, 2, 0).numpy()\n",
    "    img = img * 255\n",
    "    height, width, c = img.shape\n",
    "    for i in range(0, 3):\n",
    "        heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)\n",
    "        result = heatmap * 0.2 + img\n",
    "        file_name = data[\"name\"][0].split('/')[-1]\n",
    "        file_name = file_name.split('.')[0]\n",
    "        cv2.imwrite(f'heatmaps/{file_name}_calc_{i}.png', result)\n",
    "        \n",
    "    attrib_list = [i for i in range(0, 5)]\n",
    "    # saving the CAM image with heatmaps\n",
    "    CAMs = returnCAM(features_margin.detach().cpu().numpy(), weight_margin_linear, attrib_list)\n",
    "    img = data[\"image\"][0][:3,:,:].cpu().permute(1, 2, 0).numpy()\n",
    "    #img = data[\"image\"][0].cpu().permute(1, 2, 0).numpy()\n",
    "    img = img * 255\n",
    "    height, width, c = img.shape\n",
    "    for i in range(0, 5):\n",
    "        heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)\n",
    "        result = heatmap * 0.2 + img\n",
    "        file_name = data[\"name\"][0].split('/')[-1]\n",
    "        file_name = file_name.split('.')[0]\n",
    "        cv2.imwrite(f'heatmaps/{file_name}_margin_{i}.png', result)\n",
    "    \n",
    "    # End of CAM saving code\n",
    "    predicted.append(np.squeeze(y_pred.detach().cpu().numpy()))\n",
    "    ground_truth.append(np.squeeze(y_im_test.detach().cpu().numpy()))\n",
    "    count += 1\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Just keeping this code even though it is untidy\n",
    "#print(attr_gt.shape, attr_pred.shape)\n",
    "#viz_attr = (np.concatenate((attr_gt, attr_pred), axis=0)>0.5)*1\n",
    "# dummy_zeros = np.zeros((attr_pred.shape[0], attr_pred.shape[1]))\n",
    "# viz_attr = np.dstack([attr_gt, attr_pred, dummy_zeros]).reshape(attr_pred.shape[0],-1)\n",
    "# #viz_attr[viz_attr > 0.5] = 1\n",
    "# #viz_attr[viz_attr != 1] = 0\n",
    "# plt.imshow(viz_attr)\n",
    "# major_ticks = np.arange(0,45,1)\n",
    "# plt.xticks(major_ticks)\n",
    "# plt.grid(axis='x')\n",
    "# plt.title(\"Left col: gt attr; Right col: pred attr interleaved with a blank one between each pair of attributes\")\n",
    "# fig = plt.gcf()\n",
    "# fig.set_size_inches(18.5, 10.5)\n",
    "\n",
    "# predicted = np.expand_dims(np.array(predicted), 1)\n",
    "# ground_truth = np.expand_dims(np.array(ground_truth), 1)\n",
    "# viz = (np.concatenate((ground_truth, predicted), axis=1)>0.5)*1\n",
    "\n",
    "# plt.imshow(viz)\n",
    "# plt.title(\"Left col: gt; Right col: pred\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Atrribute errors:\n",
      " [0.64 0.6  0.36 0.96]\n",
      "Type prediction error:\n",
      " 0.72\n"
     ]
    }
   ],
   "source": [
    "# printing the accuracy of attribute and type prediction\n",
    "#attr_pred = np.array(attr_pred)[:,0,:,0]\n",
    "#attr_pred_binary = np.zeros((attr_pred.shape[0], attr_pred.shape[1]))\n",
    "#attr_pred_binary[attr_pred > 0.5] = 1\n",
    "#attr_pred_binary[attr_pred != 1] = 0\n",
    "#attr_gt= np.array(attr_gt)[:,0,:]\n",
    "#error = np.abs(attr_pred_binary - attr_gt)\n",
    "#error = 100 * np.sum(error, axis=0) / error.shape[0]\n",
    "\n",
    "attr_pred = np.array(attr_pred).flatten()\n",
    "attr_gt = np.array(attr_gt).flatten()\n",
    "error = np.zeros((attr_pred.shape[0]))\n",
    "error[attr_pred != attr_gt] = 1\n",
    "error = error.reshape(-1, 4)\n",
    "print(\"Atrribute errors:\\n\",np.sum(error, axis = 0)/25)\n",
    "\n",
    "predicted = np.expand_dims(np.array(predicted), 1)\n",
    "ground_truth = np.expand_dims(np.array(ground_truth), 1)\n",
    "viz = (np.concatenate((ground_truth, predicted), axis=1)>0.5)*1\n",
    "err_perc = np.sum(np.abs(viz[:, 0] - viz[:, 1])) / np.size(viz[:, 0])\n",
    "print(\"Type prediction error:\\n\", err_perc)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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2F5he8WmhzdNTKEA2jLWPr5+w9aJKRToMfS97pMqpNmeJekrkt4REY+03hcRISmGahcR50fE/IiYP2P3GRNQKYB2dKKP77PXlAmoFOgt9L3ulCak2Z6m6gX0hYj4arn5Ptv7lfv3k3R/U/UBGBv6ONAM+l73ShHSbFvSBZJ+EREfb7nrakknlr9PlPS1ia4NwNjRx4Fm0Pe6W5bLQr1Y0psk/dz2ynLbeyV9VNLltk+WdKek1zZTHoAxoo8DzaDvdbEUIS0ifijJw9z90omsBUDn0ceBZtD3uluKw50AAAB4shQjaZ100PRHddHspbXazLnstLaWdfvrzqvdpm/JkB+g2aH1/bWbAACALsdIGgAAQEKENAAAgIQIaQAAAAkR0gAAABIipAEAACRESAMAAEiIkAYAAJAQIQ0AACAhQhoAAEBChDQAAICECGkAAAAJEdIAAAAS6rkLrP/8oX1qXzB94RGrxqma7a1d6NptNi8fh0KALnXrTTN1VN+8pssAgHHHSBoAAEBChDQAAICECGkAAAAJEdIAAAASIqQBAAAkREgDAABIiJAGAACQECENAAAgIUIaAABAQoQ0AACAhAhpAAAACRHSAAAAEuq5C6zvustva18wfcmyQ9pb2OyltZv0LYnabdb3124C9KyDD92oxYtX1mrDBdkBdCNG0gAAABIipAEAACRESAMAAEiIkAYAAJAQIQ0AACAhQhoAAEBChDQAAICECGkAAAAJEdIAAAASIqQBAAAkREgDAABIiJAGAACQECENAAAgoalNF5DBwiNWNV0CAADAkzCSBgAAkBAhDQAAICFCGgAAQEKENAAAgIQIaQAAAAkR0gAAABIipAEAACRESAMAAEiIkAYAAJAQIQ0AACAhQhoAAEBChDQAAICEHBFN19BRttdLunOIu/aW9MAElzOUduo4MCL2GY9igG6zgz4utde/dtSGvgcU7F8nXs+FtOHYXhER86kD6F3t9C/6JDA2WfpQljo6icOdAAAACRHSAAAAEppMIe38pgsostQB9KJ2+hd9EhibLH0oSx0dM2nOSQMAAOgmk2kkDQAAoGsQ0gAAABLquZBm+2jbv7J9m+33DHH/TrYvK/ffYPugDi//ANvfs32L7VW23z7ENEfa/o3tleXnf3WyBmCyGanfD9PmQtv32755vOsDegH714nXUyHN9hRJ50p6paS5kk6wPXfQZCdLeiginiXpHElnd7iMLZLeGRFzJR0u6fQhapCkpRExr/x8oMM1AJPGKPv9UBZJOnocSwN6BvvXZvRUSJN0mKTbIuKOiHhC0pckHTtommMlXVz+vkLSS227UwVExLqIuLH83S/pF5L279T8AWxnNP1+OxGxRNKD410c0CPYvzag10La/pLuavn/bm3/BG6bJiK2SPqNpL3Go5gy1PsCSTcMcfcRtn9m+1u2DxmP5QOTxGj6PYCxYf/agKlNF9CrbM+S9BVJ74iIRwbdfaOq64U9avsYSVdJevYElwgAQNeZTPvXXhtJu0fSAS3/P73cNuQ0tqdK2k3Shk4WYXuaqhfQFyLiq4Pvj4hHIuLR8vc1kqbZ3ruTNQCTyGj6PYCxYf/agF4LacslPdv2M2xPl3S8pKsHTXO1pBPL36+W9N3o4Df6luPvF0j6RUR8fJhpnjZwnN72Yaqeh46+kIFJZDT9HsDYsH9tQE8d7oyILbbPkLRY0hRJF0bEKtsfkLQiIq5W9QRfYvs2VScNH9/hMl4s6U2Sfm57ZbntvZJmlxrPU/XifZvtLZJ+K+n4Tr6QgclkuH4/Ujvbl0o6UtLetu+WdGZEXDCuxQJdiv1rM7gsFAAAQEK9drgTAACgJxDSAAAAEiKkAQAAJERIAwAASIiQBgAAkBAhDQAAICFCGgAAQEL/HzsAzZhTYGTSAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x720 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## ALL this should be on TEST set\n",
    "# Visualizing the results for attributes and labels\n",
    "attr_pred = attr_pred.reshape(-1, 4)\n",
    "attr_gt = attr_gt.reshape(-1, 4)\n",
    "\n",
    "fig = plt.figure(figsize=(10, 10))\n",
    "ax1 = fig.add_subplot(231)    # The big subplot\n",
    "ax2 = fig.add_subplot(232)\n",
    "ax3 = fig.add_subplot(233)\n",
    "\n",
    "\n",
    "\n",
    "ax1.imshow(attr_gt)\n",
    "ax1.set_title(\"Attributes Ground Truth\")\n",
    "#ax2.imshow((attr_pred > 0.5) * 1)\n",
    "#ax2.set_title(\"Attributes Predicted and Thresholded\")\n",
    "\n",
    "ax2.imshow(viz)\n",
    "ax2.set_title(\"Label (malign or benign)\")\n",
    "\n",
    "#ax4.imshow(attr_gt)\n",
    "#ax4.set_title(\"Attributes Ground Truth\")\n",
    "ax3.imshow(attr_pred )\n",
    "ax3.set_title(\"Attributes Predicted\")\n",
    "\n",
    "#ax6.imshow(viz)\n",
    "#ax6.set_title(\"Label (malign or benign)\")\n",
    "\n",
    "\n",
    "plt.tight_layout(pad=0.6, w_pad=0, h_pad=1.0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Attribute probability')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualizing G matrix\n",
    "plt.imshow(model.G_.data.detach().cpu().numpy())\n",
    "plt.title(\"Group attribute probabilites (G Matrix)\")\n",
    "plt.ylabel(\"Group\")\n",
    "plt.xlabel(\"Attribute probability\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'W values')"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualizing W Matrix\n",
    "plt.imshow(model.W_.data.detach().cpu().numpy())\n",
    "plt.title(\"Group contribution to label pred (W Matrix)\")\n",
    "plt.ylabel(\"Group\")\n",
    "plt.xlabel(\"W values\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Legacy CAM code\n",
    "# class CamExtractor():\n",
    "#     \"\"\"\n",
    "#         Extracts cam features from the model\n",
    "#     \"\"\"\n",
    "#     def __init__(self, model, target_layer):\n",
    "#         self.model = model\n",
    "#         self.target_layer = target_layer\n",
    "        \n",
    "\n",
    "#     def forward_pass_on_convolutions(self, x):\n",
    "#         \"\"\"\n",
    "#             Does a forward pass on convolutions, hooks the function at given layer\n",
    "#         \"\"\"\n",
    "#         conv_output = None\n",
    "#         module_pos = 1\n",
    "#         for module in self.model.features:\n",
    "#             x = model.pool(module(x))  # Forward\n",
    "#             if int(module_pos) == self.target_layer:\n",
    "#                 conv_output = x  # Save the convolution output on that layer\n",
    "#             module_pos += 1\n",
    "#         return conv_output, x\n",
    "\n",
    "#     def forward_pass(self, x):\n",
    "#         \"\"\"\n",
    "#             Does a full forward pass on the model\n",
    "#         \"\"\"\n",
    "#         conv_output, _ = self.forward_pass_on_convolutions(x)\n",
    "#         x = F.interpolate(x, (300, 300), mode='bilinear')\n",
    "#         (x_, _) = model(x)\n",
    "#         return conv_output, x_\n",
    "\n",
    "    \n",
    "# extractor = CamExtractor(model, 4)\n",
    "\n",
    "# count = 0\n",
    "# for data in test_generator:\n",
    "#     y_im_test = data[\"labels\"][:, 15].to(device)\n",
    "#     x_im_test = data[\"image\"]\n",
    "#     x_im_test = x_im_test.to(device)\n",
    "\n",
    "\n",
    "#     [conv_out, x] = extractor.forward_pass_on_convolutions(x_im_test)\n",
    "#     target = conv_out\n",
    "#     cam = np.ones((target.shape[1:]), dtype=np.float32)\n",
    "#     sal_map = np.zeros((224, 224))\n",
    "#     for i in range(np.shape(target)[1]):\n",
    "#                 # Unsqueeze to 4D\n",
    "#                 saliency_map = torch.unsqueeze(torch.unsqueeze(target[0, i, :, :],0),0)\n",
    "#                 # Upsampling to input size\n",
    "#                 saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)\n",
    "#                 if saliency_map.max() == saliency_map.min():\n",
    "#                     continue\n",
    "#                 # Scale between 0-1\n",
    "#                 norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())\n",
    "#                 inim = F.interpolate(x_im_test, size=(224, 224), mode='bilinear')\n",
    "#                 w = extractor.forward_pass(inim * norm_saliency_map)[1]\n",
    "#                 sal_map += w.cpu().data.numpy() * norm_saliency_map.cpu().data.numpy()[0, 0, :, :]\n",
    "#                 cam += w.cpu().data.numpy() * target[0, i, :, :].cpu().data.numpy()\n",
    "\n",
    "\n",
    "#     fig = plt.figure(figsize=(15, 10))\n",
    "\n",
    "#     ax0 = fig.add_subplot(131)\n",
    "#     ax0.imshow(np.transpose(np.squeeze(x_im_test.cpu().detach().numpy()), (1, 2, 0)))\n",
    "\n",
    "#     ax1 = fig.add_subplot(132)\n",
    "#     ax1.imshow(sal_map)\n",
    "\n",
    "#     ax2 = fig.add_subplot(133)\n",
    "#     ax2.imshow(cam[0, :, :])\n",
    "#     count += 1\n",
    "#     if(count == 10):\n",
    "#         break\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f1045a1c100>"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Legacy CAM code\n",
    "# fig = plt.figure()\n",
    "# ax0 = fig.add_subplot(121)\n",
    "# ax0.imshow(sal_map)\n",
    "\n",
    "# ax1 = fig.add_subplot(122)\n",
    "# ax1.imshow(np.transpose(np.squeeze(x_im_test.cpu().detach().numpy()), (1, 2, 0)))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Sankey plot code for G matrix\n",
    "source = [i % a for i in range(z*a)]\n",
    "target = [(i // 15) + 15 for i in range(z*a)]\n",
    "G[G < 0.1] = 0.0\n",
    "value = G.flatten().tolist()\n",
    "#print(len(source), len(target), len(value))\n",
    "\n",
    "color_node = [\n",
    "              '#808080', '#808080', '#808080', '#808080', '#808080',\n",
    "              '#808080', '#808080', '#808080', '#808080', '#808080',\n",
    "              '#808080', '#808080', '#808080', '#808080', '#808080',\n",
    "              '#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#FF00FF',\n",
    "              '#00CED1', '#FF8C00', '#BDB76B', '#2F4F4F', '#B8860B'\n",
    "              ]\n",
    "\n",
    "color_link = []\n",
    "link_colors = ['#F08080', '#FFFACD', '#98FB98', '#87CEFA', '#EE82EE',\n",
    "              '#AFEEEE', '#FFA500', '#F0E68C', '#708090', '#DAA520']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "        <script type=\"text/javascript\">\n",
       "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
       "        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
       "        if (typeof require !== 'undefined') {\n",
       "        require.undef(\"plotly\");\n",
       "        define('plotly', function(require, exports, module) {\n",
       "            /**\n",
       "* plotly.js v1.49.4\n",
       "* Copyright 2012-2019, Plotly, Inc.\n",
       "* All rights reserved.\n",
       "* Licensed under the MIT license\n",
       "*/\n",
       "        });\n",
       "        require(['plotly'], function(Plotly) {\n",
       "            window._Plotly = Plotly;\n",
       "        });\n",
       "        }\n",
       "        </script>\n",
       "        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "link": {
          "color": [
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#F08080",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#FFFACD",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#98FB98",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#87CEFA",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#EE82EE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#AFEEEE",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500",
           "#FFA500"
          ],
          "source": [
           0,
           1,
           2,
           3,
           4,
           5,
           6,
           7,
           8,
           9,
           10,
           11,
           12,
           13,
           14,
           15,
           16,
           17,
           0,
           1,
           2,
           3,
           4,
           5,
           6,
           7,
           8,
           9,
           10,
           11,
           12,
           13,
           14,
           15,
           16,
           17,
           0,
           1,
           2,
           3,
           4,
           5,
           6,
           7,
           8,
           9,
           10,
           11,
           12,
           13,
           14,
           15,
           16,
           17,
           0,
           1,
           2,
           3,
           4,
           5,
           6,
           7,
           8,
           9,
           10,
           11,
           12,
           13,
           14,
           15,
           16,
           17,
           0,
           1,
           2,
           3,
           4,
           5,
           6,
           7,
           8,
           9,
           10,
           11,
           12,
           13,
           14,
           15,
           16,
           17,
           0,
           1,
           2,
           3,
           4,
           5,
           6,
           7,
           8,
           9,
           10,
           11,
           12,
           13,
           14,
           15,
           16,
           17,
           0,
           1,
           2,
           3,
           4,
           5,
           6,
           7,
           8,
           9,
           10,
           11,
           12,
           13,
           14,
           15,
           16,
           17
          ],
          "target": [
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           15,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           16,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           17,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           18,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           19,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           20,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           21,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           22,
           23,
           23,
           23,
           23,
           23,
           23
          ],
          "value": [
           0,
           0.5742798092550099,
           1.4462932919096387,
           0,
           0.7219867252701788,
           0,
           0,
           0,
           0,
           1.804223003255076,
           0.8645586560432021,
           0,
           0,
           0,
           0,
           2.1527651290272996,
           0,
           0.41687069719833925,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0.37990818804354026,
           0,
           0.11831243208781775,
           0,
           0,
           1.4083036422669302,
           0,
           0,
           0,
           0.30041032735881085,
           0,
           0,
           0.5587496338520443,
           0,
           0.16235297858856185,
           0,
           0,
           1.5911560340328275,
           2.6510117066906713,
           0.199992858273891,
           0.19306410217780265,
           0.34365514757870547,
           0,
           0.4706098312438546,
           0.47869614076475986,
           0.11210612593194598,
           0,
           0,
           0.9205251687588732,
           0,
           1.1142486176440627,
           0.836756903901324,
           0,
           0,
           0,
           0,
           0.11717463149043145,
           0.5537502531913172,
           1.099415670998657,
           0,
           0,
           0,
           0,
           3.0082686619556345,
           0.17209201983747568,
           0,
           0,
           0.5181939426034305,
           0,
           0,
           0.2501111664109562,
           0,
           0,
           0,
           0.7401347802186962,
           0.3748203274267869,
           0,
           0.8266382972906902,
           0,
           0.8065470740801478,
           0,
           0.8296045009916732,
           0.17501637960141941,
           0.408087347252248,
           1.7328710436870007,
           0,
           1.135373514828572,
           0,
           0,
           0,
           0,
           0,
           0,
           0.2047848350561543,
           0,
           0,
           0,
           1.7281551753010391,
           1.1209961553121288,
           0.29768235887319094,
           0.7775237838265311,
           0,
           0,
           0,
           0.3704282386968272,
           1.7259924253795693,
           0.3048858754690262,
           0,
           0.6416033084986074,
           1.183449316150125,
           0,
           0,
           1.3731630099785896,
           0,
           0,
           0.4424308025491873,
           0,
           0,
           0,
           0,
           0
          ]
         },
         "node": {
          "color": [
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#808080",
           "#FF0000",
           "#FFFF00",
           "#00FF00",
           "#00FFFF",
           "#FF00FF",
           "#00CED1",
           "#FF8C00",
           "#BDB76B",
           "#2F4F4F",
           "#B8860B"
          ],
          "label": [
           "cystic",
           "mostly solid",
           "solid",
           "spongiform",
           "hyper",
           "hypo",
           "iso",
           "marked",
           "ill-defined",
           "micro",
           "spiculated",
           "smooth",
           "macro",
           "micro",
           "non",
           "G1",
           "G2",
           "G3",
           "G4",
           "G5",
           "G6",
           "G7",
           "G8",
           "G9",
           "G10"
          ],
          "pad": 15,
          "thickness": 20
         },
         "type": "sankey"
        }
       ],
       "layout": {
        "font": {
         "size": 10
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Basic Sankey Diagram"
        }
       }
      },
      "text/html": [
       "<div>\n",
       "        \n",
       "        \n",
       "            <div id=\"0e43fc5c-225e-41ce-a892-e07d53c26e00\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>\n",
       "            <script type=\"text/javascript\">\n",
       "                require([\"plotly\"], function(Plotly) {\n",
       "                    window.PLOTLYENV=window.PLOTLYENV || {};\n",
       "                    \n",
       "                if (document.getElementById(\"0e43fc5c-225e-41ce-a892-e07d53c26e00\")) {\n",
       "                    Plotly.newPlot(\n",
       "                        '0e43fc5c-225e-41ce-a892-e07d53c26e00',\n",
       "                        [{\"link\": {\"color\": [\"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#F08080\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#FFFACD\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#98FB98\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#87CEFA\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#EE82EE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#AFEEEE\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\", \"#FFA500\"], \"source\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], \"target\": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23], \"value\": [0.0, 0.5742798092550099, 1.4462932919096387, 0.0, 0.7219867252701788, 0.0, 0.0, 0.0, 0.0, 1.804223003255076, 0.8645586560432021, 0.0, 0.0, 0.0, 0.0, 2.1527651290272996, 0.0, 0.41687069719833925, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37990818804354026, 0.0, 0.11831243208781775, 0.0, 0.0, 1.4083036422669302, 0.0, 0.0, 0.0, 0.30041032735881085, 0.0, 0.0, 0.5587496338520443, 0.0, 0.16235297858856185, 0.0, 0.0, 1.5911560340328275, 2.6510117066906713, 0.199992858273891, 0.19306410217780265, 0.34365514757870547, 0.0, 0.4706098312438546, 0.47869614076475986, 0.11210612593194598, 0.0, 0.0, 0.9205251687588732, 0.0, 1.1142486176440627, 0.836756903901324, 0.0, 0.0, 0.0, 0.0, 0.11717463149043145, 0.5537502531913172, 1.099415670998657, 0.0, 0.0, 0.0, 0.0, 3.0082686619556345, 0.17209201983747568, 0.0, 0.0, 0.5181939426034305, 0.0, 0.0, 0.2501111664109562, 0.0, 0.0, 0.0, 0.7401347802186962, 0.3748203274267869, 0.0, 0.8266382972906902, 0.0, 0.8065470740801478, 0.0, 0.8296045009916732, 0.17501637960141941, 0.408087347252248, 1.7328710436870007, 0.0, 1.135373514828572, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2047848350561543, 0.0, 0.0, 0.0, 1.7281551753010391, 1.1209961553121288, 0.29768235887319094, 0.7775237838265311, 0.0, 0.0, 0.0, 0.3704282386968272, 1.7259924253795693, 0.3048858754690262, 0.0, 0.6416033084986074, 1.183449316150125, 0.0, 0.0, 1.3731630099785896, 0.0, 0.0, 0.4424308025491873, 0.0, 0.0, 0.0, 0.0, 0.0]}, \"node\": {\"color\": [\"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#808080\", \"#FF0000\", \"#FFFF00\", \"#00FF00\", \"#00FFFF\", \"#FF00FF\", \"#00CED1\", \"#FF8C00\", \"#BDB76B\", \"#2F4F4F\", \"#B8860B\"], \"label\": [\"cystic\", \"mostly solid\", \"solid\", \"spongiform\", \"hyper\", \"hypo\", \"iso\", \"marked\", \"ill-defined\", \"micro\", \"spiculated\", \"smooth\", \"macro\", \"micro\", \"non\", \"G1\", \"G2\", \"G3\", \"G4\", \"G5\", \"G6\", \"G7\", \"G8\", \"G9\", \"G10\"], \"pad\": 15, \"thickness\": 20}, \"type\": \"sankey\"}],\n",
       "                        {\"font\": {\"size\": 10}, \"template\": {\"data\": {\"bar\": [{\"error_x\": {\"color\": \"#2a3f5f\"}, \"error_y\": {\"color\": \"#2a3f5f\"}, \"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"bar\"}], \"barpolar\": [{\"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"barpolar\"}], \"carpet\": [{\"aaxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"baxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"type\": \"carpet\"}], \"choropleth\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"choropleth\"}], \"contour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"contour\"}], \"contourcarpet\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"contourcarpet\"}], \"heatmap\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmap\"}], \"heatmapgl\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmapgl\"}], \"histogram\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"histogram\"}], \"histogram2d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2d\"}], \"histogram2dcontour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2dcontour\"}], \"mesh3d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"mesh3d\"}], \"parcoords\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"parcoords\"}], \"scatter\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter\"}], \"scatter3d\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter3d\"}], \"scattercarpet\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattercarpet\"}], \"scattergeo\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergeo\"}], \"scattergl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergl\"}], \"scattermapbox\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattermapbox\"}], \"scatterpolar\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolar\"}], \"scatterpolargl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolargl\"}], \"scatterternary\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterternary\"}], \"surface\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"surface\"}], \"table\": [{\"cells\": {\"fill\": {\"color\": \"#EBF0F8\"}, \"line\": {\"color\": \"white\"}}, \"header\": {\"fill\": {\"color\": \"#C8D4E3\"}, \"line\": {\"color\": \"white\"}}, \"type\": \"table\"}]}, \"layout\": {\"annotationdefaults\": {\"arrowcolor\": \"#2a3f5f\", \"arrowhead\": 0, \"arrowwidth\": 1}, \"colorscale\": {\"diverging\": [[0, \"#8e0152\"], [0.1, \"#c51b7d\"], [0.2, \"#de77ae\"], [0.3, \"#f1b6da\"], [0.4, \"#fde0ef\"], [0.5, \"#f7f7f7\"], [0.6, \"#e6f5d0\"], [0.7, \"#b8e186\"], [0.8, \"#7fbc41\"], [0.9, \"#4d9221\"], [1, \"#276419\"]], \"sequential\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"sequentialminus\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]]}, \"colorway\": [\"#636efa\", \"#EF553B\", \"#00cc96\", \"#ab63fa\", \"#FFA15A\", \"#19d3f3\", \"#FF6692\", \"#B6E880\", \"#FF97FF\", \"#FECB52\"], \"font\": {\"color\": \"#2a3f5f\"}, \"geo\": {\"bgcolor\": \"white\", \"lakecolor\": \"white\", \"landcolor\": \"#E5ECF6\", \"showlakes\": true, \"showland\": true, \"subunitcolor\": \"white\"}, \"hoverlabel\": {\"align\": \"left\"}, \"hovermode\": \"closest\", \"mapbox\": {\"style\": \"light\"}, \"paper_bgcolor\": \"white\", \"plot_bgcolor\": \"#E5ECF6\", \"polar\": {\"angularaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"radialaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"scene\": {\"xaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"yaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"zaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}}, \"shapedefaults\": {\"line\": {\"color\": \"#2a3f5f\"}}, \"ternary\": {\"aaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"baxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"caxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"title\": {\"x\": 0.05}, \"xaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}, \"yaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}}}, \"title\": {\"text\": \"Basic Sankey Diagram\"}},\n",
       "                        {\"responsive\": true}\n",
       "                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('0e43fc5c-225e-41ce-a892-e07d53c26e00');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })\n",
       "                };\n",
       "                });\n",
       "            </script>\n",
       "        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i in range(z):\n",
    "    color_link.extend([link_colors[i]] * a)\n",
    "#print(color_link)\n",
    "fig = go.Figure(data=[go.Sankey(\n",
    "    node = dict(\n",
    "      pad = 15,\n",
    "      thickness = 20,\n",
    "      color = color_node,\n",
    "      label = [\"cystic\", \"mostly solid\", \"solid\", \"spongiform\",\n",
    "               \"hyper\", \"hypo\", \"iso\", \"marked\",\n",
    "               \"ill-defined\", \"micro\", \"spiculated\", \"smooth\",\n",
    "               \"macro\", \"micro\", \"non\",\n",
    "               \"G1\", \"G2\", \"G3\", \"G4\", \"G5\",\n",
    "               \"G6\", \"G7\", \"G8\", \"G9\", \"G10\"],\n",
    "    ),\n",
    "    link = dict(\n",
    "      source = source,\n",
    "      target = target,\n",
    "      value = value,\n",
    "      color = color_link\n",
    "  ))])\n",
    "fig.update_layout(title_text=\"Basic Sankey Diagram\", font_size=10)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}