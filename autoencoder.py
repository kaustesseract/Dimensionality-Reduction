
from __future__ import print_function, division
from torch.utils.data.dataset import Dataset
from PIL import Image
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torch.autograd import Variable
import copy
from torchvision.utils import save_image
from torchvision import IntTensor

start_time = time.time()

plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize the data
    ]),
}

data_dir = "C:\\Users\\kaust\\Desktop\\Prep for MS\\AWL\\Autoencoder\\morning\\0\\" # Provide path

class get_data(Dataset):
    def __init__(self, folder_path='folder_path'):
        list_imgs = sorted(os.listdir(folder_path))
        self.images =[os.path.join(folder_path, name) for name in list_imgs] # joining the path with the images in the folder
        self.transforms = transforms.Compose([transforms.ToTensor()]) # creating trnasform object for converting to tensor
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index] 
        img = Image.open(path)
        img = img.convert('RGB')
        img = self.transforms(img)
        return img, path
    

image_datasets = {x: get_data(data_dir) for x in ['train']} # converting image to tensor and storing 

#print(image_datasets['train'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, # creating dataloader
                                             shuffle=False, num_workers=0)
              for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(dataloaders['train'])


def imshow(inp, title=None):
   
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    # plt.show()
    if title is not None:
        plt.title(title)
        plt.savefig(title)
    # plt.pause(0.1)  # pause a bit so that plots are updated
#print(dataset_sizes)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
    
num_epochs = 5 
batch_size = 128
num_samples = len(dataloaders["train"])

model = Autoencoder().cpu()
distance = nn.MSELoss()
total_distance = 0
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloaders['train']:
        img,_ = data
        img = Variable(img).cpu()
        output = model(img)
        loss = distance(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_distance += loss

    

print(total_distance)

