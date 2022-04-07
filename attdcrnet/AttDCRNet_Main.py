# Pythorch implementation of the Att-DCRNet 

#%%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json
import helper
from itertools import product
from collections import namedtuple
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

from attdcr_model import *

#%% ---------- Inputing parameters and directories ------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='Path of training set')
    parser.add_argument('--val_path', type=str, default=r'', help='Path of validation set')
    parser.add_argument('--experiment_root', type=str, default='./output/', help='Path of the experiment output folder')
    parser.add_argument('--models_path', action='store_true', help='Path of the trained model weights')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--end_epoch', type=int, default=2, help='end epoch')
    parser.add_argument('--imageSize', type=int, default=136, help='Input size of Att-DCRNet')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=int, default=0.01, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--cuda'  , action='store_false', help='enables cuda')
    #opt = parser.parse_args()
    return parser.parse_args()

def main(opt):
    
    train_path = opt.train_path
    val_path = opt.val_path
    experiment_root = opt.experiment_root
    models_path = opt.models_path
    start_epoch = opt.start_epoch
    end_epoch = opt.end_epoch
    imageSize = opt.imageSize
    batchsize = opt.batchsize
    lr = opt.lr
    lr_decay = opt.lr_decay

    #%% Making subfolders & defining device
    
    if 1- os.path.exists(experiment_root):
        #experiment_root = r'./output/'
        os.makedirs(experiment_root, exist_ok = True)
    
    print('Experiment output folder name:', experiment_root)
    
    if opt.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    
    #%% Loading data
    
    print('loading data..')
    
    train_set = datasets.ImageFolder(root=train_path,
                                    transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize(imageSize),
                                        transforms.CenterCrop(imageSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), 
                                    ]))
    
    assert train_set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize,
                                            shuffle=True, num_workers=0, pin_memory=False)    
    
    if not val_path == '':
        val_set = datasets.ImageFolder(root=val_path,
                                        transform=transforms.Compose([
                                            transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize(imageSize),
                                            transforms.CenterCrop(imageSize),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,)),
                                        ]))
        
        assert val_set
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batchsize,
                                            shuffle=True, num_workers=0, pin_memory=False) 
    
    # Displaying the size of each class of the training Set
    def classes_num(Subdataset):
        clToIdx = Subdataset.class_to_idx
        count = {}
        for key in clToIdx: count[key] = 0
        for i in Subdataset.targets:
            for key, value in clToIdx.items():
                if i == clToIdx[key]: count[key] +=1
                    
        print('Samples in each class of training Set:')
        for key, value in count.items():
            print('{0}: {1}'.format(key, value))
        
    classes_num(train_set)
    
    #%% Displaying some images!
    
    # =============================================================================
    # # functions to show an image
    # def imshow(img):
    #     img = img / 2. + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # 
    # # get some random training images
    # dataiter = iter(train_loader)
    # images_, _ = dataiter.next()
    # 
    # # show images
    # imshow(torchvision.utils.make_grid(images_))
    # =============================================================================
    
    #%% Define the Att-DCRNet 
    
    attdcrnet = AttDCRNET(2, 1).to(device)
    if models_path:
        attdcrnet.load_state_dict(torch.load(models_path))
        
    
    #%% Define loss and optimizer
    
    optimizer = optim.SGD(attdcrnet.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 15, gamma=lr_decay)# 15
    loss_function = nn.CrossEntropyLoss()
    
    #%% Train the model
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(start_epoch, end_epoch):
        # Train on training set
        attdcrnet.train()
        epoch_loss = 0
        epoch_num_correct = 0
        epoch_val_loss = 0
        epoch_val_num_correct = 0
        for batch in train_loader: # Get batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            preds = attdcrnet(images)# passforward batch
            loss = loss_function(preds, labels) # calculate loss
            
            optimizer.zero_grad() #zero Gradients
            loss.backward() #calculate the gradients
            optimizer.step() # Update the weights
        
            epoch_loss += loss.item()*train_loader.batch_size
            epoch_num_correct += preds.argmax(dim=1).eq(labels).sum().item()
            
        # Eval on validation set
        if not val_path == '':
            attdcrnet.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images_,labels_ = batch
                    images_,labels_ = images_.to(device), labels_.to(device)
                    
                    preds_ = attdcrnet(images_)# passforward batch
                    loss_ = loss_function(preds_, labels_) # calculate loss
                    
                    epoch_val_loss += loss_.item()*val_loader.batch_size
                    epoch_val_num_correct += preds_.argmax(dim=1).eq(labels_).sum().item()
        
        # Learning rate decay step
        exp_lr_scheduler.step()  
        
        # compute trainign metrics
        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_acc = epoch_num_correct / len(train_loader.dataset)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc) 
        
        if not val_path == '':
            epoch_val_loss = epoch_val_loss / len(val_loader.dataset)
            epoch_val_acc = epoch_val_num_correct / len(val_loader.dataset)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc) 
    
        if val_path == '':
            print(f'epoch:{epoch} ==> train loss:{epoch_loss:.4f} - train acc:{epoch_acc:.4f}')
        else:
            print(f'epoch:{epoch} ==> train loss:{epoch_loss:.4f} - train acc:{epoch_acc:.4f} - val loss:{epoch_val_loss:.4f} - val acc:{epoch_val_acc:.4f}')
        
        # save chechpoints 
        torch.save(attdcrnet.state_dict(), experiment_root + '/' + 'AttDCRnet_' + str(epoch) + '.pth')
    
    # save training metrics
    results  = OrderedDict()
    results["epoch"] = [i for i in range(start_epoch, end_epoch)]
    results["Train loss"] = train_loss #epoch_loss
    results["Train accuracy"] = train_acc
    if not val_path == '':
        results["Valid loss"] = val_loss
        results["Valid accuracy"] = val_acc    
    pd.DataFrame.from_dict(results, orient='columns').to_csv(experiment_root + '/' + 'results.csv')
    
    
if __name__ == "__main__":
    main(parse_args())