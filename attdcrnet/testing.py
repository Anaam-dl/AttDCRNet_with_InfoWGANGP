# Evaluating test data

#%%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
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
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import*
from attdcr_model import *

#%% ---------- Inputing parameters and directories ------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True, help='Path of the test set') 
    parser.add_argument('--model_path', type=str, required=True, help='Path of the trained model weights')
    parser.add_argument('--imageSize', type=int, default=136, help='Input size of Att-DCRNet')
    parser.add_argument('--cuda'  , action='store_false', help='enables cuda')
    #opt = parser.parse_args()
    return parser.parse_args()

#%% Testing function
def test(test_loader, net, device):
    net = net.to(device)
    net.eval()
    correct = 0
    n_samples = 0
    pred_labels_total = np.array([])
    true_labels_total = np.array([])
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            preds = net(images)
            _, pred_labels = torch.max(preds, 1)
            if pred_labels_total.size == 0:
                pred_labels_total = pred_labels.cpu().numpy()
                true_labels_total = labels.cpu().numpy()
            else:
                pred_labels_total = np.concatenate((pred_labels_total, pred_labels.cpu().numpy()))
                true_labels_total = np.concatenate((true_labels_total, labels.cpu().numpy()))
                
            correct += (pred_labels == labels).sum().item()
            n_samples += images.shape[0]
        Accuracy = correct/n_samples
        #print()
        #print('-'* 10)
        #print('Correct: {}, Total: {}'.format(correct, n_samples))
        return Accuracy, pred_labels_total, true_labels_total


def main(opt):
    
    test_path = opt.test_path
    model_path = opt.model_path
    imageSize = opt.imageSize
        
    #%% Define device
    if opt.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
        
    #%% loading data
    mean = [0.5] 
    std = [0.5] 
    test_set = datasets.ImageFolder(root=test_path,
                                    transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((imageSize, imageSize)),
                                        #transforms.CenterCrop(imageSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                    ]))
    
    assert test_set
    test_loader = DataLoader(test_set, batch_size= 128)
    
    def classes_num(Subdataset):
        clToIdx = Subdataset.class_to_idx
        count = {}
        for key in clToIdx: count[key] = 0
        for i in Subdataset.targets:
            for key, value in clToIdx.items():
                if i == clToIdx[key]: count[key] +=1
                    
        print('Samples in each Testset class:')
        for key, value in count.items():
            print('{0}: {1}'.format(key, value))
    
    classes_num(test_set)
    print('\n')
    
    #%% Define the network 
    attdcrnet = AttDCRNET(2, 1).to(device)
    if not model_path == '':
        attdcrnet.load_state_dict(torch.load(model_path))
    else:
        raise Exception('Please specify path to the pre-trained model!')
    attdcrnet.eval()
    
    
    #%% Testing
    
    accuracy, preds_total, true_total = test(test_loader, attdcrnet, device)
    print()
    cm = confusion_matrix(true_total, preds_total)
    print('Confusion Matrix:')
    print(cm)
    
    total_class_num = np.sum(cm, axis=1)
    class_cottect = np.diag(cm)
    MCA = 0
    for i in range(len(test_set.classes)):
        ccr = class_cottect[i]/total_class_num[i]
        #print('CCR_{0} = {1}'.format(i, ccr))
        MCA +=ccr
        
    MCA = MCA/len(test_set.classes)
    print()
    print('Testset Overall Acc: {:.6} % '.format(accuracy*100))
    print('Testset BcA Acc: {:.6} % '.format(MCA*100))
    
    f1 = f1_score(true_total, preds_total, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    print('F1 = {:.6}'.format(f1))
    mcc = matthews_corrcoef(true_total, preds_total, sample_weight=None)
    print('MCC = {:.6}'.format(mcc))


if __name__ == "__main__":
    main(parse_args())