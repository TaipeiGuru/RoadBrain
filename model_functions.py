import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import processing_functions

from collections import OrderedDict

# Function for saving the model checkpoint
def save_checkpoint(model, training_dataset, arch, epochs, lr, hidden_units, input_size):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'input_size': (3, 224, 224),
                  'output_size': 102,
                  'hidden_layer_units': hidden_units,
                  'batch_size': 64,
                  'learning_rate': lr,
                  'model_name': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clf_input': input_size}

    torch.save(checkpoint, 'checkpoint.pth')
    
# Function for loading the model checkpoint    
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    if checkpoint['model_name'] == 'vgg':
        model = models.vgg16(pretrained=True)
        
    elif checkpoint['model_name'] == 'alexnet':  
        model = models.alexnet(pretrained=True)
    else:
        print("Model architecture not recognized.")
        
    for param in model.parameters():
            param.requires_grad = False    
    
    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(checkpoint['clf_input'], checkpoint['hidden_layer_units'])),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(checkpoint['hidden_layer_units'], 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model