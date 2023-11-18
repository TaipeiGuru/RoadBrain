import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import model_functions
import processing_functions
import json
import argparse

parser = argparse.ArgumentParser(description='Train Image Classifier')

parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 10000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 20, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
arguments = parser.parse_args()

data_directory = 'flowers'
train_directory = 'flowers/train'
valid_directory = 'Flowers/valid'
test_directory = 'flowers/test'

transforms_tuple = processing_functions.data_transforms()
training_transforms = transforms_tuple[0]
validation_transforms = transforms_tuple[1]
testing_transforms = transforms_tuple[2]
dataset_tuple = processing_functions.load_datasets(train_directory, training_transforms, valid_directory, validation_transforms, test_directory, testing_transforms)
training_dataset = dataset_tuple[0]
validation_dataset = dataset_tuple[1]
testing_dataset = dataset_tuple[2]

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)