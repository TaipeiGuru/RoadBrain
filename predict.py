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

parser = argparse.ArgumentParser(description='Image Classifier Predictions')

parser.add_argument('--image_dir', type = str, default = 'flowers/test/15/image_06351.jpg', help = 'Path to image')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--json', type = str, default = 'flower_to_name.json', help = 'class_to_name json file')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')

arguments = parser.parse_args()

class_to_name_dict = processing_functions.load_json(arguments.json)
model = model_functions.load_checkpoint(arguments.checkpoint)
checkpoint = torch.load(arguments.checkpoint)
image = processing_functions.process_image(arguments.image_dir)
processing_functions.imshow(image)

probabilities, classes = model_functions.predict(arguments.image_dir, model, arguments.topk, arguments.gpu)  

print(probabilities)
print(classes)

processing_functions.display_image(arguments.image_dir, class_to_name_dict, classes, probabilities)