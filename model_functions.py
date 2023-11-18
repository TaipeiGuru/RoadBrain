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

def save_checkpoint(model, training_dataset, arch, epochs, lr, hidden_units, input_size):
    model.class_to_idx = training_dataset.class_to_idx
    checkpoint = {'input_size': (3, 224, 224),
                  'output_size': 11,
                  'hidden_layer_units': hidden_units,
                  'batch_size': 64,
                  'learning_rate': lr,
                  'model_name': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clf_input': input_size}
    torch.save(checkpoint, 'checkpoint.pth')

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
                                            ('fc2', nn.Linear(checkpoint['hidden_layer_units'], 11)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def validation(model, validateloader, criterion, gpu):
    val_loss = 0
    accuracy = 0
    for images, labels in iter(validateloader):
        images, labels = images.to(gpu), labels.to(gpu)
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        probabilities = torch.exp(output)
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return val_loss, accuracy

def test_accuracy(model, test_loader, gpu):
    model.eval()
    model.to(gpu)
    with torch.no_grad():
        accuracy = 0
        for images, labels in iter(test_loader):
            images, labels = images.to(gpu), labels.to(gpu)
            output = model.forward(images)
            probabilities = torch.exp(output)
            equality = (labels.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))    
        
def train_classifier(model, optimizer, criterion, arg_epochs, train_loader, validate_loader, gpu):
    epochs = arg_epochs
    steps = 0
    print_every = 40
    model.to(gpu)
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in iter(train_loader):
            steps += 1
            images, labels = images.to(gpu), labels.to(gpu)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validate_loader, criterion, gpu)
        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
        
                running_loss = 0
                model.train()      

def predict(image_path, model, topk=5, gpu='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(gpu)
    
    image = processing_functions.process_image(image_path)
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    probabilities = torch.exp(output)
    top_probabilities, top_indices = probabilities.topk(topk)
    
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
        
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes                    