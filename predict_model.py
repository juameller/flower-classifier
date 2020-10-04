import torch
import sys
from train_model import MyClassifier
from torchvision import models

def loadcheckpoint(args):
    """
    Loads a saved checkpoint of the model, builds it and returns it
    
    Input: args specified by the user
    Output: model
    """
    archs = ['vgg16','alexnet']
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    except FileNotFoundError:
        print("The checkpoint introduced by the user has not been found.")
        print("Please introduce a valid checkpoint")
        sys.exit("Program terminating")
   
    if checkpoint['arch'] not in archs:
        print("The selected architecture is not contemplated in this application")
        sys.exit("Program terminating")
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)      
    classifier = MyClassifier(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'],checkpoint['dropout'])
    classifier.load_state_dict(checkpoint['state_dict'])
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    # Now we deactivate the gradiente computation    
    for param in model.parameters():
        param.requieres_grad = False
    return model