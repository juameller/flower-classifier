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

def predict(image, model, topk, device):
    
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    Inputs:
        image: TensorFlow array.
        model: Trained deep learning model
        topk: Number of most likely classes the function will return.
        device: CPU or GPU
        
    Outputs:
        top_classes: List with the topk most likely categories.
        top_p: List with the probabilities of the topk most likely categories.
    
    '''
    
    img = image.unsqueeze_(0)
    # Now we have the tensorFlow array out model is prepared to do
    img = img.float()
    img = img.to(device)
    logout = model.forward(img)
    out = torch.exp(logout)
    top_p, top_classes = out.topk(topk,dim=1)
    # We invert the dictionary
    inv_map = {val: key for key, val in model.class_to_idx.items()}
    
    # We want to return the classes and probs like lists
    if topk > 1:
        top_classes = [inv_map[item.item()] for item in top_classes.squeeze()]
        top_p = [item.item() for item in top_p.squeeze()]
    else:
        top_classes = [inv_map[top_classes.item()]]
        top_p = [top_p.item()]
    return top_p, top_classes
