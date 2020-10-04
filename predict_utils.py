import argparse
from train_utils import positiveint
from PIL import Image
import sys
import numpy as np
import torch

def parse_inputs():
    
    """ Parses the input arguments
    
    Input: Command line inputs specified by the user.
    Output: Parsed command line inputs
    
    """
    parser = argparse.ArgumentParser(description = 'This will make a prediction on an image')
    parser.add_argument('image', type = str, help = 'path-to-image')
    parser.add_argument('checkpoint', type = str, help ='checkpoint of the model')
    parser.add_argument('--top_k',type = positiveint, default = 1, help = 'N most likely predictions')
    parser.add_argument('--category_names', type = str, help = 'Category names to map the categories')
    parser.add_argument('--gpu', action="store_true", default=False)
    
    return parser.parse_args()

def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Tensor of the image
        
        Input:
        image: Image path
        Output: Tensor of the image
    '''
    try:
        pil_image = Image.open(image)
    except FileNotFoundError:
        print('The image was not found!')
        print('Please introduce another image')
        sys.exit("Program terminating")
        
    pil_image = pil_image.resize((256,256))
    left = int(pil_image.size[0]/2-224/2)
    upper = int(pil_image.size[1]/2-224/2)
    right = left +224
    lower = upper + 224
    im_cropped = pil_image.crop((left, upper,right,lower))
    np_image = np.array(im_cropped)/255
    # Normalizing the chanels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    np_image = torch.from_numpy(np_image)
    return np_image