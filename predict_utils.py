import argparse
from train_utils import positiveint

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