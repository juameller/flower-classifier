import argparse

def positiveint(value):
    """
    This function will be used with parser.add_argument()
    to ensure that some int values are > 0
    """
    ivalue = int(value)
    if ivalue<=0:
        raise argparse.ArgumentTypeError("{} is an invalid positive int value".format(value))
    return ivalue


def positivefloat(value):
    """
    This function will be used with parser.add_argument()
    to ensure that some float values are > 0
    """
    fvalue = float(value)
    if fvalue<=0:
        raise argparse.ArgumentTypeError("{} is an invalid positive float value".format(value))
    return fvalue

def parsing_inputs():
    
    """ Parses the input arguments
    
    Input: Command line inputs specified by the user.
    Output: Parsed command line inputs
    
    """
    parser = argparse.ArgumentParser(description = 'This script will train a depp neural network from torchvision.models to identify flowers')
    parser.add_argument('data_dir', type = str, help = 'Data directory')
    parser.add_argument('--save_dir',type = str, default = 'checkpoints/default_chekpoint.pth', help = 'Checkpoint')
    parser.add_argument('--arch',type = str, default = 'vgg16', help = 'Architecture', choices=['vgg16','alexnet'])
    parser.add_argument('--learning_rate',type = positivefloat, default = 0.003, help = 'Learning rate')
    parser.add_argument('--epochs',type = positiveint, default = 2, help = 'Number of Epochs')
    parser.add_argument('--batch',type = positiveint, default = 32, help = 'Batch Size')
    parser.add_argument('--print_every', type= positiveint, default = 20, help = 'It selects how often the training progress is shown')
    parser.add_argument('--hidden_units', nargs ='+', type = positiveint, help = 'Hidden Layers (the input and output layers are fixed by the problem and arch)', default = [1024,512])
    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--dropout',type = float, default = 0.2, help = 'Dropout Probability')
    
    return parser.parse_args()