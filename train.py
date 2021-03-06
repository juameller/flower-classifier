from train_utils import parsing_inputs
from train_model import load_data, build_model, train, validation_test, save_check
from torch import nn
import torch
"""
Command line application to train a pretrained deep neural networks to predict flower types.

Basic usage: 
    python train.py path/to/imagefolder

Options:
    Save a checkpoint:
    python train.py path/to/imagefolder --save_dir checkpointdir
    
    Select the architecture of the classifier:
    python train.py path/to/imagefolder --arch vgg16
    
    Set different hyperparameters:
    - Specify the learning rate: python train.py path/to/imagefolder --learning_rate 0.001
    - Specify the number of epochs: python train.py path/to/imagefolder --epochs 5
    - Specify the batch size: python train.py path/to/imagefolder --batch_size 32
    - Specify how often the training info is printed: python train.py path/to/imagefolder --printed_every 20
    - Specify the number of hidden units: python train.py path/to/imagefolder --hidden_units 1024 512
    - Specify the dropout: python train.py path/to/imagefolder --dropout 0.1
    - Compute in GPU: python train.py path/to/imagefolder --gpu
"""

def main():
    # Parse input data 
    args = parsing_inputs()

    # Obtain the dataloaders and a dictionary class_to_idx we will use during prediction
    trainloader, validloader, testloader, class_to_idx = load_data(args)
    

    # Now we download the model based on the input and select the device we will train it on
    possible_inputs = {'vgg16':25088, 'alexnet': 9216}
    model, device = build_model(possible_inputs, args)

    # The next step is to define the criterion and train the model
    criterion = nn.NLLLoss()
    train(model, device, args, trainloader, validloader, criterion)

    # We then perform a validation test on new unseen data
    with torch.no_grad():
        validation_test(model, testloader, device, criterion)

    # Finally we save the checkpoint
    save_check(args, model, class_to_idx, possible_inputs)

if __name__ == '__main__':
    main()