import sys
import torch
from torchvision import transforms, datasets

def load_data(args):
    
    """
    Function to obtain the dataloaders and the class_to_index mapping.
    Inputs:
        args : User input from command line.
    Outputs:
        trainloader, validloader, testloader and train_dataset.class_to_index
    """
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    
    # Now we can define the transforms to perform in training, validation and testing
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valtest_transforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    # Obtain the datasets with ImageFolder
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
        valid_dataset = datasets.ImageFolder(valid_dir, transform=valtest_transforms)
        test_dataset = datasets.ImageFolder(test_dir, transform=valtest_transforms)
    except FileNotFoundError:
        print('The datadir introduced does not fit the expected model (datadir/train, datadir/valid and datadir/test)')
        print('Please introduce another directory')
        sys.exit("Wrong filder. Program terminating")
        
    # Obtain the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch)
    validloader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=args.batch)
    testloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch)
    
    return trainloader, validloader, testloader, train_dataset.class_to_idx