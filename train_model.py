import sys
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, datasets, models

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


class MyClassifier(nn.Module):
    """
    Fully conneceted classifier we will train to predict flower names from images
    Inputs:
        input_size: Depending on the model
        output_size: Depending on the problem (102 classes in this case)
        hidden_layers: The user can choose the number of ReLU hidden layers.
        dropout_p: Probability of dropout.
    """
    def __init__(self, input_size, output_size, hidden_layers,dropout_p):
        super().__init__()
        # The first layer 
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # We pair the rest of the layers and define them
        paired_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        for p1,p2 in paired_layers:
            self.hidden_layers.append(nn.Linear(p1,p2))
        # Define the output layer
        self.output_layer = nn.Linear(hidden_layers[-1],output_size)

        # Define that we will be using dropout - We will also check that it is between 0 and 1
        try:
            self.dropout = nn.Dropout(p=dropout_p)
        except ValueError:
            print("The dropout probability has to be between 0 and 1 amd got ",dropout_p)
            print("Please introduce a valid p")
            sys.exit("Program terminating.")


    # Then we define the forward method
    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        x = F.log_softmax(x,dim=1)
        return x
    
def build_model(possible_inputs, args):
    
    """
    Inputs:
        possible_models: A dictionary with the models that this application uses.
        args: User input from command line
    Outputs:
        model: Pretrained model with the classifier defined by the user
        device: It selects whether the model is trained in CPU/GPU
    
    """
    #if args.arch not in possible_inputs:
    #    print("The architecture you have selected is not recognized.")
    #    print("Please select an architecrure from: ",list(possible_inputs.keys()))
    #    sys.exit("Wrong architecture. Program terminating")
    #else:
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained = True)        
    # From here we can indicate not to compute the gradient
    for param in model.parameters():
        param.requieres_grad = False
    # Now we attach our classifier
    input_size = possible_inputs[args.arch]
    output_size = 102
    classifier = MyClassifier(input_size, output_size, args.hidden_units, args.dropout)
    model.classifier = classifier
    
    device = "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    model = model.to(device)
    
    return model, device


def train(model, device, args, trainloader, validloader, criterion):
    
    """
    Training loop for the model
    
    Inputs:
        model: Pretrained model
        device: To select wether we will train on CPU/GPU
        args: Command line input from user (contains number of epochs, lr, etc.)
        trainloader: Data loader for the training subset.
        validloader: Data loader for the validation subset
        criterion: To specify how the loss is computed
    Output:
        model: Trained model
    """
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    epochs = args.epochs
    steps = 0
    print_every = args.print_every
    running_loss = 0        
    for e in range(epochs):
        for images,labels in trainloader:
            # We are going to count the number of steps performed
            steps += 1
            # We move the inputs and labels to GPU
            images, labels = images.to(device), labels.to(device)
            # After we zero_grad everything
            optimizer.zero_grad()
            # Then we perform the feedforward
            logps = model.forward(images)
            # Compute the error
            loss = criterion(logps, labels)
            # We back propagate it
            loss.backward()
            # We perform one move
            optimizer.step()
            # We sum up the loss
            running_loss += loss.item()
            

            # Now we can check if we do the validation test
            if steps % print_every == 0:
                # We can set the model to evaluation mode
                with torch.no_grad():
                    model.eval()
                    validation_loss = 0
                    accuracy = 0
                    for images, labels in validloader:
                        images,labels = images.to(device), labels.to(device)
                        # Forward feed
                        logps = model.forward(images)
                        # Compute the error
                        validation_loss += criterion(logps, labels).item()
                        # Obtain the probabilities
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equal = top_class == labels.view(*top_class.shape)
                        # We have to convert the byte tensor to FloatTensor in order to do the mean
                        accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
                    # Now we can print the results:
                print(f'Epoch: {e+1}/{epochs} ... '
                      f'Training Error: {running_loss/print_every:.3f} ...'
                      f'Validation Loss {validation_loss/len(validloader):.3f} ...' 
                      f'Accuracy: {accuracy/len(validloader)*100:.3f}%')
                running_loss = 0
                # Set the model to training mode
                model.train()
  
