from predict_utils import parse_inputs, process_image
from predict_model import loadcheckpoint, predict
import torch
import json
"""
Command line application to predict flower name from image.

Basic usage: 
    python predict.py path/to/image path/to/checkpoint

Options:
    To return K most likely cases:
    python predict.py path/to/image path/to/checkpoint --top_k 3
    
    To use a mapping of categories to real names:
    python predict.py path/to/image path/to/checkpoint --category_names cat_to_name.json
    
    To use GPU for inference:
    python predict.py path/to/image path/to/checkpoint --gpu


"""

def main():
    # Parse input arguments
    args = parse_inputs()
    

    # We get the image as a FloatTensor
    img = process_image(args.image)
  

    # We now have to load the checkpoint and build the model
    model = loadcheckpoint(args)
    

    # We will performe the calculation in the select device
    device = "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    print(f'These computations are performed in {device}')
    model.to(device)

    # We set the model to evaluation mode (so that we are not using dropout)
    model.eval()
    with torch.no_grad():
        probs,classes = predict(img, model, args.top_k, device)

    # Now, if we indicated a category_name:
    if args.category_names:
        try:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
        except FileNotFoundError:
            print("The category names file has not been found.")
            print("Please introduce a valid file.")
            sys.exit("Program terminating.")
        classes = [cat_to_name[item] for item in classes]
 

    # Printing out the results
    print(f'The {args.top_k} most likely classes of flowers are:')
    
    for key, value in zip(classes, probs):
        print(f'Flower: {key};    '
              f'Probability: {value}')
    
        
if __name__ == '__main__':
    main()