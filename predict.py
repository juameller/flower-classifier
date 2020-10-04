from predict_utils import parse_inputs, process_image
from predict_model import loadcheckpoint
import torch
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
    print(args)

    # We get the image as a FloatTensor
    img = process_image(args.image)
    print(img.shape)

    # We now have to load the checkpoint and build the model
    model = loadcheckpoint(args)
    print(model)

    # We will performe the calculation in the select device
    device = "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    print(f'These computations are performed in {device}')
    model.to(device)
    
        
if __name__ == '__main__':
    main()