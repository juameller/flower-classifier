from predict_utils import parse_inputs
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
    
        
if __name__ == '__main__':
    main()