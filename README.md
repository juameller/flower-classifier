# Flower Classifier

This application can be used to:
 - Train an image classifier to recognize species of flowers with ``train.py``.
 - Classify a given image with ``predict.py``.
 
A dataset of 102 species of flowers can be downloaded from, **you need to download the data from https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz**

All the requirements can be found in requirements.txt.

## ``train.py``
Command line application to train a pretrained deep neural networks to predict flower types.
This application uses pretrained models from ``Torchvision`` whose classifier has been adapted to recognize 102 types of flowers.

Basic usage: 
- ``python train.py path/to/imagefolder``

Options:
- Save a checkpoint for the trained network:
-- ``python train.py path/to/imagefolder --save_dir checkpointdir``
- Select the architecture of the classifier:
-- ``python train.py path/to/imagefolder --arch vgg16``
    
- Set different hyperparameters:
-- Specify the learning rate: ``python train.py path/to/imagefolder --learning_rate 0.001``
--Specify the number of epochs: ``python train.py path/to/imagefolder --epochs 5``
--Specify the batch size: ``python train.py path/to/imagefolder --batch_size 32``
--Specify how often the training info is printed: ``python train.py path/to/imagefolder --printed_every 20``
-- Specify the number of hidden units: ``python train.py path/to/imagefolder --hidden_units 1024 512``
-- Specify the dropout: ``python train.py path/to/imagefolder --dropout 0.1``
- Compute in GPU: ``python train.py path/to/imagefolder --gpu``

## ``predict.py``
Command line application to predict flower name from image. It uses the checkpoint previously saved during the training process.

Basic usage: 
- ``python predict.py path/to/image path/to/checkpoint``

Options:
- To return K most likely cases:
-- ``python predict.py path/to/image path/to/checkpoint --top_k 3``
    
- To use a mapping of categories to real names:
-- ``python predict.py path/to/image path/to/checkpoint --category_names cat_to_name.json``
    
- To use GPU for inference:
-- ``python predict.py path/to/image path/to/checkpoint --gpu``



