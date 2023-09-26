# Glacier Movement Prediction with Attention-based Recurrent Neural Networks and Satellite Data
![Alt Text](animationParvatialigned.gif)

This repository is desigend to facilitate the process of extracting satellite images from Microsoft's Planetary Computer database and gives all functionalities to preprocess (alignment, patching) the data with enhanced correlation coefficient maximization. Furthermore the repo entails a newly developed methodology (paper_reference) to divide the images into sequential patches in order to train recurrent neural networks

In order to run the code it is advised to use a virtual environment with the packages in the requirements.txt installed

## Image extraction
The code uses Landsat-8 images, but can be adapted to other satellite databases in the Planetary Computer. The main file used to control the code output is the config.py file. Here all hyperparameters for image extraction and alignment can be specified. Additionally the jupyter notebook getCoordinatesFromMap.ipynb can be used to get a interactive map with the ipyleaflet tool in order to extract bounding boxes for a specific region of interest in lattitude and longitude coordinates. The dataAPI.py file extracts the necessary files and saves them in a pickle object for each year. All data is saved in specified folders in the created /datasets folder. The alignment.py file is then used in order to align the data and extract the needed bands from the images. The extractNDSI variable in the config.py file can be used if NDSI maps should be extracted for glacier/ice/snow data. 

## Patching for deep learning applications
The createPatches.py file is used to extract tensors containing consecutive patches from the same coordinates across time from the images. The used methodology is explained in detail in <paper link>. The sript creates two folders, /images and /targets, where the respective images file is the model input (4 consecutive patches) and the targets file corresponds to the ground truths (connecting 4 consecutive patches). 

## Train RNN models
The /DeepLearning folder contains functionalities to train different deep learning models. Each folder for a different model class contains a train script, which can be used to control hyperparameters and the general training process. The scripts use the weights and biases tool, which can be used for free in order to visualize tarining progress in a web browser. The tool only requires the creation of an account. 
