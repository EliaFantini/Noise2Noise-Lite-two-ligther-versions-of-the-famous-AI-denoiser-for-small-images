# Deep Learning Mini Project

## General Information

### Creator: Elia Fantini, Kieran Vaudaux, Kaan Okumuş

### Environment:

This project has been developed and tested with python 3.8. The required libraries are:
    
- PyTorch: for the implementation of deep learning models, methods.
           If you don't have it, you can download it by following the instructions [here](https://pytorch.org/).
***
## Project Description & Guidance for Reproducing the Project:

The goal of the mini-projects is to implement a Noise2Noise model. A Noise2Noise model is an image denoising network trained without a clean reference image. The original paper can be found at [here](https://arxiv.org/abs/1803.04189).

The project has two parts, focusing on two different facets of deep learning. The first one is to build a network that denoises using the PyTorch framework, in particular the torch.nn modules and autograd. The second one is to understand and build a framework, its constituent modules, that are the standard building blocks of deep networks without PyTorch’s autograd.

### Dataset:
- Dataset is available [here](https://drive.google.com/drive/u/2/folders/1CYsJ5gJkZWZAXJ1oQgUpGX7q5PxYEuNs).
- Please put `train_data.pkl` and `val_data.pkl` inside both `Miniproject_1\others\dataset` and `Miniproject_2\others\dataset` folders. 

### Project Structure:

```bash
├── Miniproject_1
│    ├── __init__.py
│    ├── model.py
│    ├── bestmodel.pth
│    ├── Report_1.pdf
│    ├── results.pkl
│    ├── Experiments.ipynb
│    └── others
│         ├── Config.py
│         ├── dataset
│         │    ├── train_data.pkl
│         │    └── val_data.pkl
│         ├── dataset.py
│         └── nets
│              ├── DeepLabV3.py
│              ├── unet.py
│              ├── unet2.py
│              └── unet3.py
└── Miniproject_2
     ├── __init__.py
     ├── model.py
     ├── bestmodel.pth
     ├── Report_2.pdf
     ├── results.pkl
     ├── Experiments.ipynb
     └── others
          ├── Config.py
          ├── dataset
          │    ├── train_data.pkl
          │    └── val_data.pkl
          ├── dataset.py
          ├── helpers_functional.py
          ├── helpers_layer.py
          ├── dataset
          ├── nets
          │    └── unet.py
          └── testing_custom_blocks
               ├── testing_conv2d.py
               └── testing_convtranspose2d.py

```


***
### Sample Run of the Model

Put your test.py in the base directory and Run `python3 test.py -p "./PROJ_336006_SCIPER2_287703" -d "./PROJ_336006_SCIPER2_287703/Miniproject_1/others/dataset/"
` in your terminal. This command does all your testings. 

You can also test Conv2D function by comparing with PyTorch one. To do so, Run
`python3 test.py -p "./PROJ_336006_SCIPER2_287703" -d "./PROJ_SCIPER1_SCIPER2_287703/Miniproject_1/others/dataset/"
`

***
### Report

Reports for Mini Project 1 and 2 can be found in `Miniproject_1\Report_1.pdf` and `Miniproject_2\Report_2.pdf` respectively.
