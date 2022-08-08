<p align="center">
  <img alt="🏁Noise2Noise_Lite" src="https://user-images.githubusercontent.com/62103572/183460246-3c3e57d0-6502-4396-a168-b7e5875d33a8.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/Noise2Noise-Lite-two-ligther-versions-of-the-famous-AI-denoiser-for-small-images">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/Noise2Noise-Lite-two-ligther-versions-of-the-famous-AI-denoiser-for-small-images">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/Noise2Noise-Lite-two-ligther-versions-of-the-famous-AI-denoiser-for-small-images">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/Noise2Noise-Lite-two-ligther-versions-of-the-famous-AI-denoiser-for-small-images">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/Noise2Noise-Lite-two-ligther-versions-of-the-famous-AI-denoiser-for-small-images?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/Noise2Noise-Lite-two-ligther-versions-of-the-famous-AI-denoiser-for-small-images?label=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/Noise2Noise-Lite-two-ligther-versions-of-the-famous-AI-denoiser-for-small-images?style=social">
</p>


This project is a revisiting of the famous denoiser Noise2Noise (click [here](https://arxiv.org/abs/1803.04189) for the original paper), an image denoising network trained without a clean reference image: it doesn't need any clean image but can be trained only using noisy data. 

The revisiting was made in two versions, focusing on two different facets of deep learning: 
- the first one relies on the U-Net architecture of the original paper with some slight changes to make the model ligther and less flexible (to avoid overfitting), since the images we had to apply our model on were much smaller than the ones used on the original model. Every component makes use of the PyTorch framework, in particular pf the torch.nn modules and autograd. 
- the second one instead implements an even simpler model but every single component of the neural network is coded from scratch, Pytorch functions use is reduced to its barebones. The main focus of this version is to understand and build a framework with all its constituent modules, that are the standard building blocks of deep networks, without PyTorch’s autograd (which we reimplemented from scratch).
The following image is an example of the performance of the original Noise2Noise architecture:

<p align="center">
<img width="517" alt="Immagine 2022-08-08 182412" src="https://user-images.githubusercontent.com/62103572/183466131-805b2ae2-1d27-4592-baf7-595edc62c304.png">
</p>

## Authors

- [Elia Fantini](https://github.com/EliaFantini/)
- [Kieran Vaudaux](https://github.com/KieranVaudaux)
- [Kaan Okumuş](https://github.com/okumuskaan)

### How to install:

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


## 🛠 Skills

Matplotlib, Pandas, Pytorch, Nltk, Seaborn, Sklearn. Big dataset manipulation with Pandas, Word frequency analysis, transfer-learning, unsupervised clustering with Bertopic, preprocessing of the quotes,
sentiment analysis with transformer models (BERT) and VADER-Sentiment, Textstat library for grammatical structure and complexity analysis, build a classifier using a CNN.

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/EliaFantini/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
