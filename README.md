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
##


### How to install:
Download this repository as a zip file and extract it into a folder.

This project has been developed and tested with python 3.8. The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official website (select python of version 3): https://www.anaconda.com/download/

The only additional library required is PyTorch for the implementation of deep learning models and methods. If you don't have it, you can download it by following the instructions [here](https://pytorch.org/).

Then, download the dataset available [here](https://drive.google.com/drive/u/2/folders/1CYsJ5gJkZWZAXJ1oQgUpGX7q5PxYEuNs) (unfortunately the drive is protected and it's for EPFL strudents only, the files were too big to be uploaded on GitHub), and put `train_data.pkl` and `val_data.pkl` inside both `Noise2Noise Lite\others\dataset` and `Noise2Noise from scratch\others\dataset` folders. 

### How to use

Put your test.py in the base directory and run in your terminal (or on Anaconda Prompt) the following command that does all the testings: 
```bash
python test.py -p "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*" -d "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*/Noise2Noise Lite/others/dataset/"
```

You can also test Conv2D function by comparing it with the PyTorch one. To do so, run:
```bash
python test.py -p "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*" -d "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*/Noise2Noise Lite/others/dataset/"
```

If you
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






## 🛠 Skills

Python, Pytorch. Deep learning knowledge, good knowledge of all the components that constitute a neural network and its training. Deep knowledge of the Pytorch framework to rebuild from scratch all its basics components and its core mechanism, autograd.

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/EliaFantini/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
