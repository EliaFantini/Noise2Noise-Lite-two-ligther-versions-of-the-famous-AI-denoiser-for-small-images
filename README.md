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
- the first one relies on the U-Net architecture of the original paper with some slight changes to make the model ligther and less flexible (to avoid overfitting), since the images we had to apply our model on were much smaller than the ones used on the original model. Every component makes use of the PyTorch framework, in particular the torch.nn modules and autograd. 
- the second one instead implements an even simpler model but every single component of the neural network is coded from scratch, Pytorch functions use is reduced to its barebones. The main focus of this version is to understand and build a framework with all its constituent modules (2D convolution, Adam, ReLU, Sigmoid, a container like torch.nn.Sequential, Mean Squared Error, Stochastic Gradient Descent ) that are the standard building blocks of deep networks, without PyTorch’s autograd (which we reimplemented from scratch).

This project was done as an assignment of the EPFL course [EE-559 Deep Learning](https://edu.epfl.ch/coursebook/en/deep-learning-EE-559). The instructions given by the professor can be found in the pdf **Project description.pdf**. The provided dataset were 50000 pairs of 3 channels 32 × 32 images, corrupted by two
different noises. 

The following image is an example of the performance of the original Noise2Noise architecture:

<p align="center">
<img width="517" alt="Immagine 2022-08-08 182412" src="https://user-images.githubusercontent.com/62103572/183466131-805b2ae2-1d27-4592-baf7-595edc62c304.png">
</p>

## Authors

- [Elia Fantini](https://github.com/EliaFantini/)
- [Kieran Vaudaux](https://github.com/KieranVaudaux)
- [Kaan Okumuş](https://github.com/okumuskaan)

## Noise2Noise Lite results
The main focus of our experiments was to achieve the fastest
convergence possible in order to get the best performance
in less than 10 minutes of training. The key to achieve this
result was to reduce the size of the original architecture (based on U-Net) thanks
to weight sharing, and another improvement was given by
reducing the channels’ depth. 

The best result was obtained
with a learning rate for Adam of 0.001, but since we aimed at
a fast convergence, we have chosen as final model the solution
starting with a learning rate of 0.006, reduced by a scheduler
over time. 

This way our model reaches a PSNR higher than 25
dB after the 2nd epoch, 25.6 dB at the 12th epoch and 25.61
at the 20th, after only 455 seconds. Longer training did not result into
significantly better results, hence the fastest solution is also
the best we could achieve. Every test loss was computed on
a random subset of the validation data to avoid overfitting,
since the scheduler computations were dependant of the test
loss values. The final PSNR on the whole set is 25.55 dB.

The following table and plots shows the different results obtained with different architectures and hyperparameters.
<p align="center">

<img width="538" alt="55" src="https://user-images.githubusercontent.com/62103572/183502653-4146267e-d24e-48c1-a7b5-218c2746cdf0.png">
</br>
<img width="auto" alt="44" src="https://user-images.githubusercontent.com/62103572/183502643-4104fefc-d419-41fb-a0cd-d6521cc02b96.png">
</p>

This image show an example of the output of the final (best) model obtained:
<p align="center">
<img width="auto" alt="33" src="https://user-images.githubusercontent.com/62103572/183502650-75d817e8-65e3-47d9-81dc-2aa0f6cc29d7.png">
</p>

For more details, please read **Report_1.pdf**.

##  Noise2Noise from scratch results

We managed to achieve a PSNR
of 22.6 dB. Subsequently, we tried to improve this result by
running further optimisations based on this ”best model”, but
whether by increasing or decreasing the learning rate, or by
varying the parameters β1, β2 and/or the batch size, we did
not manage to obtain significantly better results. Nevertheless,
the results of our denoising seem
to be correct, even if we notice that our predictions remain
blurrier than the target images.

The main objective of this part of the project was the **implementation
of a from-scratch framework** to be able to reproduce, to some
extent, the results we obtained in Noise2Noise Lite. While the
first project allowed us to achieve a PSNR of 25.6 dB using
the Pytorch environment and a network with a fairly large
architecture, this second project allowed us to achieve a PSNR
of 22.6 dB with a fairly modest network size. 

The key factor
that allowed us to achieve this result was the implementation of
another optimizer than the SGD, namely the Adam optimizer.
Subsequently, most of our efforts were focused on optimising
the parameters of the Adam optimizer. 

As a future improvement of this project, the implementation
of a new Upsampling module using Transposed Convolution
would be interesting to study in comparison with the one we
have implemented which combines Nearest Neighbor upsampling + Convolution.


The two following images show an example of the ouput of the final model and the final architecture chosen.
<p align="center">

<img width="500" alt="66" src="https://user-images.githubusercontent.com/62103572/183504236-d6670dee-f82a-48ca-a317-2fe3939a04ce.png">
</br>
<img width="500" alt="99" src="https://user-images.githubusercontent.com/62103572/183504240-74f370c4-6c4d-4e2f-9d77-ff8b11272f22.png">
</p>


For more details, please read **Report_2.pdf**.


## How to install:
Download this repository as a zip file and extract it into a folder.

This project has been developed and tested with python 3.8. The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official website (select python of version 3): https://www.anaconda.com/download/

The only additional library required is PyTorch for the implementation of deep learning models and methods. If you don't have it, you can download it by following the instructions [here](https://pytorch.org/).

Then, download the dataset available [here](https://drive.google.com/drive/u/2/folders/1CYsJ5gJkZWZAXJ1oQgUpGX7q5PxYEuNs) (unfortunately the drive is protected and it's for EPFL strudents only, the files were too big to be uploaded on GitHub), and put `train_data.pkl` and `val_data.pkl` inside both `Noise2Noise Lite\others\dataset` and `Noise2Noise from scratch\others\dataset` folders. 

## How to use

Run on your terminal (or on Anaconda Prompt if you choose to install anaconda) the following command that does all the testings: 
```bash
python test.py -p "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*" -d "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*/Noise2Noise Lite/others/dataset/"
```

You can also test Conv2D function by comparing it with the PyTorch one. To do so, run:
```bash
python test.py -p "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*" -d "./*NAME_OF_THE_FOLDER_WHERE_YOU_EXTRACTED_THE_REPOSITORY*/Noise2Noise Lite/others/dataset/"
```
The file **test.py** was created by the professor and his assistants to test the code, otherwise you can directly run the file **__init__.py** in both folders to directly run a full training of both model variants. Noise2Noise Lite folder also contains the jupyter notebook **Experiments.ipynb** with all the experiments that guided the final architecture choice.
## Project Structure:

```bash
├── Noise2Noise Lite
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
└── Noise2Noise from scratch
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

Python, Pytorch. Deep learning knowledge, good knowledge of all the components that constitute a neural network and its training. Deep knowledge of the Pytorch framework to rebuild from scratch all its basics components and its core mechanism, included **autograd**. Implementation from scratch of 2D convolution, Adam optimizer, ReLU, Sigmoid, a container like torch.nn.Sequential to put together an arbitrary configuration of modules together, Mean Squared Error as a Loss Function, Stochastic Gradient Descent (SGD) optimizer.

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/EliaFantini/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
