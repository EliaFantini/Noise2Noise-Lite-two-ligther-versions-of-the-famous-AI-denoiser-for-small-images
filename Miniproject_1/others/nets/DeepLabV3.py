""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
import torch
def createDeepLabv3():
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    net = models.googlenet(pretrained=True,progress=True)
    print(net)
    net.layer4 = nn.Sequential(
            nn.Conv2d(1280, 200, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))
    return net