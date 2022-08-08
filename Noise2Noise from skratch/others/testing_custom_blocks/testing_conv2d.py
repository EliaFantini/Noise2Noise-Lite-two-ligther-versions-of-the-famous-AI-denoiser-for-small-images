import time
import torch
from torch import empty

from pathlib import Path
import os
import sys

# Paths should be added to import necessary functions
p = Path(__file__)
p = p.resolve()
sys.path.insert(0, os.path.join(p.parents[0], ".."))
sys.path.insert(0, os.path.join(p.parents[1], ".."))

import helpers_functional as F
from helpers_layer import *

# Testing of Conv2d Layer:

def test_conv(i, N, in_channels, out_channels, kernel_size, padding, stride, dilation, groups):
    # Testing Function with given parameters
    
    # PyTorch Conv2D class
    conv_pytorch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # Custom Conv2D_class
    conv_custom = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, weight=conv_pytorch.weight, bias=conv_pytorch.bias)
    
    x = torch.randn((N, in_channels, 32, 44))

    print("\n#", i, " Testing  of Conv2D:")
    print("-"*100)
    t1 = time.time()
    expected = conv_pytorch(x)
    t2 = time.time()
    res_layer = conv_custom.forward(x)
    t3 = time.time()

    print("\n\tShape of Output of PyTorch Implementation:\t", expected.shape)
    print("\tShape of Output of Custom Implementation:\t", res_layer.shape)

    torch.testing.assert_allclose(expected, res_layer)
    print("\n\t-> Passed the Second Test: Pytorch Implementation & Custom Implementation are the Same")

    print("\n\tComputation Time Comparison:")
    print("\t\tPytorch Conv2d Layer Forward: \t\t", t2-t1, "\tsec")
    print("\t\tImplemented Conv2d Layer Forward: \t", t3-t2, "\tsec")

    print("")

def test_all():
    # Testing Different Parameters Function:
    N = [1000, 1000, 1000, 1000]
    in_channels = [4, 10, 2, 6]
    out_channels = [10, 40, 6, 12]
    kernel_size = [(2, 3), (6,3), (4,4), (2,4)]
    padding = [(2,3), (5,2), [6,4], [4,3]]
    stride = [1, 2, 2, 2]
    dilation = [1, 1, 2, 2]
    groups = [1, 1, 2, 3]
    print("\nTestings starting...\n")
    print("*"*100)
    for i in range(4):
        test_conv(i+1, N[i], in_channels[i], out_channels[i], kernel_size[i], padding[i], stride[i], dilation[i], groups[i])
    
    print("*"*100)
    print("\n\t->All Tests are Passed Successfully\n")
    


if __name__ == "__main__":
    # Run all the tests
    test_all()
