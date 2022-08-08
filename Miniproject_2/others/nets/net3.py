from pathlib import Path
import os
import sys

p = Path(__file__)
p = p.resolve()
sys.path.insert(0, os.path.join(p.parents[1], ".."))

import others.helpers_layer as module

class Network():

    def __init__(self, in_channels=3, out_channels=3, depthChannels_1 = 9, depthChannels_2 = 18, device="cpu"):
            
        self.block1 = module.Sequential([
            module.Conv2d(in_channels,depthChannels_1,2,stride = 2, padding = 0, device=device),
            module.ReLU(),
            module.Conv2d(depthChannels_1,depthChannels_2,2,stride = 2, padding = 1, device=device),
            module.ReLU(),
            module.Upsampling(depthChannels_2,depthChannels_1,3,4,padding=0,stride = 2, device=device),
            module.ReLU(),
            module.Upsampling(depthChannels_1,out_channels,3,5,padding = 0, device=device),
            module.Sigmoid()
            ])
            
    def forward(self,input):
        return self.block1.forward(input)
        
    def backward(self, gradwrtouput):
        return self.block1.backward(gradwrtouput)
        
    def param(self):
        return self.block1.param()
        
    def __call__(self, input):
        return self.forward(input)
