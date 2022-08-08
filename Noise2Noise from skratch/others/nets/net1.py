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
            module.Conv2d(in_channels, depthChannels_1,(3,3), stride = 2, padding = 2, device=device),
            module.ReLU(),
            module.Conv2d(depthChannels_1, depthChannels_2,(3,3), stride = 2, padding = 2, device=device),
            module.ReLU(),
            module.Upsampling(depthChannels_2, depthChannels_1,2,(6,6), padding = 1, device=device),
            module.ReLU(),
            module.Upsampling(depthChannels_1, out_channels,2,(3,3), device=device),
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
        
    def load_param(self, param):
        self.block1.modules[0].load_param([param[0][0], param[1][0]])
        self.block1.modules[2].load_param([param[2][0], param[3][0]])
        self.block1.modules[4].load_param([param[4][0], param[5][0]])
        self.block1.modules[6].load_param([param[6][0], param[7][0]])
