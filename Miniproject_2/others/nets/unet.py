import torch
import others.helpers_layer as module


class Network():

    def __init__(self, in_channels=3, out_channels=3,depthChannels_1 = 9,epthChannels_2 = 18):
        
        self.blok1 = module.Sequential([
            module.Conv2d(in_channels,depthChannels_1,(3,3),stride = 2, padding = 2),
            module.ReLU(),
            module.Conv2d(depthChannels_1,depthChannels_2,(3,3),stride = 2, padding = 2),
            module.ReLU(),
            module.Upsampling(depthChannels_2,depthChannels_1,2,(4,4),padding = 1),
            module.ReLU(),
            module.Upsampling(depthChannels_1,out_channels,2,(3,3)),
            module.Sigmoid()
            ])
            
    def forward(self,input):
        return self.blok1.forward(input)
        
    def backward(self, gradwrtouput):
        return self.block1.backward(gradwrtouput)
