from torch import Tensor, empty
from typing import Optional, Tuple, Union, List
import others.helpers_functional as F
from math import sqrt

class Module(object):
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self,gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []
    

class Sequential(Module):
    
    def __init__(self, modules):
        self.modules = modules
        self.input = None
    
    def forward(self, input):
        self.input = input
        output = self.input
        
        for module in self.modules:
            output = module.forward(output)
        return output
    
    def backward(self,gradwrtouput):
        gradient = gradwrtouput
        for module in reversed(self.modules):
            gradient = module.backward(gradient)
        self.input = None
        return gradient
    
    def param(self):
        params = []
        for module in self.modules:
            params.extend(module.param())
        return params

    
class Optimizer(object):
    def step(self):
        return NotImplementedError
    
    def zero_grad(self):
        return NotImplementedError
    
    
class SGD(Optimizer):
    
    def __init__(self, params, lr, mu = 0, tau = 0):
        self.params = params
        self.lr = lr
        
        # parameters in order to add momemtum 
        self.momemtum = mu
        self.dampening = tau
        self.state_momemtum = None
        
    def step(self):
        if self.momemtum == 0:
            for x, grad in self.params: 
                x.add_(-self.lr * grad)
        else:
            if self.state_momemtum is None:
                self.state_momemtum = []
                for x, grad in self.params: 
                    self.state_momemtum.append(grad)
                    x.add_(-self.lr * grad)
            else:
                i = 0
                for x, grad in self.params:
                    self.state_momemtum[i] *= self.momemtum
                    self.state_momemtum[i].add_(grad,alpha = (1-self.dampening))
                    x.add_(-self.lr * self.state_momemtum[i])
                    i+=1
                
            
    
    def zero_grad(self):
        for (x, grad) in self.params: # TODO: Make sure that params have (param, grad)
            grad.zero_() #TODO: Update these grad
            
    def update_lr(self,lr, mode = "new"):
        if mode == "scale":
            self.lr *= lr
        else:
            self.lr = lr
            
class Adam(Optimizer):
    
    def __init__(self, params, lr, beta = (0.9,0.99), eps = 1e-08):
        self.params = params
        self.lr = lr
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.eps = eps
        self.moment1 = None
        self.moment2 = None
        
        self.iter = 0
        
    def step(self):
        if (self.moment1 is None) or (self.moment2 is None):
            self.moment1 = []
            self.moment2 = []
            for x, grad in self.params: 
                self.moment1.append(empty(grad.size()).fill_(0.))
                self.moment2.append(empty(grad.size()).fill_(0.))
                x.add_(-self.lr * grad)
        else:
            i = 0
            for x, grad in self.params:
                self.moment1[i] *= self.beta1
                self.moment1[i].add_(grad,alpha = (1-self.beta1))
                
                self.moment2[i] *= self.beta2
                self.moment2[i].add_(grad.pow(2),alpha = (1-self.beta2))
                
                m_ = (self.moment1[i]/(1-self.beta1**self.iter))
                v_ = (self.moment2[i]/(1-self.beta2**self.iter)) + self.eps 
                x.add_(-self.lr * m_.div(v_.sqrt()))
                i+=1
        
        self.iter += 1
                
            
    
    def zero_grad(self):
        for (x, grad) in self.params: # TODO: Make sure that params have (param, grad)
            grad.zero_() #TODO: Update these grad
            
    def update_lr(self,lr, mode = "new"):
        if mode == "scale":
            self.lr *= lr
        else:
            self.lr = lr

class MSE(Module):
    
    def forward(self, input, target):
        self.input = input
        self.target = target
        return (self.input - self.target).pow(2).mean() 
    
    def backward(self):
        return 2*(self.input - self.target).div(self.input.size(0))

        
class Sigmoid(Module):
    
    def forward(self,input):
        self.input = input
        self.sigmoid = 1./(1+(-self.input).exp())
        return  self.sigmoid
    
    def backward(self,gradwrtouput):
        return gradwrtouput*self.sigmoid*(1-self.sigmoid)
            
class ReLU(Module):
    def forward(self, input):
        self.input = input
        return (self.input>0.)*self.input
    
    def backward(self, gradwrtouput):
        return gradwrtouput*(self.input>=0.)

            

# TODO: F should have pad,

 
class NearestUpsampling(Module):
        
    def __init__(self, scale_factor: None ):
        self.scale_factor = F._pair(scale_factor)
    
    def forward(self, input):
        self.input_size = input.size()
        
        if (len(input.size()) == 4) :
            return F.nearest_upsampling(input, self.scale_factor)

        
    # the error between the our gradient and the true value of the gradient
    ## is small but not enough ...
    # for scale 2 it's quite ok
    def backward(self,gradwrtouput):
        return F.conv2d(gradwrtouput, empty(self.input_size[1],1,self.scale_factor[0],self.scale_factor[1]).fill_(1.), bias=None, stride=self.scale_factor, groups = self.input_size[1])

class Conv2d(Module):
    
    def __init__(self,in_channel, out_channel, kernel_size = (2,2),stride=1, padding=0, dilation=1, groups=1,
                 weight = None, bias = None):
        self.stride = F._pair(stride)
        self.padding = F._pair(padding)
        self.dilation = F._pair(dilation)
        self.kernel = F._pair(kernel_size)
        
        self.groups = groups
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        k = sqrt(self.groups/(self.in_channel*self.kernel[0]*self.kernel[1]))
        
        if weight == None:
            self.weight = empty((self.out_channel,self.in_channel// self.groups,
                                       self.kernel[0],self.kernel[1])).uniform_(-k,k)
        else:
            self.weight = weight
        if bias is None:
            self.bias = empty(self.out_channel).uniform_(-k,k)
        else:
            self.bias = bias
        
        self.weight_grad = empty((self.out_channel,self.in_channel// self.groups,
                                       self.kernel[0],self.kernel[1])).fill_(0.)
        self.bias_grad = empty(self.out_channel).fill_(0.)
    
    
    def forward(self, input):
        self.input = input
        
        if (len(input.size()) == 4) :
            return F.conv2d(self.input,weight = self.weight, bias = self.bias,stride =self.stride,
                                  padding = self.padding, dilation = self.dilation,groups = self.groups)
    
    
    def backward(self, grad_output):
        self.weight_grad.add_(F.grad_conv2d_weight(self.input, self.weight.shape, grad_output, self.stride,self.padding, 
                                                   self.dilation, self.groups))
        
        if self.bias is not None:
            self.bias_grad.add_(grad_output.sum((0, 2, 3)).squeeze(0))
                #grad_output.transpose(0,1).contiguous().view(grad_output.size(1),-1).sum(dim = 1))
            
        return F.grad_conv2d_input(self.input.shape, self.weight, grad_output=grad_output,stride=self.stride, 
                                   padding=self.padding, dilation=self.dilation,groups=self.groups)
    
    def param(self):
        return [(self.weight,self.weight_grad),(self.bias,self.bias_grad)]
    
    def update_params(self, params):
        self.weight = params[0]
        self.grad_weight = params[1]
        self.bias = params[2]
        self.grad_bias = params[3]
    
    
class Upsampling(Module):
    def __init__(self,in_channel, out_channel,scale_factor = 2, kernel_size = (2,2),stride=1, padding=0, 
                 dilation=1, groups=1,weight = None, bias = None):
    
        self.modules = [NearestUpsampling(scale_factor),
                   Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,weight, bias)
        ]
        self.sequential = Sequential(self.modules)
        
   
    def forward(self,input):
        return self.sequential.forward(input)
    def backward(self,gradwrtoutput):
        return self.sequential.backward(gradwrtoutput)
    def param(self):
        return self.sequential.param()
