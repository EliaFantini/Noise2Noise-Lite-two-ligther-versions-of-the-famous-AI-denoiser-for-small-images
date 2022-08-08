"""
This script implements all the layers for the framework including.

    The implemented layers inside this script can be split into two groups:
        
        - Module Layer: (Module also implemented)
            + Sequential
            + MSE
            + Sigmoid
            + ReLU
            + NNUpsampling
            + Conv2d
            + Upsampling
            
        - Optimizer Layer: (Optimizer also implemented)
            + SGD
            + Adam

"""


from torch import empty
import others.helpers_functional as F
from math import sqrt


# MODULE LAYERS:


class Module(object):
    """
        Module Class
    """
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self,gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []
        
    def load_param(self):
        pass

    
    
    
class Sequential(Module):
    """
        Sequential Module
    """
    def __init__(self, *modules):
        self.modules = []
        for module in modules:
            if isinstance(module, tuple) or isinstance(module, list):
                for mdl in module:
                    self.modules.append(mdl)
            else:
                self.modules.append(module)
        self.input = None
    
    def forward(self, input):
        self.input = input
        output = self.input
        for module in self.modules:
            output = module.forward(output)
        return output
    
    def backward(self, gradwrtouput):
        gradient = gradwrtouput
        for module in reversed(self.modules):
            gradient = module.backward(gradient)
        self.input = None
        return gradient
    
    def param(self):
        params = []
        # Parameters of each modules added in a list:
        for module in self.modules:
            params.extend(module.param())
        return params
        
    def load_param(self, param):
        for module in self.modules:
            module.load_param(param)
            
    def summary(self):
        print("\nNetwork Summary:")
        print("*"*100)
        for i, module in enumerate(self.modules):
            module_info = module.info()
            print(" - Layer "+str(i+1)+": " + module_info)
            print("-"*50)
        print("*"*100)




class MSE(Module):
    """
        Mean Square Error Layer
    """
    def forward(self, input, target):
        self.input = input
        self.target = target
        return (self.input - self.target).pow(2).mean() 
    
    def backward(self):
        return 2*(self.input - self.target).div(self.input.size(0))
        
    def __call__(self, input, target):
        return self.forward(input, target)
        
        

        
        
class Sigmoid(Module):
    """
        Sigmoid Layer
    """
    def forward(self,input):
        self.input = input
        self.sigmoid = 1./(1+(-self.input).exp())
        return  self.sigmoid
    
    def backward(self,gradwrtouput):
        return gradwrtouput*self.sigmoid*(1-self.sigmoid)
        
    def info(self):
        return "Sigmoid Layer"
           
           
           
           
class ReLU(Module):
    """
        Rectified Linear Unit Layer
    """
    def forward(self, input):
        self.input = input
        return (self.input>0.)*self.input
    
    def backward(self, gradwrtouput):
        return gradwrtouput*(self.input>=0.)
        
    def info(self):
        return "ReLU Layer"

            

 
class NNUpsampling(Module):
    """
        Nearest Neighbor Upsampling Layer
    """
    def __init__(self, scale_factor: None, device="cpu"):
        self.scale_factor = F._pair(scale_factor)
        self.device=device
    
    def forward(self, input):
        self.input_size = input.size()
        
        if (len(input.size()) == 4) :
            return F.nearest_upsampling(input, self.scale_factor)

    def backward(self,gradwrtouput):
        mtx = empty(self.input_size[1],1,self.scale_factor[0],self.scale_factor[1], device=self.device).fill_(1.)
        return F.conv2d(gradwrtouput, mtx, bias=None, stride=self.scale_factor, groups = self.input_size[1], device=self.device)


    
    
class Conv2d(Module):
    """
        2D Convolution Layer
    """
    
    def __init__(self, in_channel, out_channel, kernel_size = (2,2),stride=1, padding=0, dilation=1, groups=1, weight = None, bias = None, device="cpu"):
        
        # Stride, padding, dilation, kernel is made to be Tuple(int, int)
        self.stride = F._pair(stride)
        self.padding = F._pair(padding)
        self.dilation = F._pair(dilation)
        self.kernel = F._pair(kernel_size)
        
        self.groups = groups
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.device = device
        
        # If weight parameter is not given, it's randomly generated from Unif[-k, k] distribution.
        # Here, this distribution is called Kaiming Uniform.
        k = sqrt(self.groups/(self.in_channel*self.kernel[0]*self.kernel[1]))
        
        if weight == None:
            self.weight = empty((self.out_channel,self.in_channel// self.groups,
                                       self.kernel[0],self.kernel[1]), device=device).uniform_(-k,k)
        else:
            self.weight = weight
        if bias is None:
            self.bias = empty(self.out_channel, device=device).uniform_(-k,k)
        else:
            self.bias = bias
        
        # Grad parameters full of zeros:
        self.weight_grad = empty((self.out_channel,self.in_channel// self.groups,
                                       self.kernel[0],self.kernel[1]), device=device).fill_(0.)
        self.bias_grad = empty(self.out_channel, device=device).fill_(0.)
    
    
    def forward(self, input):
        self.input = input
        # Only 4d-tensors are allowed:
        if (len(input.size()) == 4) :
            # Apply 2d convolution:
            return F.conv2d(self.input, weight = self.weight, bias = self.bias,stride =self.stride,
                                  padding = self.padding, dilation = self.dilation, groups = self.groups, device=self.device)
    
    
    def backward(self, grad_output):
    
        # Update Gradient of Weight:
        self.weight_grad.add_(F.grad_conv2d_weight(self.input, self.weight.shape, grad_output, self.stride, self.padding, self.dilation, self.groups, device=self.device))
        
        # Update Gradient of Bias:
        if self.bias is not None:
            self.bias_grad.add_(grad_output.sum((0, 2, 3)).squeeze(0))
            
        # Gradient with respect to input returned:
        return F.grad_conv2d_input(self.input.shape, self.weight, grad_output=grad_output,stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, device=self.device)
    
    
    def param(self):
        # Parameters stored with their grad
        return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]
    
    def update_params(self, params):
        # Updates self parameters
        self.weight = params[0]
        self.grad_weight = params[1]
        self.bias = params[2]
        self.grad_bias = params[3]
                                      
    def load_param(self, *param):
        self.weight = param[0][0]
        self.bias = param[0][1]
        self.grad_weight = param[0][2]
        self.grad_bias = param[0][3]
        
    def info(self):
        weight_info = "(" + str(self.weight.size(0))+", "+str(self.weight.size(1))+", "+str(self.weight.size(2))+", "+str(self.weight.size(3)) + ")"
        bias_info = "(" + str(self.bias.size(0))+")"
        return "Convolution 2D \n\tWeight params of shape " + weight_info + "\n\tBias params of shape " + bias_info
    
    
    
    
class Upsampling(Module):
    """
        Upsampling Module (Equivalently Nearest Neighbor Upsampling + Convolution Layers)
    """
    def __init__(self, in_channel, out_channel, scale_factor = 2, kernel_size = (2,2), stride=1, padding=0, dilation=1, groups=1, weight = None, bias = None, device="cpu"):
        
        # Sequential model that has Nearest Neighbor Upsampling + Convolution Layers
        self.modules = [NNUpsampling(scale_factor),
                   Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,weight, bias, device=device)
        ]
        self.sequential = Sequential(self.modules)
        
    def forward(self,input):
        return self.sequential.forward(input)
        
    def backward(self,gradwrtoutput):
        return self.sequential.backward(gradwrtoutput)
        
    def param(self):
        return self.sequential.param()
        
    def load_param(self, *param):
        self.modules[1].load_param(param[0])
        
      
    def info(self):
        weight_info = "(" + str(self.modules[1].weight.size(0))+", "+str(self.modules[1].weight.size(1))+", "+str(self.modules[1].weight.size(2))+", "+str(self.modules[1].weight.size(3)) + ")"
        bias_info = "(" + str(self.modules[1].bias.size(0))+")"
        return "Upsampling \n\tWeight params of shape " + weight_info + "\n\tBias params of shape " + bias_info
    
    
    

# OPTIMIZER LAYERS:


class Optimizer(object):
    """
        Optimizer Class
    """
    def step(self):
        return NotImplementedError
    
    def zero_grad(self):
        return NotImplementedError
    
    
    
    
class SGD(Optimizer):
    """
        Stochastic Gradient Descent Optimizer
    """
    def __init__(self, params, lr, mu = 0, tau = 0):
        self.params = params
        self.lr = lr
        
        # Parameters in order to add momentum
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
        for (x, grad) in self.params:
            grad.zero_()
            
    def update_lr(self,lr, mode = "new"):
        if mode == "scale":
            self.lr *= lr
        else:
            self.lr = lr
            
            
            
            
class Adam(Optimizer):
    """
        Adam Optimizer
    """
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
        for (x, grad) in self.params:
            grad.zero_()
            
    def update_lr(self,lr, mode = "new"):
        if mode == "scale":
            self.lr *= lr
        else:
            self.lr = lr
