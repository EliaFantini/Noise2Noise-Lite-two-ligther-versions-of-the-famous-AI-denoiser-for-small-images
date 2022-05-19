from torch import Tensor, empty
from typing import Optional, Tuple, Union, List
import helpers_functional as F

class Module(object):
    
    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self,*gradwrtoutput):
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
    
    def backward(self,*gradwrtouput):
        gradient = gradwrtouput
        for module in reversed(self.modules):
            gradient = module.backward(gradient)
        self.input = None
        return gradient
    
    def param(self):
        params = []
        for module in self.modules:
            params.append(module.param())
        return params
    


# TODO: Groups argument & padding mode is not taken care!

class _ConvNd(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'padding_mode', 'in_channels', 'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[Tensor]}
    
    def _conv_forward(self, input: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...
        
    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
            
    def __init__(self, in_channels: int,
                out_channels: int, kernel_size: Tuple[int, ...],
                stride: Tuple[int, ...], padding: Tuple[int, ...],
                dilation: Tuple[int, ...], transposed: bool,
                output_padding: Tuple[int, ...],  
                groups: int, bias: bool, padding_mode: str, 
                device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError("Invalid padding string {!r}, should be one of {}".format(padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' not supported for strided convolutions")
        
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode={}".format(valid_padding_modes, padding_mode))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        
        # TODO: Replace Parameter class
        if transposed:
            self.weight = empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs
            )
        else:
            self.weight = empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs
            )
        if bias:
            self.bias = empty(out_channels, **factory_kwargs)
        else:
            self.bias = None
        
        self.reset_params()
    
    def reset_params(self) -> None:
        # TODO: Reset weight and bias. Sample from U(-sqrt(k), sqrt(k))
        if isinstance(self.kernel_size, int):
            prod_kernel_size = self.kernel_size**2
        else:
            prod_kernel_size = self.kernel_size[0]*self.kernel_size[1]
        sqrt_k = 1/(self.weight.size(1) * prod_kernel_size)**(0.5)
        self.weight.uniform_(-sqrt_k, sqrt_k)
        if self.bias is not None:
            self.bias.uniform_(-sqrt_k, sqrt_k)
            

# TODO: F should have pad, 

def _pair(x):
    if isinstance(x, int):
        return (x, x)
    return x

class Conv2d(_ConvNd):
    
    # TODO: replace _size_2_t
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # TODO: replace _pair function
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, False, 
            _pair(0), groups, bias, padding_mode, **factory_kwargs
        )
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            # TODO: Handle non zeros padding. Replace REVERSED_PADDING
            return F.conv2d(F.pad(input, REVERSED_PADDING, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)   
    
    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    

class _ConvTransposeNd(_ConvNd):
    def __init__(self, in_channels: int, 
                 out_channels: int, kernel_size: Tuple[int,...], 
                 stride: Tuple[int,...], padding: Tuple[int,...], 
                 dilation: Tuple[int,...], transposed: bool, 
                 output_padding: Tuple[int,...],
                 groups: int, bias: bool, padding_mode: str, 
                 device=None, dtype=None) -> None:
        
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_ConvTransposeNd, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode, **factory_kwargs)
        
    def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
                       stride: List[int], padding: List[int], kernel_size: List[int],
                       dilation: Optional[List[int]] = None) -> List[int]:
        if output_size is None:
            if isinstance(self.output_padding, int):
                ret = self.output_padding
            else:
                ret = self.output_padding[0]
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError("output_size must have {} or {} elements (got {})"
                                .format(k, k + 2, len(output_size)))
            
            min_sizes = []
            max_sizes = []
            
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] - 
                            2 * padding[d] + 
                           (dilation[d] if (dilation is not None) else 1) *
                           (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)
                
            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        "requested an output size of {}, but valid sizes range from {} to {} (for an input of {})"
                        .format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = []
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
            
        return ret


class ConvTranspose2d(_ConvTransposeNd):
    
    # TODO: replace _size_2_t, _pair
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride_ = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride_, padding,
            dilation, True, output_padding, groups, bias, padding_mode,
            **factory_kwargs)
        
    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for ConvTranspose2d')
            
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,
            self.dilation)
        return F.conv_transpose2d(
            input, self.weight, self.bias, stride=self.stride, padding=self.padding,
            output_padding=output_padding, groups=self.groups, dilation=self.dilation)
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
