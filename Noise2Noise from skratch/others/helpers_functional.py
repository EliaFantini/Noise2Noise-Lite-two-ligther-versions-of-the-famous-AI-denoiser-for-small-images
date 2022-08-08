"""
This script implements some functionals that can be needed for the framework. The implemented functionals inside this script can be split into three groups:
        
        - Miscellaneous functionals:
            + _pair
            + pad
            + zero_internal_pad
            + _grad_input_padding
            
        - Functionals that have equivalent layers:
            + conv2d
            + conv_transpose2d
            + nearest_upsampling
            
        - Grad Functionals that return gradient wrt some parameters:
            + grad_conv2d_input
            + grad_conv2d_weight

"""


from torch.nn.functional import fold, unfold
from torch import empty


# MISCELLANEOUS FUNCTIONALS:

def _pair(x):
    """
        Helper function that pairs integer input: a -> (a, a) given a is integer.
        
        @Param:
        x: int or Tuple(int, int)
        
        @Returns:
        output: if a is int, (a, a), else returns a
    """
    if isinstance(x, int):
        return (x, x)
    return x


def pad(input, pad=(1,1,1,1), mode="constant", value=0.0, device="cpu"):
    """
        Helper function that pads to input. (Used inside transpose convolution 2d)
        
        @Param:
        input: input tensor
        pad: (pad_left, pad_right, pad_up, pad_down)
        mode: constant
        value: value of padded element
        
        @Returns:
        output: zero padded tensor
    """
    # If 2-tuple pad is given, then it's made to 4-tuple with pad_left=pad_right & pad_up = pad_down:
    if len(pad) == 2:
        pad = (pad[0], pad[0], pad[1], pad[1])
        
    input_shape = [input.size(0), input.size(1), input.size(2), input.size(3)]
    
    # Padding:
    input_shape[3] += (pad[0] + pad[1])
    input_shape[2] += (pad[2] + pad[3])
    i1 = pad[2]
    j1 = pad[0]
    i2 = input_shape[2] - pad[3]
    j2 = input_shape[3] - pad[1]
    result = empty(input_shape, device=device).fill_(value)
    result[:,:,i1:i2,j1:j2] = input
    return result


def zero_internal_pad(x, pad=(1,1), device="cpu"):
    """ 
        Adds zero padding between elements of input x with certain numbers
        @Param:
        x: input tensor
        pad: (pad_horizontal, pad_vertical)
    """
    x_shape = [x.size(0), x.size(1), x.size(2), x.size(3)]
    x_shape[2] += (pad[0]*(x_shape[2]-1))
    x_shape[3] += (pad[1]*(x_shape[3]-1))
    res = empty(x_shape, device=device).fill_(0.0)
    res[:,:,::pad[0]+1,::pad[1]+1] = x
    return res


def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size, dilation=None):
    if dilation is None:
        # For backward compatibility
        warnings.warn("_grad_input_padding 'dilation' argument not provided. Default of 1 is used.")
        dilation = [1] * len(stride)

    input_size = list(input_size)
    k = grad_output.dim() - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError("input_size must have {} elements (got {})"
                         .format(k + 2, len(input_size)))

    def dim_size(d):
        return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] + 1
                + dilation[d] * (kernel_size[d] - 1))

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an input grad size of {}, but valid sizes range "
                 "from {} to {} (for a grad_output of {})").format(
                     input_size, min_sizes, max_sizes,
                     grad_output.size()[2:]))

    return tuple(input_size[d] - min_sizes[d] for d in range(k))



# FUNCTIONALS THAT HAVE EQUIVALENT LAYERS:

def nearest_upsampling(input, scale_factor):
    """
        Nearest Upsampling functional
    """
    scale = _pair(scale_factor)
    return input.repeat_interleave(scale[0], dim = 2).repeat_interleave(scale[1], dim = 3)
    
    
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, device="cpu"):
    """
        2d Convolution functional
        
        @Param:
        input - input tensor of shape (N, C_in, H_in, W_in)
        weight - filters of shape (C_in, C_out/groups, kH, kW)
        bias - optional bias of shape (c_out). Default: None
        stride - the stride of the convolving kernel. Either int or (sH, sW). Default: 1
        padding - implicit paddings on input. Either int or (padH, padW). Default: 0
        groups - split input into groups, in_channels should be divisible by the number of groups. Default: 1
        dilation - the spacing between kernel elements. Either int or (dH, dW). Default: 1
        
        @Returns:
        res - output tensor of shape (N, C_out, H_out, W_out)
    """
    
    # Necessary parameters are paired if they have only single element:
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # Getting input and output shape statistics:
    N = input.size(0)
    C_in = input.size(1)
    H_in = input.size(-2)
    W_in = input.size(-1)
    kernel_size = (weight.size(-2), weight.size(-1))
    C_out = weight.size(0)
    H_out = int((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = int((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    
    # Initialize output tensor's shape:
    out_conv = empty(N, C_out, H_out, W_out, device=device)

    # Apply unfold operation into input:
    inp_unf = unfold(input.contiguous().view(N*groups, C_in//groups, H_in, W_in), kernel_size, dilation, padding, stride)
    inp_unf = inp_unf.contiguous().view(N,groups,inp_unf.size(-2),inp_unf.size(-1)).transpose(0,1)

    # Do resizing to weight such that it can be used for more than one group:
    weight = weight.contiguous().view(groups, C_out//groups, C_in//groups, kernel_size[0], kernel_size[1])

    # For each group:
    for i in range(groups):
        # Compute flattened output with weight multiplication without bias:
        out_unf = inp_unf[i,...].transpose(1,2).matmul(weight[i,...].view(C_out//groups, -1).t()).transpose(1, 2)
        # Resize into desired output shape:
        out_conv[:,i*(C_out//groups):(1+i)*(C_out//groups),...] = out_unf.view(N,C_out//groups,H_out,W_out)
    
    # If bias is given, added:
    if bias!=None:
        return out_conv + bias.view(1, -1, 1,1).repeat(N,1,H_out,W_out)
    
    return out_conv
    

def conv_transpose2d(input, weight, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, output_padding=0, device="cpu"):
    """
        2d Transpose Convolution functional
        
        @Param:
        input - input tensor of shape (N, C_in, H_in, W_in)
        weight - filters of shape (C_in, C_out/groups, kH, kW)
        bias - optional bias of shape (c_out). Default: None
        stride - the stride of the convolving kernel. Either int or (sH, sW). Default: 1
        padding - "dilation * (kernel_size - 1)" zero_padding will be added to input tensor. Either int or (padH, padW). Default: 0
        output_padding - additional size added to one side of each dimension in output shape. Either int or (out_padH, out_padW). Default: 0
        groups - split input into groups, in_channels should be divisible by the number of groups. Default: 1
        dilation - the spacing between kernel elements. Either int or (dH, dW). Default: 1
        
        @Returns:
        res - output tensor of shape (N, C_out, H_out, W_out)
    """
    
    # Necessary parameters are paired if they have only single element:
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)

    # Getting input and output shape statistics:
    N = input.size(0)
    C_in = input.size(1)
    H_in = input.size(-2)
    W_in = input.size(-1)
    kernel_size = (weight.size(-2), weight.size(-1))
    C_out = weight.size(1)*groups
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1 + output_padding[0]
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1 + output_padding[1]
    
    # Initialize output tensor's shape:
    out_conv = empty(N, C_out, H_out, W_out, device=device)
    
    # Zero-padding numbers calculated:
    pad0 = (dilation[0] * (kernel_size[0] - 1) - padding[0])
    pad1 = (dilation[1] * (kernel_size[1] - 1) - padding[1])

    # If at least one of them is negative, Raise an Error!:
    if (pad0<0) or (pad1<0):
        raise ValueError("Invalid inputs, transposed convolution not possible")
    
    # Zero-padding to input:
    if (stride[0]>1) or (stride[1]>1):
        input = zero_internal_pad(input, pad=(stride[0]-1,stride[1]-1), device=device)
    if (output_padding[0]>0) or (output_padding[1]>0):
        input = pad(input, pad=(0,output_padding[1],0,output_padding[0]), device=device)
        
    # Unfolding operation:
    inp_unf = unfold(input.contiguous().view(N*groups, C_in//groups, input.size(2), input.size(3)), kernel_size=kernel_size, dilation=dilation, stride=1, padding=(pad0, pad1))
    inp_unf = inp_unf.contiguous().view(N,groups,inp_unf.size(-2),inp_unf.size(-1)).transpose(0,1)
    
    # Rotation and Transpose of calculated weight:
    weight = weight.transpose(0,1).rot90(-2, [-2,-1])
    weight = weight.contiguous().view(groups, weight.size(0), weight.size(1)//groups, kernel_size[0], kernel_size[1])
    
        # For each group:
    for i in range(groups):
        # Compute flattened output with weight multiplication without bias:
        w = weight[i,...]
        inp_unf_i = inp_unf[i,...].transpose(1,2)
        out_unf = inp_unf_i.matmul(w.reshape(w.size(0), -1).t()).transpose(1, 2)
        # Resize into desired output shape:
        out_conv[:,i*(C_out//groups):(1+i)*(C_out//groups),...] = out_unf.view(N,C_out//groups,H_out,W_out)
        
    if bias != None:
        return out_conv + bias.view(1, -1, 1,1).repeat(N,1,H_out,W_out)
    
    return out_conv
    

# GRAD FUNCTIONALS:

def grad_conv2d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1, device="cpu"):
    
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    N = input.shape[0]

    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1, 1)
    grad_output = grad_output.contiguous().view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2], grad_output.shape[3])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3])

    grad_weight = conv2d(input, grad_output, None, dilation, padding,
                               stride, in_channels * N, device=device)

    grad_weight = grad_weight.contiguous().view(
        N, grad_weight.shape[1] // N, grad_weight.shape[2],
        grad_weight.shape[3])

    return grad_weight.sum(dim=0).view(
        in_channels // groups, out_channels,
        grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(2, 0, weight_size[2]).narrow(3, 0, weight_size[3])
                
                
def grad_conv2d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1, device="cpu"):

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = (weight.shape[2], weight.shape[3])

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size, dilation)

    return conv_transpose2d(grad_output, weight, None, stride = stride, padding = padding, output_padding = grad_input_padding, groups = groups, dilation = dilation, device=device)


