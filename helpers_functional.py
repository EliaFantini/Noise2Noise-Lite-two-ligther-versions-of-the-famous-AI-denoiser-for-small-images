from torch.nn.functional import fold, unfold
from torch import Tensor, empty, einsum


def conv2d(input: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    # input is 4d tensor
    
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    N = input.size(0)
    H_in = input.size(-2)
    W_in = input.size(-1)
    
    kernel_size = (weight.size(-2), weight.size(-1))
    C_out = weight.size(0)
    H_out = int((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = int((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        
    unfolded = unfold(input, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)
    
    #wxb  = empty(N, C_out, unfolded.size(2))
    #for ind, unfdd in enumerate(unfolded):
    #    wxb[ind] =  weight.view(C_out, -1) @ unfdd + bias.view(-1,1)
    
    #wxb = einsum('nij,njk->nik', weight.view(1, C_out, -1).repeat(N, 1, 1), unfolded) + bias.view(1, -1, 1).repeat(N,1,1)
    
    wxb = weight.view(1, C_out, -1).repeat(N, 1, 1).matmul(unfolded) + bias.view(1, -1, 1).repeat(N,1,1)
    
    return wxb.view(N, C_out, H_out, W_out)



# TODO: output padding and stride not working
def conv_transpose2d(input: Tensor, weight: Tensor, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, output_padding=0) -> Tensor:
    # input is 4d tensor
    if isinstance(stride, int):
        stride = (stride, stride)    
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    N = input.size(0)
    H_in = input.size(-2)
    W_in = input.size(-1)
        
    kernel_size = (weight.size(-2), weight.size(-1))
    C_out = weight.size(1)
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    
    pad0 = (dilation[0] * (kernel_size[0] - 1) - padding[0]) * stride[0]
    pad1 = (dilation[1] * (kernel_size[1] - 1) - padding[1]) * stride[1]

    if (pad0<0) or (pad1<0):
        raise ValueError("Invalid inputs, transposed convolution not possible")
        
    unfolded = unfold(input, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=(pad0,pad1))
    
    w = weight.transpose(0,1).rot90(-2, [-2,-1])
        
    wxb = unfolded.transpose(1, 2).matmul(w.reshape(w.size(0), -1).t()).transpose(1, 2) + bias.view(1, -1, 1).repeat(N,1,1)

    return wxb.view(N, C_out, H_out, W_out)

