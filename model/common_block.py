import torch
import torch.nn as nn
from collections import OrderedDict

# Make integer a tuple type
def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

# CNN function that padding size that equalizes the size of the I/O
def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))

    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

# Activation function for "ReLU, LeakyReLU, PReLU"
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    
    return layer

class basic_conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 act:str="ReLU"):
        super().__init__()
        
        self.conv = conv_layer(in_channels, out_channels, kernel_size, bias)
        self.act = activation(act_type=act)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x

# Modules will be added to the a Sequential Container in the order the are passed
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    
    return nn.Sequential(*modules)

# Upsample features according to 'upscale_factor'
def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=3,
                       kernel_size=3):
   
    # [*, C, H, W] -> [*, C x r^2, H, W] -> [*, C, H x r, W x r]
    # Increases the output channel in advance
    # Because nn.PixelShuffle divides it by r^2
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    return sequential(conv, pixel_shuffle)

