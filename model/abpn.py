import torch
import torch.nn as nn
from model.common_block import *
# from common_block import *

class basicConv(nn.Module):    
    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int=3,
                 stride:int=1,
                 padding:int=1,
                 act:bool=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        if act == True:
            self.act = nn.ReLU()
        else:
            self.act = None    
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ABPN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=49,
                 out_channels=3,
                 repeat:int=5,
                 upscale:int=4,
                 normalization=False):
        super().__init__()

        if mid_channels is None:
            mid_channels = 49
        # Raise AssertionError 
        assert repeat > 2, "Repeat value should be larger than 2" 
        self.upscale = upscale

        if not normalization:
            self.norm = False
        else:
            self.norm = True
        
        self.shallow = basic_conv(in_channels, mid_channels, 3)
        self.deep = []
        for _ in range(repeat):
            self.deep.append(
                basic_conv(mid_channels, mid_channels, 3)
            )
        self.deep.append(conv_layer(mid_channels,
                                    out_channels*upscale**2,
                                    kernel_size=3))
        
        self.body = nn.Sequential(*self.deep)
        self.upsampler = nn.PixelShuffle(upscale_factor=upscale)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        residual = torch.cat([residual for _ in range(self.upscale**2)], dim=1)      # (1, 3 * upscale**2, H, W)
        c1 = self.shallow(x)
        c2 = self.body(c1)
        c3 = c2 + residual
        c4 = self.upsampler(c3)

        if not self.norm:
            output = c4.clamp(0.0, 255.0)
        else:
            output = c4.clamp(0, 1)
        
        del residual, c1, c2, c3, c4

        return output


if __name__ == '__main__':
    net = ABPN().train()
    print(net)
    lr_input = torch.randn(1, 3, 270, 480)
    sr_output = net(lr_input)
    print(f"sr_output = {sr_output.shape}")
