import torch
import torch.nn as nn
from model.common_block import *

class InnoPeak(nn.Module):
    def __init__(self,
                 in_channels=3,
                 knot_channels=27,
                 mid_channels=32,
                 out_channels=3,
                 upscale=3):
        super(InnoPeak, self).__init__()

        if knot_channels is None:
            knot_channels = 27
        if mid_channels is None:
            mid_channels = 32

        self.c1 = basic_conv(in_channels, knot_channels, 3)
        self.c2 = basic_conv(knot_channels, mid_channels, 3)
        self.c3 = basic_conv(mid_channels, mid_channels, 3)
        self.c4 = basic_conv(mid_channels, knot_channels, 3)

        self.upsampler = pixelshuffle_block(knot_channels,
                                            out_channels,
                                            upscale_factor=upscale)
        
    def forward(self, x:torch.Tensor):
        out_c1 = self.c1(x)

        out_c2 = self.c2(out_c1)

        out_c3_a = self.c3(out_c2)
        out_sum_A = out_c2 + self.c3(out_c3_a)

        out_c3_c = self.c3(out_sum_A)
        out_sum_B = out_sum_A + self.c3(out_c3_c)

        out_feature = out_c1 + self.c4(out_sum_B)

        output = self.upsampler(out_feature).clamp(0, 255)

        return output


if __name__ == "__main__":
    net = InnoPeak().train()
    lr_input = torch.randn(1, 3, 64, 64)
    sr_output = net(lr_input)
    print(net)
    print(f"sr_output = {sr_output.shape}")
