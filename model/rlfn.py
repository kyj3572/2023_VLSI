import torch
import torch.nn as nn
from model import rlfn_block
        
class RLFN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3, 
                 feature_channels:int=52,
                 upscale:int=4):
        super().__init__()

        if feature_channels is None:
            feature_channels = 52    # DEFAULT
        
        self.conv_1 = rlfn_block.conv_layer(in_channels,
                                            feature_channels,
                                            kernel_size=3)
        
        self.block_1 = rlfn_block.RLFB(feature_channels)
        self.block_2 = rlfn_block.RLFB(feature_channels)
        self.block_3 = rlfn_block.RLFB(feature_channels)
        self.block_4 = rlfn_block.RLFB(feature_channels)
        self.block_5 = rlfn_block.RLFB(feature_channels)
        self.block_6 = rlfn_block.RLFB(feature_channels)
        
        self.conv_2 = rlfn_block.conv_layer(feature_channels,
                                            feature_channels,
                                            kernel_size=3)
        
        self.upsampler = rlfn_block.pixelshuffle_block(feature_channels,
                                                       out_channels,
                                                       upscale_factor=upscale)
        
    def forward(self, x:torch.Tensor):
        out_feature = self.conv_1(x)
        
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)
        
        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)
        output.clamp(0, 1)

        return output
    

if __name__ == "__main__":
    net = RLFN().train()
    lr_input = torch.randn(1, 3, 64, 64)
    sr_output = net(lr_input)
    print(net)
    print(f"sr_output = {sr_output.shape}")