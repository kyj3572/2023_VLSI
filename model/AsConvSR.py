import torch
import torch.nn as nn

from Assembled_block import AssembledBlock

class AsConvSR(nn.Module):
    def __init__(
        self,
        scale_factor:int=2,
        device=torch.device('cpu')
    ):
        super().__init__()
        self.pixelUnShuffle = nn.PixelUnshuffle(scale_factor)
        self.conv1 = nn.Conv2d(3*scale_factor**2, 32, kernel_size=3, stride=1, padding=1)
        self.assemble = AssembledBlock(32, 32, E=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)                                                
        self.pixelShuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixelUnShuffle(x)      # (B, 3, H, W) -> (B, 12, H/2, W/2)
        x = self.conv1(x)               # (B, 32, H/2, W/2)
        x = self.assemble(x)
        x = self.conv2(x)
        x = self.pixelShuffle(x)
        
        residual = x                    # (B, 3, H, W)
        residual = torch.cat([residual for _ in range(self.upscale_ratio**2)], dim=1)      # (B, 3 * upscale_ratio**2, H, W)
        x = torch.add(x, residual)
        x = self.pixelShuffle(x)        # (B, 3, H, W)
        
        return x
