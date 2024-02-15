import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.functional.image \
    import structural_similarity_index_measure

class InnoPeak_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_lambda = 1.0
        self.ssim_lambda = 0.3
        self.vgg_lambda = 0.3

    def _perceptual_loss(self, input, target):
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg19 = nn.Sequential(*list(vgg.features[:37])).to(input.device)
        pc_loss = nn.L1Loss()(vgg19(input), vgg19(target))
        return pc_loss

    def forward(self, sr:torch.Tensor, hr:torch.Tensor):
        l1_loss = nn.L1Loss()(sr, hr)
        ssim_loss = 1 - structural_similarity_index_measure(sr, hr)
        perceptual_loss = self._perceptual_loss(sr, hr)

        # print("L1 : {:.3f}, SSIM : {:.3f}, VGG : {:.3f}"
        #       .format(l1_loss, ssim_loss, perceptual_loss))

        loss = (self.l1_lambda * l1_loss) + \
               (self.ssim_lambda * ssim_loss) + \
               (self.vgg_lambda * perceptual_loss)

        del l1_loss, ssim_loss, perceptual_loss
        return loss
    
if __name__ == "__main__":
    sr = torch.rand(1, 3, 64, 64)
    hr = torch.rand(1, 3, 64, 64)

    loss_fn = InnoPeak_loss()

    loss = loss_fn(sr, hr)

    print(f'loss: {loss.item()}')
