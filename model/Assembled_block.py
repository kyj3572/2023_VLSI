import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlModule(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        hidden_channels:int=8,
        E:int=4     # # of candidate convolution kernels
    ):
        super().__init__()
        self.out_channels = out_channels
        self.E = E

        self.avgpool = nn.AdaptiveAvgPool2d(1)        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, E * out_channels, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        coeff = self.avgpool(x)                                             # (B, 32, H/2, W/2) -> (B, 32, 1, 1)
        coeff = self.conv(coeff).view(x.shape[0], -1)                       # (B, E * out_channels)
        coeff = coeff.view(coeff.shape[0], self.out_channels, self.E)       # (B, out_channels, E)
        coeff = F.softmax(coeff/30, dim=2)                                  # 0 ~ 1 사이의 값으로 정규화

        return coeff
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
class AssembledBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        groups=1,
        bias=True,
        E=4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.E = E
        
        self.control_module = ControlModule(in_channels, out_channels, E)
        self.weight1 = nn.Parameter(torch.randn(E, out_channels, in_channels // groups, kernel_size, kernel_size), requires_grad=True)
        self.weight2 = nn.Parameter(torch.randn(E, out_channels, out_channels // groups, kernel_size, kernel_size), requires_grad=True)
        self.weight3 = nn.Parameter(torch.randn(E, out_channels, out_channels // groups, kernel_size, kernel_size), requires_grad=True)
        
        if self.bias:
            self.bias1 = nn.Parameter(torch.randn(E, out_channels), requires_grad=True) # E, out_channels
            self.bias2 = nn.Parameter(torch.randn(E, out_channels), requires_grad=True) # E, out_channels
            self.bias3 = nn.Parameter(torch.randn(E, out_channels), requires_grad=True) # E, out_channels
    
    def forward(self, x):
        bs, in_channels, h, w = x.shape                             # (B, 32, H/2, W/2)
        coeff = self.control_module(x)                              # (B, out_channels, E) = (B, 32, 3)
        weight1 = self.weight1.view(self.E, self.out_channels, -1) # E, out_channels, in_channels // groups * k * k
        weight2 = self.weight2.view(self.E, self.out_channels, -1) # E, out_channels, in_channels // groups * k * k
        weight3 = self.weight3.view(self.E, self.out_channels, -1) # E, out_channels, in_channels // groups * k * k
        x = x.view(1, bs * in_channels, h, w) # 1, bs * in_channels, h, w
        
        print(weight1.shape)
        print(weight2.shape)
        print(weight3.shape)        

        aggregate_weight1 = torch.zeros(bs, self.out_channels, self.in_channels // self.groups, self.kernel_size, 
                                        self.kernel_size) # bs, out_channels, in_channels // groups, k, k
        aggregate_weight2 = torch.zeros(bs, self.out_channels, self.out_channels // self.groups, self.kernel_size,
                                        self.kernel_size) # bs, out_channels, in_channels // groups, k, k
        aggregate_weight3 = torch.zeros(bs, self.out_channels, self.out_channels // self.groups, self.kernel_size,
                                        self.kernel_size) # bs, out_channels, in_channels // groups, k, k
        
        print(aggregate_weight1.shape)
        print(aggregate_weight2.shape)
        print(aggregate_weight3.shape)

        if self.bias:
            aggregate_bias1 = torch.zeros(bs, self.out_channels) # bs, out_channels
            aggregate_bias2 = torch.zeros(bs, self.out_channels) # bs, out_channels
            aggregate_bias3 = torch.zeros(bs, self.out_channels) # bs, out_channels
        
        for i in range(self.out_channels):
            sub_coeff = coeff[:, i, :] # bs, E
            sub_weight1 = weight1[:, i, :] # E, in_channels // groups * k * k
            sub_weight2 = weight2[:, i, :] # E, in_channels // groups * k * k
            sub_weight3 = weight3[:, i, :] # E, in_channels // groups * k * k
            
            print(sub_weight1.shape)
            print(sub_weight2.shape)
            print(sub_weight3.shape)

            sub_aggregate_weight1 = torch.mm(sub_coeff, sub_weight1) # bs, in_channels // groups * k * k
            aggregate_weight1[:, i, :, :, :] = sub_aggregate_weight1.view(bs, self.in_channels // self.groups, 
                                                                          self.kernel_size, self.kernel_size)
            
            sub_aggregate_weight2 = torch.mm(sub_coeff, sub_weight2) # bs, in_channels // groups * k * k
            aggregate_weight2[:, i, :, :, :] = sub_aggregate_weight2.view(bs, self.out_channels // self.groups,
                                                                          self.kernel_size, self.kernel_size)
            
            sub_aggregate_weight3 = torch.mm(sub_coeff, sub_weight3) # bs, in_channels // groups * k * k
            aggregate_weight3[:, i, :, :, :] = sub_aggregate_weight3.view(bs, self.out_channels // self.groups,
                                                                          self.kernel_size, self.kernel_size)
            
            print(sub_aggregate_weight1.shape)

            if self.bias:
                aggregate_bias1[:, i] = torch.mm(sub_coeff, self.bias1[:, i].view(self.E, 1)).view(bs)  # bs
                aggregate_bias2[:, i] = torch.mm(sub_coeff, self.bias2[:, i].view(self.E, 1)).view(bs)  # bs
                aggregate_bias3[:, i] = torch.mm(sub_coeff, self.bias3[:, i].view(self.E, 1)).view(bs)  # bs
            
        aggregate_weight1 = aggregate_weight1.view(bs * self.out_channels, self.in_channels // self.groups, 
                                                   self.kernel_size, self.kernel_size)  # 1, bs * out_channels, in_channels // groups, h, w
        aggregate_weight2 = aggregate_weight2.view(bs * self.out_channels, self.out_channels // self.groups,
                                                   self.kernel_size, self.kernel_size)  # bs * out_channels, in_channels // groups, h, w
        aggregate_weight3 = aggregate_weight3.view(bs * self.out_channels, self.out_channels // self.groups,
                                                   self.kernel_size, self.kernel_size)  # bs * out_channels, in_channels // groups, h, w
        print(aggregate_weight1.shape)

        if self.bias:
            aggregate_bias1 = aggregate_bias1.view(bs * self.out_channels) # bs * out_channels
            aggregate_bias2 = aggregate_bias2.view(bs * self.out_channels) # bs * out_channels
            aggregate_bias3 = aggregate_bias3.view(bs * self.out_channels) # bs * out_channels
        else:
            aggregate_bias1, aggregate_bias2, aggregate_bias3 = None, None, None
            
        
        out = F.conv2d(x, weight=aggregate_weight1, bias=aggregate_bias1, stride=self.stride, padding=self.padding, 
                       groups=self.groups * bs)   # bs * out_channels, in_channels // groups, h, w
        print(out.shape)
        out = F.conv2d(out, weight=aggregate_weight2, bias=aggregate_bias2, stride=self.stride, padding=self.padding,
                       groups=self.groups * bs)  # bs * out_channels, in_channels // groups, h, w
        print(out.shape)
        out = F.conv2d(out, weight=aggregate_weight3, bias=aggregate_bias3, stride=self.stride, padding=self.padding,
                       groups=self.groups * bs)  # bs * out_channels, in_channels // groups, h, w
        print(out.shape)
        out = out.view(bs, self.out_channels, out.shape[2], out.shape[3])
        print(out.shape)

        return out
    

def test_control_module():
    x = torch.randn(1, 32, 160, 90)
    net = ControlModule(32, 32)
    print(net(x).shape)


def test_assembled_block():
    x = torch.randn(2, 32, 160, 90)
    net = AssembledBlock(32, 32, 3, 1, 1, groups=1)
    print(net(x).shape)
    
if __name__ == '__main__':
    test_control_module()
    print("------")
    test_assembled_block()
