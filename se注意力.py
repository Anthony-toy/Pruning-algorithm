
import torch
from torch import nn
from torchinfo import summary

class SEAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.input_channels = input_channels
        self.reduction_ratio = reduction_ratio
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // reduction_ratio, input_channels),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        batch_size, channels, _, _ = inputs.size()

        squeezed = self.squeeze(inputs).view(batch_size, channels)
        weights = self.excitation(squeezed).view(batch_size, channels, 1, 1)
        attended_inputs = inputs * weights

        return attended_inputs
data = torch.ones(size=(3,64,10,10))
net = SEAttention(64)
net(data)
summary(net,input_size=(3,64,10,10))

