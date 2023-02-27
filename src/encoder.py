import torch

from torch import nn
from torchvision import models, transforms


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.ResNet152_Weights.DEFAULT
        self.resnet_model = models.resnet152(weights=weights)
        self.encoder_layers = torch.nn.Sequential(*(list(self.resnet_model.children())[:-4]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_layers(x)
