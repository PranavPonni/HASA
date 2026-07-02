import torch
import torch.nn as nn
import torchvision.transforms as transforms

class EIPLSARNNAugment(nn.Module):
    def __init__(self, device):
        super(EIPLSARNNAugment, self).__init__()
        self.device = device
        self.transform = nn.Sequential(
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(self.device)

    def forward(self, x):
        return self.transform(x)