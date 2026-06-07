# models.py
# Model definitions for FSL-based HSI grain classification.
#
# Backbone interface convention used throughout this project:
#   backbone.forward(x) -> (features, attention_weights_or_None)
#
# This ensures PrototypicalNetwork works with both backbone types
# without requiring separate forward methods.
#
# Usage:
#   from models import ResNet18Backbone, ResNet18BackboneWithSE, PrototypicalNetwork
#   backbone = ResNet18BackboneWithSE(in_channels=204, reduction_ratio=8)
#   model = PrototypicalNetwork(backbone)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SEBlock(nn.Module):
    # Squeeze-and-Excitation block with modified squeeze operation.
    # Instead of using only average pooling (original SE paper), we average
    # the outputs of adaptive average pooling and adaptive max pooling.
    # This better captures heterogeneous activation patterns in hyperspectral data.
    # See eq. (6) in the paper.
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        squeeze = (avg + mx) / 2.0
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        scale = x * excitation.view(b, c, 1, 1)
        return scale, excitation.view(b, c, 1, 1)


class ResNet18Backbone(nn.Module):
    # ResNet18 with a 1x1 conv spectral downsampling layer prepended.
    # The 1x1 conv maps the hyperspectral channels (e.g. 204) down to 3
    # channels so the pre-trained RGB ResNet18 can be used as-is.
    # Input format: (batch, H, W, C) - channels-last, as loaded from CDF.
    def __init__(self, in_channels=204):
        super().__init__()
        self.spectral_down = nn.Conv2d(in_channels, 3, kernel_size=1)
        net = resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Flatten()
        self.net = net

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # (B, H, W, C) -> (B, C, H, W)
        x = self.spectral_down(x)
        return self.net(x), None     # (features, attention=None)


class ResNet18BackboneWithSE(nn.Module):
    # ResNet18 with SE channel attention applied before spectral downsampling.
    # Order: SE block -> 1x1 spectral down -> ResNet18.
    # The SE block learns which spectral bands are most informative for
    # grain classification and weights them accordingly.
    # Input format: (batch, H, W, C) - channels-last.
    def __init__(self, in_channels=204, reduction_ratio=8):
        super().__init__()
        self.se = SEBlock(in_channels, reduction_ratio)
        self.spectral_down = nn.Conv2d(in_channels, 3, kernel_size=1)
        net = resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Flatten()
        self.net = net

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # (B, H, W, C) -> (B, C, H, W)
        x, attn = self.se(x)
        x = self.spectral_down(x)
        return self.net(x), attn     # (features, attention_weights)


class PrototypicalNetwork(nn.Module):
    # Prototypical network adapted for HSI grain classification.
    # During forward:
    #   1. Embeds support images -> computes per-class prototypes (mean embedding)
    #   2. Embeds query images
    #   3. Returns negative Euclidean distances as classification scores
    #
    # self.prototypes is stored after each forward pass so it can be accessed
    # externally for CCP computation during and after training.
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.prototypes = None

    def forward(self, support_images, support_labels, query_images):
        z_support, _ = self.backbone(support_images)
        z_query, attn = self.backbone(query_images)

        n_way = len(torch.unique(support_labels))
        self.prototypes = torch.stack([
            z_support[support_labels == k].mean(0)
            for k in range(n_way)
        ])

        dists = torch.cdist(z_query, self.prototypes)
        return -dists, attn
