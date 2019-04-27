import torch.nn as nn
from .human_model_layer import KinematicLayer

class AvgPoolHead(nn.Module):
    def __init__(self, in_channels, out_channels, fea_map_size):
        super(AvgPoolHead, self).__init__()
        self.avgpool = nn.AvgPool2d(fea_map_size, stride=1)
        self.fc = nn.Linear(in_channels, out_channels)
        self.kin_layer = KinematicLayer()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.kin_layer(x)
        return x
