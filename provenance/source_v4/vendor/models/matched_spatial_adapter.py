"""Parameter-matched spatial residual control for the 16-channel DTCWT adapter."""
import torch
from torch import nn
from baseline_code.nets.backbone import CIF

class SpatialResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(512, 16, 1)
        self.fuse = nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False), nn.GELU(), nn.Conv2d(16,16,3,padding=1), nn.GELU())
        self.expand = nn.Conv2d(16,512,1)
        self.alpha = nn.Parameter(torch.zeros(()))
    def forward(self, x):
        return x + self.alpha.to(dtype=x.dtype)*self.expand(self.fuse(self.reduce(x)))

class SpatialCIF(CIF):
    def __init__(self,num_classes,last_inp_channels):
        super().__init__(num_classes=num_classes,last_inp_channels=last_inp_channels)
        self.dtcwt_adapter=SpatialResidualAdapter()
    def forward(self,x,x_1):
        mask_1=self.attention_1(x_1); mask=self.attention(x)
        element=self.last_layer_e(x+x_1*mask)
        defect=self.last_layer_d(self.dtcwt_adapter(x_1+x*mask_1))
        return element,defect
