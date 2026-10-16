"""Small RGB-crop phase classifier; no simulator or GT in forward."""
import numpy as np
from PIL import Image
import torch
from torch import nn


class LightStateNet(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]; channels=3
        for width in [8,16,32,64]:
            layers.extend([nn.Conv2d(channels,width,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,width),nn.ReLU()]);channels=width
        self.features=nn.Sequential(*layers);self.classifier=nn.Linear(64,4)

    def forward(self,rgb_crop):
        return self.classifier(self.features(rgb_crop).mean((-2,-1)))


def crop_rgb(rgb,box,width=32,height=64):
    h,w=rgb.shape[:2];x0,y0,x1,y1=box
    bounds=(max(0,int(np.floor(x0))),max(0,int(np.floor(y0))),min(w,int(np.ceil(x1))),min(h,int(np.ceil(y1))))
    if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]: raise ValueError('Empty lamp crop')
    image=Image.fromarray(rgb).crop(bounds).resize((width,height),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image).transpose(2,0,1).copy())
