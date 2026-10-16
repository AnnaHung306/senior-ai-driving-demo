"""COCO MobileNet encoder with a trainable multiscale semantic decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.ops.misc import FrozenBatchNorm2d
from .segmentation_model import Block


class MobileUNet(nn.Module):
    def __init__(self,classes=17,precision='bfloat16'):
        super().__init__();self.precision=precision
        if precision not in ['float32','bfloat16']:raise ValueError('Unsupported semantic precision')
        self.encoder=mobilenet_v3_large(weights=None,norm_layer=FrozenBatchNorm2d).features
        self.project=nn.Conv2d(960,128,1)
        self.dec16=Block(128+112,96);self.dec8=Block(96+40,64)
        self.dec4=Block(64+24,48);self.dec2=Block(48+16,32)
        self.stem=Block(3,16);self.dec1=Block(32+16,32);self.head=nn.Conv2d(32,classes,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def load_encoder(self,path):
        weights=torch.load(path,map_location='cpu',weights_only=True)
        body={k.removeprefix('backbone.body.'):v for k,v in weights.items() if k.startswith('backbone.body.')}
        # Exact same MobileNet topology; no permissive missing-key load.
        self.encoder.load_state_dict(body,strict=True)

    def forward(self,rgb):
        enabled=rgb.device.type=='cuda' and self.precision=='bfloat16'
        with torch.autocast(device_type=rgb.device.type,dtype=torch.bfloat16,enabled=enabled):
            image=(rgb-self.mean)/self.std;x=image;skips={}
            for i,layer in enumerate(self.encoder):
                x=layer(x)
                if i in [1,3,6,12]:skips[i]=x
            x=self.project(x)
            for i,block in [(12,self.dec16),(6,self.dec8),(3,self.dec4),(1,self.dec2)]:
                skip=skips[i];x=block(torch.cat((F.interpolate(x,size=skip.shape[-2:],mode='nearest'),skip),1))
            x=F.interpolate(x,size=rgb.shape[-2:],mode='nearest');x=self.dec1(torch.cat((x,self.stem(image)),1))
            result=self.head(x)
        # Keep losses, softmax and metrics in float32; parameters/optimizer too.
        return result.float()
