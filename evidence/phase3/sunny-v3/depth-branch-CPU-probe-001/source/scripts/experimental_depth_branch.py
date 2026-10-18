"""Isolated depth auxiliary prototype; not used by production training/inference.

The shared feature is tapped only during this wrapper's call. RGB segmentation
uses the original MobileUNet forward unchanged; depth is a training target only.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def decode_depth_rgb(packed):
    values=np.asarray(packed)
    if values.dtype!=np.uint8 or values.ndim!=3 or values.shape[-1]!=3:
        raise ValueError('Raw packed RGB uint8 depth required')
    values=values.astype(np.uint32)
    return (values[...,0]+256*values[...,1]+65536*values[...,2])*(1000./16777215)


def depth_target(depth_m,raw_semantic,minimum=.1,maximum=120.):
    if not 0<minimum<maximum or not math.isfinite(maximum):raise ValueError('Finite positive depth interval required')
    if depth_m.ndim!=3 or raw_semantic.shape!=depth_m.shape:raise ValueError('Aligned NHW depth/semantic required')
    valid=torch.isfinite(depth_m)&(depth_m>minimum)&(depth_m<maximum)&(raw_semantic!=0)&(raw_semantic!=11)&(raw_semantic!=255)
    safe=torch.where(valid,depth_m,torch.zeros_like(depth_m))
    return torch.log1p(safe)/math.log1p(maximum),valid


def masked_depth_loss(prediction,target,valid):
    if prediction.ndim!=4 or prediction.shape[1]!=1 or target.ndim!=3 or target.shape!=valid.shape or prediction.shape[0]!=target.shape[0]:
        raise ValueError('Depth prediction N1hw and aligned NHW targets required')
    if valid.dtype!=torch.bool or not torch.isfinite(prediction).all():raise ValueError('Finite prediction and boolean validity required')
    if not bool(valid.any()):return prediction.sum()*0
    if not torch.isfinite(target[valid]).all() or bool(((target[valid]<0)|(target[valid]>1)).any()):raise ValueError('Valid normalized log-depth must be in [0,1]')
    full=F.interpolate(prediction.float(),size=target.shape[-2:],mode='bilinear',align_corners=False)[:,0]
    safe=torch.where(valid,target,torch.zeros_like(target))
    pixel=F.smooth_l1_loss(full,safe,reduction='none',beta=.05)
    counts=valid.sum((1,2));per_image=(pixel*valid).sum((1,2))/counts.clamp_min(1)
    return per_image[counts>0].mean()


class ExperimentalDepthBranch(nn.Module):
    """Serial-call feasibility wrapper, RGB input only; no production integration."""
    def __init__(self,segmentation,seed=20261018):
        super().__init__();self.segmentation=segmentation
        # Adding the head does not consume the segmentation/augmentation RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.depth_head=nn.Sequential(nn.Conv2d(48,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1),nn.Sigmoid())

    def forward(self,rgb,with_depth=True):
        if not with_depth:return self.segmentation(rgb)
        features=[]
        handle=self.segmentation.dec4.register_forward_hook(lambda module,inputs,output:features.append(output))
        try:logits=self.segmentation(rgb)
        finally:handle.remove()
        if len(features)!=1:raise RuntimeError('Exactly one stride4 shared feature required')
        # Production BF16 forward owns its autocast context; this tiny head is FP32.
        return logits,self.depth_head(features[0].float())
