"""Paired train-only semantic geometry; no camera/world GT enters the network."""
import random
import torch
from torch.nn import functional as F


def paired_geometry(images,targets,scales=(.75,1.,1.25),flip_probability=.5):
    """Same resize/crop/flip for RGB and IDs, nearest labels, ignore padding."""
    height,width=images.shape[-2:];result=[];labels=[]
    for image,target in zip(images,targets):
        scale=random.choice(scales);h,w=round(height*scale),round(width*scale)
        rgb=F.interpolate(image[None],size=(h,w),mode='bilinear',align_corners=False)[0]
        ids=F.interpolate(target[None,None].float(),size=(h,w),mode='nearest')[0,0].long()
        if h<height or w<width:
            top=random.randrange(max(0,height-h)+1);left=random.randrange(max(0,width-w)+1)
            bottom=max(0,height-h)-top;right=max(0,width-w)-left
            rgb=F.pad(rgb,(left,right,top,bottom),value=0);ids=F.pad(ids,(left,right,top,bottom),value=255)
        h,w=ids.shape;top=random.randrange(h-height+1);left=random.randrange(w-width+1)
        rgb=rgb[:,top:top+height,left:left+width];ids=ids[top:top+height,left:left+width]
        if random.random()<flip_probability:rgb=rgb.flip(-1);ids=ids.flip(-1)
        result.append(rgb);labels.append(ids)
    return torch.stack(result),torch.stack(labels)
