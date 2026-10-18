"""Training-only paired zoom-out/flip candidate; not used by the active v5 run.

Returns only the three fields consumed by the box detector losses. Original
visible-pixel counts, clipping flags and phase labels are not measurements of
the augmented image and are deliberately not passed off as such.
"""
import random
import torch
import torch.nn.functional as F


def transform_one(rgb, target, scale=1., flip=False, offset=None):
    if rgb.ndim!=3 or rgb.shape[0]!=3 or not rgb.is_floating_point():
        raise ValueError('Floating CHW RGB required')
    if not .25<=scale<=1.: raise ValueError('Zoom-out scale must be .25..1')
    if not {'boxes','labels','ignore_boxes'}.issubset(target): raise ValueError('Boxes, labels and ignores required')
    if 'masks' in target or 'keypoints' in target: raise ValueError('Box-only training transform')
    h,w=rgb.shape[-2:];nh,nw=max(1,round(h*scale)),max(1,round(w*scale))
    top,left=offset if offset is not None else (random.randint(0,h-nh),random.randint(0,w-nw))
    if type(top)is not int or type(left)is not int or not 0<=top<=h-nh or not 0<=left<=w-nw:
        raise ValueError('Paste offset outside original canvas')
    boxes=target['boxes'];ignored=target['ignore_boxes'];labels=target['labels']
    if boxes.shape!=(len(labels),4) or ignored.ndim!=2 or ignored.shape[1]!=4:
        raise ValueError('Target matrix shape differs')
    for values in [boxes,ignored]:
        if not torch.isfinite(values).all() or values.numel() and (
            (values[:,2:]<=values[:,:2]).any() or (values<0).any() or
            (values[:,[0,2]]>w).any() or (values[:,[1,3]]>h).any()):
            raise ValueError('Target outside source image')
    if (nh,nw)==(h,w):out=rgb.clone()
    else:
        fill=rgb.new_tensor([.485,.456,.406])[:,None,None]
        out=fill.expand_as(rgb).clone()
        out[:,top:top+nh,left:left+nw]=F.interpolate(rgb[None],size=(nh,nw),mode='bilinear',align_corners=False)[0]
    def move(values):
        factor=values.new_tensor([nw/w,nh/h,nw/w,nh/h]);delta=values.new_tensor([left,top,left,top])
        return values*factor+delta
    boxes,ignored=move(boxes),move(ignored)
    padding=[]
    if top:padding.append([0,0,w,top])
    if top+nh<h:padding.append([0,top+nh,w,h])
    if left:padding.append([0,top,left,top+nh])
    if left+nw<w:padding.append([left+nw,top,w,top+nh])
    if padding:ignored=torch.cat([ignored,ignored.new_tensor(padding)],0)
    if flip:
        out=out.flip(-1)
        for values in [boxes,ignored]:
            x1,x2=values[:,0].clone(),values[:,2].clone()
            values[:,0],values[:,2]=w-x2,w-x1
    return out,{'boxes':boxes,'labels':labels.clone(),'ignore_boxes':ignored}, {
        'scale_xy':[nw/w,nh/h],'offset_top_left':[top,left], 'horizontal_flip':flip,
        'padding_ignored':True,'scope':'Training augmentation; raw provenance and eval targets unchanged'}


def paired_batch(rgb, targets, scales=(.5,.75,1.), flip_probability=.5):
    if rgb.ndim!=4 or len(rgb)!=len(targets):raise ValueError('BCHW RGB/target batch required')
    if not scales or any(not .25<=s<=1. for s in scales) or not 0<=flip_probability<=1:
        raise ValueError('Invalid frozen augmentation policy')
    images=[];truth=[];ledger=[]
    for image,target in zip(rgb,targets):
        out,labels,record=transform_one(image,target,random.choice(scales),random.random()<flip_probability)
        images.append(out);truth.append(labels);ledger.append(record)
    return torch.stack(images),truth,ledger
