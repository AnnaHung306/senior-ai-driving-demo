"""Depth targets and exact paired geometry for a bounded research A/B only."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,random
import numpy as np
import torch
from torch.nn import functional as F
from experimental_depth_branch import decode_depth_rgb,depth_target
from senior_ai.sunny_data import png,sha
from senior_ai.dataset import safe_path


def paired_depth_geometry(images,targets,depth,valid,scales=(.75,1.,1.25),flip_probability=.5):
    """Same draw order/math as production paired_geometry, plus nearest depth.

    Depth is normalized log-depth, not packed RGB. Metric values do not change
    with image scale. Padding is semantic255, depth0 and invalid.
    """
    if images.ndim!=4 or targets.shape!=depth.shape or valid.shape!=depth.shape or images.shape[0]!=len(targets) or images.shape[-2:]!=targets.shape[-2:]:
        raise ValueError('Aligned RGB/semantic/depth/valid batch required')
    if valid.dtype!=torch.bool:raise ValueError('Boolean depth validity required')
    height,width=images.shape[-2:];rgbs=[];labels=[];depths=[];masks=[];ledger=[]
    for image,target,z,mask in zip(images,targets,depth,valid):
        scale=random.choice(scales);h,w=round(height*scale),round(width*scale)
        rgb=F.interpolate(image[None],size=(h,w),mode='bilinear',align_corners=False)[0]
        ids=F.interpolate(target[None,None].float(),size=(h,w),mode='nearest')[0,0].long()
        dz=F.interpolate(z[None,None],size=(h,w),mode='nearest')[0,0]
        vm=F.interpolate(mask[None,None].float(),size=(h,w),mode='nearest')[0,0].bool()
        padding=[0,0,0,0]
        if h<height or w<width:
            top=random.randrange(max(0,height-h)+1);left=random.randrange(max(0,width-w)+1)
            bottom=max(0,height-h)-top;right=max(0,width-w)-left;padding=[left,right,top,bottom]
            rgb=F.pad(rgb,padding,value=0);ids=F.pad(ids,padding,value=255)
            dz=F.pad(dz,padding,value=0);vm=F.pad(vm,padding,value=False)
        h,w=ids.shape;top=random.randrange(h-height+1);left=random.randrange(w-width+1)
        rgb=rgb[:,top:top+height,left:left+width];ids=ids[top:top+height,left:left+width]
        dz=dz[top:top+height,left:left+width];vm=vm[top:top+height,left:left+width]
        flipped=random.random()<flip_probability
        if flipped:rgb=rgb.flip(-1);ids=ids.flip(-1);dz=dz.flip(-1);vm=vm.flip(-1)
        rgbs.append(rgb);labels.append(ids);depths.append(dz);masks.append(vm)
        ledger.append({'scale':scale,'padding_lrtb':padding,'crop_top_left':[top,left],'flipped':flipped})
    return torch.stack(rgbs),torch.stack(labels),torch.stack(depths),torch.stack(masks),ledger


def nearest_depth_loss(prediction,target,valid):
    """Deterministic nearest upsampling: no CUDA bilinear backward dependency."""
    if prediction.ndim!=4 or prediction.shape[1]!=1 or target.ndim!=3 or target.shape!=valid.shape or len(prediction)!=len(target) or valid.dtype!=torch.bool:
        raise ValueError('Depth N1hw and aligned NHW targets required')
    if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite predicted depth')
    if not bool(valid.any()):return prediction.sum()*0
    if not torch.isfinite(target[valid]).all():raise FloatingPointError('Nonfinite valid target')
    full=F.interpolate(prediction.float(),size=target.shape[-2:],mode='nearest')[:,0]
    safe=torch.where(valid,target,torch.zeros_like(target))
    pixel=F.smooth_l1_loss(full,safe,reduction='none',beta=.05);count=valid.sum((1,2))
    per_image=(pixel*valid).sum((1,2))/count.clamp_min(1)
    return per_image[count>0].mean()


class DepthStream:
    """SHA-check on each cache miss; max128 float/mask entries (~141MiB)."""
    def __init__(self,main_root,config,capacity=128,minimum=.1,maximum=120.):
        if type(capacity)!=int or not 1<=capacity<=128:raise ValueError('Explicit bounded depth cache1..128')
        self.main_root=Path(main_root).resolve();self.capacity=capacity;self.minimum=minimum;self.maximum=maximum;self.cache=OrderedDict();self.inventories={}
        self.inventories[str(self.main_root)]={r['path']:r['sha256'] for r in map(json.loads,(self.main_root/'image_inventory.jsonl').read_text(encoding='utf-8').splitlines())}
        for entry in [config['segmentation'].get('training_scenery'),*config['segmentation'].get('training_expansions',[])]:
            if not entry:continue
            root=Path(entry['root']);root=(self.main_root.parents[1]/root).resolve() if not root.is_absolute() else root.resolve()
            self.inventories[str(root)]={r['path']:r['sha256'] for r in json.loads((root/'audit.json').read_text(encoding='utf-8'))['files']}
        self.misses=0;self.verified=set()

    def get(self,row):
        if row['split']!='train':raise ValueError('Auxiliary depth loader is training-only')
        root=Path(row.get('data_root',self.main_root)).resolve()
        if str(root) not in self.inventories:raise ValueError('Unaudited auxiliary root')
        relative=row.get('depth',row['semantic'].replace('labels/semantic_segmentation/','labels/depth/'))
        key=(str(root),relative,row['semantic'])
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        arrays=[]
        for rel in [relative,row['semantic']]:
            path=safe_path(root,rel)
            if sha(path)!=self.inventories[str(root)][rel]:raise ValueError('Auxiliary bytes differ from audited inventory: '+rel)
            image=png(path)
            if image.shape!=(360,640,3):raise ValueError('Original aligned rig required')
            arrays.append(image)
        z=torch.from_numpy(decode_depth_rgb(arrays[0]).astype(np.float32))[None]
        target,valid=depth_target(z,torch.from_numpy(arrays[1][...,0].copy())[None],self.minimum,self.maximum)
        result=(target[0],valid[0]);self.cache[key]=result;self.misses+=1;self.verified.add(key)
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return result


def nonwall_iou(result):
    values=[r['iou'] for r in result['classes'] if r['name'] not in {'other_annotated','wall'} and r['support_pixels']>0]
    if len(values)!=15:raise ValueError('All15 non-Wall focus classes required')
    return float(np.mean(values))


def update_augmentation_digest(hasher,indices,images,targets,ledger):
    hasher.update(json.dumps({'indices':[int(i) for i in indices],'geometry':ledger},sort_keys=True).encode())
    for tensor in [images,targets]:
        values=tensor.detach().contiguous().cpu().numpy();hasher.update(str(values.dtype).encode());hasher.update(values.tobytes())
