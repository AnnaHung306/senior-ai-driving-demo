"""CPU construction prototype for an extra stride2 detector pyramid level.

Not integrated in production config/checkpoint loaders; no trained P1 model.
"""
import torch
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.anchor_utils import AnchorGenerator


def add_p1_cpu_prototype(parent):
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Construct this prototype on CPU before any future CUDA preflight')
    expected={'2':'0','4':'1','7':'2','16':'3'}
    if dict(parent.backbone.body.return_layers)!=expected or len(parent.backbone.fpn.inner_blocks)!=4:
        raise ValueError('Only the current four-level project detector can initialize this prototype')
    if parent.rpn.anchor_generator.num_anchors_per_location()!=[3]*5:raise ValueError('Expected original three aspect-ratio channels')
    previous=parent.backbone.fpn
    expanded=FeaturePyramidNetwork([16,24,40,80,960],256,extra_blocks=previous.extra_blocks)
    for old_index in range(4):
        expanded.inner_blocks[old_index+1].load_state_dict(previous.inner_blocks[old_index].state_dict(),strict=True)
        expanded.layer_blocks[old_index+1].load_state_dict(previous.layer_blocks[old_index].state_dict(),strict=True)
    parent.backbone.body.return_layers={'1':'0','2':'1','4':'2','7':'3','16':'4'}
    parent.backbone.fpn=expanded
    parent.rpn.anchor_generator=AnchorGenerator(((8,),(16,),(32,),(64,),(128,),(256,)),((.5,1.,2.),)*6)
    parent.roi_heads.box_roi_pool.featmap_names=['0','1','2','3','4']
    parent.roi_heads.box_roi_pool.scales=None;parent.roi_heads.box_roi_pool.map_levels=None
    return parent
