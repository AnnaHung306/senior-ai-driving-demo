"""TorchVision 0.25 Faster R-CNN with ignored foreground at both samplers.

Targets are training-only. In eval mode the model takes RGB tensors alone.
"""
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RegionProposalNetwork,RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads,fastrcnn_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes
from torchvision.ops import MultiScaleRoIAlign,FeaturePyramidNetwork
from torchvision.ops.poolers import _filter_input, _setup_scales, _convert_to_roi_format
from torchvision.ops.roi_align import _roi_align


class EagerDeterministicRoIAlign(MultiScaleRoIAlign):
    """Use upstream's deterministic differentiable ROIAlign without torch.compile.

    Windows CUDA in this pinned environment has no Triton/Inductor support.
    Fixed sampling_ratio=2 bounds the eager interpolation tensor memory. The
    multi-level dispatch follows TorchVision 0.25 poolers.py (BSD-3-Clause).
    """
    def forward(self,features,boxes,image_shapes):
        # The fused forward is deterministic; only its CUDA backward uses
        # atomic accumulation. Evaluation needs no backward or compilation.
        if not self.training and not torch.is_grad_enabled():
            return super().forward(features,boxes,image_shapes)
        values=_filter_input(features,self.featmap_names)
        if self.scales is None or self.map_levels is None:
            self.scales,self.map_levels=_setup_scales(values,image_shapes,self.canonical_scale,self.canonical_level)
        rois=_convert_to_roi_format(boxes)
        eager=_roi_align.__wrapped__
        if len(values)==1:
            return eager(values[0],rois,self.scales[0],self.output_size[0],self.output_size[1],self.sampling_ratio,False)
        levels=self.map_levels(boxes)
        result=values[0].new_zeros((len(rois),values[0].shape[1],*self.output_size))
        for level,(feature,scale) in enumerate(zip(values,self.scales)):
            indices=torch.where(levels==level)[0]
            if len(indices):
                pooled=eager(feature,rois[indices],scale,self.output_size[0],self.output_size[1],self.sampling_ratio,False)
                result[indices]=pooled.to(result.dtype)
        return result


def ignore_background(boxes, labels, ignored, threshold=.5):
    """Same policy as visibility_policy.ignore_background_proposals, on device."""
    result = labels.clone()
    if not boxes.numel() or not ignored.numel():
        return result
    # Chunking bounds memory for full-resolution RPN anchor grids.
    for start in range(0, len(boxes), 4096):
        b = boxes[start:start + 4096]
        intersection = (torch.minimum(b[:, None, 2:], ignored[None, :, 2:]) -
                        torch.maximum(b[:, None, :2], ignored[None, :, :2])).clamp(min=0).prod(2)
        ioa = intersection / (b[:, 2:] - b[:, :2]).prod(1)[:, None].clamp(min=1e-8)
        center = (b[:, :2] + b[:, 2:]) / 2
        inside = ((center[:, None] >= ignored[None, :, :2]) &
                  (center[:, None] < ignored[None, :, 2:])).all(2)
        rows = result[start:start + len(b)]
        rows[(rows == 0) & ((ioa > threshold) | inside).any(1)] = -1
    return result


class IgnoreTransform(GeneralizedRCNNTransform):
    def resize(self, image, target=None):
        original = image.shape[-2:]
        image, target = super().resize(image, target)
        if target is not None and 'ignore_boxes' in target:
            target['ignore_boxes'] = resize_boxes(target['ignore_boxes'], original, image.shape[-2:])
        return image, target


class IgnoreRPN(RegionProposalNetwork):
    def assign_targets_to_anchors(self, anchors, targets):
        labels, matched = super().assign_targets_to_anchors(anchors, targets)
        labels = [ignore_background(a, l, t['ignore_boxes']) for a, l, t in zip(anchors, labels, targets)]
        self.ignore_assignment_calls += 1
        self.last_supervised_samples=sum(int((l>=0).sum()) for l in labels)
        return labels, matched

    def compute_loss(self,objectness,pred_bbox_deltas,labels,regression_targets):
        if not any(bool((l>=0).any()) for l in labels):
            return objectness.sum()*0,pred_bbox_deltas.sum()*0
        return super().compute_loss(objectness,pred_bbox_deltas,labels,regression_targets)


class IgnoreROI(RoIHeads):
    def select_training_samples(self, proposals, targets):
        # RoIHeads does matching and sampling inside this method. Carry the
        # per-image ignore boxes only through that call; never through inference.
        self._training_ignore = [t['ignore_boxes'] for t in targets]
        try:
            return super().select_training_samples(proposals, targets)
        finally:
            self._training_ignore = None

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched, labels = super().assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        if self._training_ignore is None:
            raise RuntimeError('ROI matching requires explicit per-image ignore context')
        labels = [ignore_background(p, l, i) for p, l, i in zip(proposals, labels, self._training_ignore)]
        self.ignore_assignment_calls += 1
        return matched, labels

    def forward(self,features,proposals,image_shapes,targets=None):
        # Box-only upstream RoIHeads forward, with a differentiable zero for
        # an entirely ignored batch. Cross entropy of zero samples is NaN.
        if self.has_mask() or self.has_keypoint():raise RuntimeError('Box-only ignore head')
        if self.training:
            proposals,matched,labels,regression=self.select_training_samples(proposals,targets)
            self.last_supervised_samples=sum(len(l) for l in labels)
        pooled=self.box_roi_pool(features,proposals,image_shapes)
        logits,boxes=self.box_predictor(self.box_head(pooled))
        if self.training:
            if self.last_supervised_samples:
                classifier,regressor=fastrcnn_loss(logits,boxes,labels,regression)
            else:
                classifier,regressor=logits.sum()*0,boxes.sum()*0
            return [],{'loss_classifier':classifier,'loss_box_reg':regressor}
        boxes,scores,labels=self.postprocess_detections(logits,boxes,proposals,image_shapes)
        return [{'boxes':b,'scores':s,'labels':l} for b,s,l in zip(boxes,scores,labels)],{}


def make_detector(num_classes=9, weights=None):
    """Background plus eight requested object categories. No implicit downloads.

    Optional weights must be a locally verified TorchVision COCO state dict.
    This model is not safe for concurrent training calls on the same instance.
    """
    if not torchvision.__version__.startswith('0.25.0'):
        raise RuntimeError('Pinned TorchVision 0.25.0 required for deterministic ROI internals')
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None,
        min_size=360, max_size=640, rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=500, rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=150, box_detections_per_img=100, box_score_thresh=.01,
        box_batch_size_per_image=128)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    # The stock mobile detector exposes only coarse stride-32 maps. Our minimum
    # visible labels include tiny lamps/obstacles, so expose strides 4/8/16/32.
    # Reuse the same pretrained MobileNet body, highest pyramid projection, RPN
    # convolution and ratio-specific anchor channels; initialize new lower FPN
    # projections before project fine-tuning. No heldout model scores select this.
    old_fpn=model.backbone.fpn
    model.backbone.body.return_layers={'2':'0','4':'1','7':'2','16':'3'}
    model.backbone.fpn=FeaturePyramidNetwork([24,40,80,960],256,extra_blocks=old_fpn.extra_blocks)
    model.backbone.fpn.inner_blocks[3].load_state_dict(old_fpn.inner_blocks[1].state_dict())
    model.backbone.fpn.layer_blocks[3].load_state_dict(old_fpn.layer_blocks[1].state_dict())
    old_head=model.rpn.head;head=RPNHead(256,3);head.conv=old_head.conv
    indices=torch.tensor([0,5,10])  # each ratio's original 32px anchor channel
    with torch.no_grad():
        head.cls_logits.weight.copy_(old_head.cls_logits.weight[indices]);head.cls_logits.bias.copy_(old_head.cls_logits.bias[indices])
        head.bbox_pred.weight.copy_(old_head.bbox_pred.weight.reshape(15,4,256,1,1)[indices].reshape(12,256,1,1))
        head.bbox_pred.bias.copy_(old_head.bbox_pred.bias.reshape(15,4)[indices].reshape(12))
    model.rpn.head=head
    model.rpn.anchor_generator=AnchorGenerator(((16,),(32,),(64,),(128,),(256,)),((.5,1.,2.),)*5)
    model.roi_heads.box_roi_pool.featmap_names=['0','1','2','3']
    model.roi_heads.box_roi_pool.scales=None;model.roi_heads.box_roi_pool.map_levels=None
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes)
    # These subclasses add no parameters, replace no learned layers, and retain
    # the installed upstream matching, loss, and balanced-sampling algorithms.
    model.transform.__class__ = IgnoreTransform
    model.rpn.__class__ = IgnoreRPN
    model.roi_heads.__class__ = IgnoreROI
    model.roi_heads.box_roi_pool.__class__ = EagerDeterministicRoIAlign
    model.rpn.ignore_assignment_calls = model.roi_heads.ignore_assignment_calls = 0
    model.roi_heads._training_ignore = None
    return model
