"""Class-balanced CE plus foreground Dice for thin/rare visible structures."""
import torch
from torch.nn import functional as F
from .segmentation_training import weighted_loss


def foreground_dice(logits,target):
    valid=target!=255;classes=logits.shape[1]
    safe=target.masked_fill(~valid,0)
    truth=F.one_hot(safe,classes).permute(0,3,1,2).float()*valid[:,None]
    probabilities=logits.float().softmax(1)*valid[:,None]
    support=truth.sum((0,2,3));eligible=support>0;eligible[0]=False
    if not bool(eligible.any()):return logits.sum()*0
    intersection=(probabilities*truth).sum((0,2,3))
    dice=(2*intersection+1)/(probabilities.sum((0,2,3))+support+1)
    return 1-dice[eligible].mean()


def segmentation_loss(logits,target,weights):
    return weighted_loss(logits,target,weights)+.5*foreground_dice(logits,target)
