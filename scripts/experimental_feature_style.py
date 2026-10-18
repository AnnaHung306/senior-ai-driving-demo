"""Unintegrated train-only feature-style experiment, not a trained model.

Method reference: Zhou et al., Domain Generalization with MixStyle, ICLR2021.
https://openreview.net/forum?id=6xHJ37MVxxp
https://github.com/KaiyangZhou/mixstyle-release

Independent implementation of feature-statistic interpolation. Population
variance and FP32 statistics are explicit adaptations for BF16 segmentation;
this is not a reproduction of the paper's reported experiments/results.
"""
import math
import torch
from torch import nn


class ExperimentalFeatureStyle(nn.Module):
    def __init__(self,probability=.5,alpha=.1,epsilon=1e-6):
        super().__init__()
        for name,value in [('probability',probability),('alpha',alpha),('epsilon',epsilon)]:
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite scalar required: '+name)
        if not 0<=probability<=1 or alpha<=0 or epsilon<=0:raise ValueError('Invalid feature-style experiment settings')
        self.probability=float(probability);self.alpha=float(alpha);self.epsilon=float(epsilon)

    def forward(self,features):
        if features.ndim!=4 or not features.is_floating_point() or min(features.shape)<=0:
            raise ValueError('Nonempty floating-point NCHW features required')
        count=features.shape[0]
        if not self.training or count==1 or self.probability==0:return features
        # Host RNG is deliberately used: existing epoch checkpoints already
        # capture CPU torch RNG, without an untracked device-local generator.
        if self.probability<1 and float(torch.rand(()))>=self.probability:return features
        values=features.float()
        center=values.mean(dim=(-2,-1),keepdim=True).detach()
        spread=(values.var(dim=(-2,-1),correction=0,keepdim=True)+self.epsilon).sqrt().detach()
        concentration=torch.tensor(self.alpha,dtype=torch.float32)
        amount=torch.distributions.Beta(concentration,concentration).sample((count,1,1,1)).to(features.device)
        other=torch.randperm(count,device='cpu').to(features.device)
        center_new=torch.lerp(center[other],center,amount)
        spread_new=torch.lerp(spread[other],spread,amount)
        result=(values-center)*(spread_new/spread)+center_new
        return result.to(dtype=features.dtype)


def attach_prototype_hooks(model,settings,stages=(3,6)):
    """CPU preflight helper only; formal source/config integration is pending."""
    blocks=[ExperimentalFeatureStyle(**settings) for _ in stages]
    handles=[]
    for index,block in zip(stages,blocks):
        def hook(layer,inputs,output,operation=block):
            operation.train(model.training)
            return operation(output)
        handles.append(model.encoder[index].register_forward_hook(hook))
    return handles
