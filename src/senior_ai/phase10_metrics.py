"""Pure evaluation helpers for fresh Phase10 observations; never imported by planner."""
import numpy as np
from .instance_labels import decode_instance_rgb,mask_bbox

RAW_TO_SEG={1:1,2:2,12:3,13:4,14:5,15:6,16:7,18:8,19:9,24:10,7:11,4:12,5:13,6:14,10:16,25:16}
TAG_TO_DET={14:1,12:2,19:3,18:4,15:5,16:6,7:7}


def semantic_confusion(raw,pred):
    if raw.shape!=pred.shape or raw.dtype!=np.uint8 or pred.max()>16 or raw.max()>28:raise ValueError('Bad semantic arrays')
    lut=np.zeros(29,np.uint8)
    for key,value in RAW_TO_SEG.items():lut[key]=value
    lut[[0,21]]=255;truth=lut[raw];valid=truth!=255
    matrix=np.bincount((truth[valid].astype(np.int64)*17+pred[valid]),minlength=289).reshape(17,17)
    return matrix,int((~valid).sum())


def visible_targets(instance,actor_ids,min_pixels=16):
    tags,ids=decode_instance_rgb(instance);targets=[];ignored=[];ambiguous=[]
    for identifier in np.unique(ids[np.isin(tags,list(TAG_TO_DET)+[13])]):
        if not identifier:continue
        mask=ids==identifier;values,counts=np.unique(tags[mask],return_counts=True)
        known=[(TAG_TO_DET[int(tag)],int(count)) for tag,count in zip(values,counts) if int(tag) in TAG_TO_DET]
        if not known:
            if 13 in values:
                box=mask_bbox(mask&(tags==13))
                if box:ignored.append(box);ambiguous.append({'instance':int(identifier),'reason':'rider_only_without_identifiable_body','pixels':int((mask&(tags==13)).sum())})
            continue
        label=max(known,key=lambda x:x[1])[0]
        wanted=[k for k,v in TAG_TO_DET.items() if v==label]
        if label in [3,4]:wanted+=[13]
        visible=mask&np.isin(tags,wanted);box=mask_bbox(visible);pixels=int(visible.sum())
        if box is None:continue
        # An untracked actor mask may be scenery. Keep it explicit, never assume negative.
        recognized=int(identifier) in actor_ids or label==7
        reason=None if recognized else 'instance_not_in_capture_actor_snapshot'
        if label in [3,4] and int((mask&np.isin(tags,[18,19])).sum())<8:reason='two_wheeler_body_too_small'
        if pixels<min_pixels or min(box[2]-box[0],box[3]-box[1])<4:reason='tiny_visible_instance'
        if reason:ignored.append(box);ambiguous.append({'instance':int(identifier),'label':label,'reason':reason,'pixels':pixels});continue
        targets.append({'box':box,'label':label,'actor_id':int(identifier) if label!=7 else None,
            'visible_pixels':pixels,'truncated':box[0]==0 or box[1]==0 or box[2]==instance.shape[1] or box[3]==instance.shape[0],
            'GT_source':'visible_raw_instance_mask','rendered_instance_id':int(identifier)})
    return targets,ignored,ambiguous


def summary(values):
    return {'count':len(values),'mean':float(np.mean(values)) if values else None,
            **{name:float(np.percentile(values,q)) if values else None for name,q in [('p50',50),('p95',95),('max',100)]}}


def binary_iou(a,b):
    union=int((a|b).sum());return int((a&b).sum())/union if union else None


def calibration_bins(rows):
    # Score reliability uses greedy per-image, class-aware IoU50 labels upstream.
    bins=[]
    for value in range(5,10):
        lo=value/10;hi=(value+1)/10 if value<9 else 1.000001
        selected=[r for r in rows if lo<=r['score']<hi]
        bins.append({'low':round(float(lo),1),'count':len(selected),
            'mean_score':float(np.mean([r['score'] for r in selected])) if selected else None,
            'empirical_precision':float(np.mean([r['correct'] for r in selected])) if selected else None})
    return bins
