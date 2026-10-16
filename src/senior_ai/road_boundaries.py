"""Visible road-to-sidewalk/terrain interface; not hidden curb or free space."""
import numpy as np


def visible_margin(classes):
    """Input uses sunny model IDs, never raw CARLA IDs (raw10 is Terrain)."""
    road=np.isin(classes,[1,10]);side=np.isin(classes,[2,16]);result=np.zeros_like(road)
    result[1:] |= road[1:]&side[:-1];result[:-1] |= road[:-1]&side[1:]
    result[:,1:] |= road[:,1:]&side[:,:-1];result[:,:-1] |= road[:,:-1]&side[:,1:]
    return result


def dilate(mask,radius=2):
    result=np.zeros_like(mask);padded=np.pad(mask,radius)
    for y in range(2*radius+1):
        for x in range(2*radius+1):result |= padded[y:y+mask.shape[0],x:x+mask.shape[1]]
    return result


def boundary_counts(target,prediction):
    valid=target!=255;supervised=valid.copy()
    supervised[1:] &= valid[:-1];supervised[:-1] &= valid[1:]
    supervised[:,1:] &= valid[:,:-1];supervised[:,:-1] &= valid[:,1:]
    truth=visible_margin(target)&supervised;predicted=visible_margin(prediction)&supervised
    return {'gt_pixels':int(truth.sum()),'predicted_pixels':int(predicted.sum()),
            'matched_gt_within_2px':int((truth&dilate(predicted)).sum()),
            'matched_prediction_within_2px':int((predicted&dilate(truth)).sum())}


def boundary_metrics(counts):
    p=counts['matched_prediction_within_2px']/counts['predicted_pixels'] if counts['predicted_pixels'] else None
    r=counts['matched_gt_within_2px']/counts['gt_pixels'] if counts['gt_pixels'] else None
    f=2*(p or 0)*(r or 0)/((p or 0)+(r or 0)) if (p or 0)+(r or 0)>0 else (0. if counts['gt_pixels'] or counts['predicted_pixels'] else None)
    return {**counts,'precision':p,'recall':r,'f1':f,'tolerance_pixels':2,
            'definition':'Visible road/road-line side adjacent to sidewalk or terrain/ground; no inferred boundary behind occluders and no drivable-free-space claim'}
