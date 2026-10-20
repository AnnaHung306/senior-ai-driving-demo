"""Causal RGB-ground visual motion and at-most-three-frame road evidence."""
from collections import deque
import numpy as np
import cv2
from .bev_geometry import warp_metric
from .bev_fusion import GROUND_IDS

def fit_se2(previous,current,minimum=12,threshold=.2):
    a=np.asarray(previous,dtype=np.float64).reshape(-1,2);b=np.asarray(current,dtype=np.float64).reshape(-1,2)
    if len(a)!=len(b) or len(a)<minimum or not np.isfinite(a).all() or not np.isfinite(b).all():return None
    cv2.setRNGSeed(20261020)
    affine,mask=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=threshold,maxIters=1500,confidence=.99)
    if affine is None:return None
    ids=mask.ravel().astype(bool)
    if ids.sum()<minimum or ids.mean()<.5:return None
    scale=np.linalg.norm(affine[:,0])
    if abs(scale-1)>.05:return None
    aa=a[ids];bb=b[ids];ca=aa.mean(0);cb=bb.mean(0);u,_,vt=np.linalg.svd((aa-ca).T@(bb-cb));rotation=vt.T@u.T
    if np.linalg.det(rotation)<0:vt[-1]*=-1;rotation=vt.T@u.T
    t=np.eye(3);t[:2,:2]=rotation;t[:2,2]=cb-rotation@ca
    residual=np.linalg.norm(a@rotation.T+t[:2,2]-b,axis=1);inliers=residual<=threshold
    if inliers.sum()<minimum or inliers.mean()<.5:return None
    return {'previous_to_current':t,'correspondences':len(a),'inliers':int(inliers.sum()),'median_residual_m':float(np.median(residual[inliers]))}

def rgb_ground_motion(previous,current,cameras,config,dt):
    before=[];after=[]
    for name,old in previous.items():
        if name not in current:continue
        new=current[name];a=cv2.cvtColor(old['rgb'],cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(new['rgb'],cv2.COLOR_RGB2GRAY)
        mask=(np.isin(old['ids'],[1,10])*255).astype(np.uint8);mask=cv2.erode(mask,np.ones((5,5),np.uint8))
        p=cv2.goodFeaturesToTrack(a,maxCorners=150,qualityLevel=.015,minDistance=7,mask=mask)
        if p is None:continue
        q,status,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(21,21),maxLevel=3)
        if q is None:continue
        back,reverse,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(21,21),maxLevel=3)
        if back is None:continue
        p=p[:,0];q=q[:,0];good=status.ravel().astype(bool)&reverse.ravel().astype(bool)&(np.linalg.norm(back[:,0]-p,axis=1)<1)
        valid_uv=np.isfinite(q).all(1)&(q[:,0]>=1)&(q[:,0]<639)&(q[:,1]>=1)&(q[:,1]<359)
        indices=np.where(valid_uv)[0];ground=np.zeros(len(q),bool);ground[indices]=np.isin(new['ids'][q[indices,1].astype(int),q[indices,0].astype(int)],GROUND_IDS)
        aa,va=cameras[name].ground(p+.5,config['ground_z_m'],25);bb,vb=cameras[name].ground(q+.5,config['ground_z_m'],25);good&=va&vb&ground
        before.extend(aa[good,:2]);after.extend(bb[good,:2])
    value=fit_se2(before,after,config['temporal']['minimum_inliers'],config['temporal']['maximum_residual_m'])
    if value is None:return None
    t=value['previous_to_current'];angle=abs(np.arctan2(t[1,0],t[0,0]))
    if np.linalg.norm(t[:2,2])>15*dt+.3 or angle>np.deg2rad(40)*dt+.02:return None
    return value

class CausalRoadFusion:
    def __init__(self,grid,config):self.grid=grid;self.config=config;self.history=deque(maxlen=2);self.previous=None

    def update(self,current,observations,cameras,scene,timestamp,calibration_sha):
        motion=None;reason='cold_start'
        if self.previous is not None:
            old=self.previous;dt=timestamp-old['timestamp']
            if old['scene']!=scene or old['calibration']!=calibration_sha or set(old['observations'])!=set(observations):reason='stream_identity_changed'
            elif dt<=0 or dt>self.config['temporal']['maximum_gap_s']:reason='timestamp_gap_or_reversal'
            else:
                motion=rgb_ground_motion(old['observations'],observations,cameras,self.config,dt);reason='aligned' if motion else 'insufficient_visual_motion'
        if motion is None:self.history.clear()
        else:
            for entry in self.history:entry['to_current']=motion['previous_to_current']@entry['to_current']
        probabilities=current['probabilities'].copy();weight=np.ones(self.grid.shape,np.float32);used=0
        for lag,entry in enumerate(reversed(self.history),1):
            valid=warp_metric(entry['observed'].astype(np.uint8),entry['to_current'],self.grid,True).astype(bool)&current['observed']
            w=self.config['temporal']['history_weight']**lag*valid
            probabilities+=warp_metric(entry['probabilities'],entry['to_current'],self.grid)*w[...,None];weight+=w;used+=1
        probabilities/=weight[...,None];probabilities[~current['observed']]=0
        self.history.append({'probabilities':current['probabilities'].copy(),'observed':current['observed'].copy(),'to_current':np.eye(3)})
        self.previous={'observations':observations,'scene':scene,'timestamp':timestamp,'calibration':calibration_sha}
        info={'status':reason,'history_frames_used':used,'motion_source':'RGB_ground_features_only'}
        if motion:info.update({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in motion.items()})
        return {**current,'probabilities':probabilities},info
