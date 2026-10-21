"""Finite evidence gates, polyline geometry and deliberate-wait progress accounting."""
import numpy as np


def validate_evidence(route,observation):
    path=np.asarray(route['local_xy_m'],float)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():raise ValueError('Invalid Nx2 local path')
    values=np.array([route[k] for k in ['remaining_m','cross_track_m','goal_distance_m']],float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Nonfinite or negative route evidence')
    if observation.get('valid'):
        if not np.isfinite(float(observation.get('timestamp_s',float('nan')))):raise ValueError('Nonfinite image timestamp')
        for value in observation.get('candidate_road_fractions',{}).values():
            if not np.isfinite(float(value)) or not 0<=float(value)<=1:raise ValueError('Invalid road support fraction')
        for signal in observation.get('signals',[]):
            score=float(signal['score'])
            if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid signal confidence')


def origin_path(path):
    p=np.asarray(path,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Invalid path')
    p=np.vstack([[0.,0.],p]);keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-8]
    p=p[keep]
    if len(p)<2:raise ValueError('Degenerate path')
    return p


def project_to_path(point,path):
    p=origin_path(path);q=np.asarray(point,float)
    if q.shape!=(2,) or not np.isfinite(q).all():raise ValueError('Invalid point')
    delta=np.diff(p,axis=0);lengths=np.linalg.norm(delta,axis=1)
    t=np.clip(((q-p[:-1])*delta).sum(1)/(lengths*lengths),0,1)
    distance=np.linalg.norm(p[:-1]+t[:,None]*delta-q,axis=1);index=int(distance.argmin())
    along=np.r_[0.,np.cumsum(lengths)][index]+t[index]*lengths[index]
    return float(along),float(distance[index])


def path_positions(path,distances):
    p=origin_path(path);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    d=np.asarray(distances,float)
    if not np.isfinite(d).all() or (d<0).any():raise ValueError('Invalid future arc distances')
    return np.stack([np.interp(d,arc,p[:,i]) for i in [0,1]],axis=-1)


class ProgressGuard:
    def __init__(self):self.best=0.;self.stall_s=0.;self.blocked_s=0.
    def step(self,progress,speed,target,state,dt):
        if not np.isfinite([progress,speed,target,dt]).all() or dt<=0:raise ValueError('Invalid progress guard evidence')
        if state=='SIGNAL_STOP':
            self.stall_s=self.blocked_s=0.;self.best=max(self.best,progress)
            return False
        self.blocked_s=self.blocked_s+dt if speed<.1 and target==0 else 0.
        if progress>self.best+.15:self.best=progress;self.stall_s=0.
        else:self.stall_s+=dt
        return max(self.stall_s,self.blocked_s)>12.
