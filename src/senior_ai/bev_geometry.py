"""Ground-plane camera geometry. Public inputs contain calibration, never GT pose/depth."""
from dataclasses import dataclass
import numpy as np
import cv2

CAMERAS = ('front','front_right','right','rear_right','rear','rear_left','left','front_left')
OPTICAL = np.array([[0,1,0],[0,0,-1],[1,0,0]],dtype=np.float64)

@dataclass(frozen=True)
class Grid:
    x_min: float=-15.
    x_max: float=35.
    y_min: float=-20.
    y_max: float=20.
    resolution_m: float=.1

    def __post_init__(self):
        vals=np.array([self.x_min,self.x_max,self.y_min,self.y_max,self.resolution_m])
        if not np.isfinite(vals).all() or self.resolution_m<=0 or self.x_max<=self.x_min or self.y_max<=self.y_min:
            raise ValueError('Invalid metric grid')
        for span in [self.x_max-self.x_min,self.y_max-self.y_min]:
            if not np.isclose(span/self.resolution_m,round(span/self.resolution_m),atol=1e-7):raise ValueError('Grid span must contain whole cells')

    @property
    def shape(self):return (round((self.x_max-self.x_min)/self.resolution_m),round((self.y_max-self.y_min)/self.resolution_m))

    def points(self,z=0.):
        r,c=np.indices(self.shape);return np.stack([self.x_max-(r+.5)*self.resolution_m,self.y_min+(c+.5)*self.resolution_m,np.full(self.shape,z)],axis=-1)

    def xy_to_uv(self,xy):
        xy=np.asarray(xy);return np.stack([(xy[...,1]-self.y_min)/self.resolution_m-.5,(self.x_max-xy[...,0])/self.resolution_m-.5],axis=-1)

    def uv_to_xy(self,uv):
        uv=np.asarray(uv);return np.stack([self.x_max-(uv[...,1]+.5)*self.resolution_m,self.y_min+(uv[...,0]+.5)*self.resolution_m],axis=-1)

class Camera:
    def __init__(self,definition,width=640,height=360):
        self.name=definition['name'];self.K=np.asarray(definition['K'],dtype=np.float64);self.T=np.asarray(definition['T_ego_from_camera_local'],dtype=np.float64)
        self.width=width;self.height=height
        if self.K.shape!=(3,3) or self.T.shape!=(4,4) or not np.isfinite(self.K).all() or not np.isfinite(self.T).all():raise ValueError('Invalid camera matrices')
        if not np.allclose(self.K[2],[0,0,1]) or self.K[0,0]<=0 or self.K[1,1]<=0:raise ValueError('Invalid intrinsics')
        if not np.allclose(self.T[3],[0,0,0,1]) or not np.allclose(self.T[:3,:3].T@self.T[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(self.T[:3,:3]),1,atol=1e-5):raise ValueError('Extrinsics must be rigid')
        self.invT=np.linalg.inv(self.T);self.invK=np.linalg.inv(self.K)

    def project(self,xyz):
        xyz=np.asarray(xyz,dtype=np.float64);local=(xyz-self.T[:3,3])@self.T[:3,:3];opt=local@OPTICAL.T
        image=opt@self.K.T;uv=np.full(image.shape[:-1]+(2,),np.nan)
        np.divide(image[...,:2],image[...,2,None],out=uv,where=np.abs(image[...,2,None])>1e-12)
        return uv,opt[...,2]

    def ground(self,uv,z=0.,max_range=35.):
        uv=np.asarray(uv,dtype=np.float64);rays=np.concatenate([uv,np.ones(uv.shape[:-1]+(1,))],axis=-1)@self.invK.T@OPTICAL@self.T[:3,:3].T
        dz=rays[...,2];t=np.full(dz.shape,np.nan);np.divide(z-self.T[2,3],dz,out=t,where=dz < -1e-9)
        xyz=self.T[:3,3]+t[...,None]*rays
        valid=np.isfinite(xyz).all(-1)&(t>.25)&(np.linalg.norm(xyz[...,:2]-self.T[:2,3],axis=-1)<=max_range)
        valid&=(uv[...,0]>=0)&(uv[...,0]<self.width)&(uv[...,1]>=0)&(uv[...,1]<self.height)
        return np.where(valid[...,None],xyz,np.nan),valid

def cameras_from_calibration(value):
    if not np.array_equal(value.get('optical_from_carla_camera'),OPTICAL):raise ValueError('Optical convention differs')
    if len(value['cameras'])!=8 or {c['name'] for c in value['cameras']}!=set(CAMERAS):raise ValueError('Exact eight-camera rig required')
    return {c['name']:Camera(c) for c in value['cameras']}

def projection_maps(camera,grid,z=0.,max_range=35.,vehicle_mask=None):
    xyz=grid.points(z);uv,axial=camera.project(xyz)
    valid=(axial>.25)&np.isfinite(uv).all(-1)&(uv[...,0]>=1)&(uv[...,0]<camera.width-2)&(uv[...,1]>=1)&(uv[...,1]<camera.height-2)
    distance=np.linalg.norm(xyz[...,:2]-camera.T[:2,3],axis=-1);valid&=distance<=max_range
    if vehicle_mask:valid&=~((np.abs(xyz[...,0])<vehicle_mask['half_length_m'])&(np.abs(xyz[...,1])<vehicle_mask['half_width_m']))
    # Prefer central views; weights are geometry, not independent evidence counts.
    edge=np.clip(1-np.abs((uv[...,0]-camera.K[0,2])/camera.K[0,0]),0,1)
    weight=np.where(valid,np.maximum(edge,.01)/(1+(distance/15)**2),0).astype(np.float32)
    # CARLA depth audit uses pixel centers (column+.5,row+.5); remap uses array indices.
    return np.nan_to_num(uv-.5,nan=-1,posinf=-1,neginf=-1).astype(np.float32),valid,weight

def sample(image,maps,nearest=False):
    uv=maps[0] if isinstance(maps,tuple) else maps
    return cv2.remap(np.asarray(image),uv[...,0],uv[...,1],cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def warp_metric(image,previous_to_current,grid,nearest=False):
    t=np.asarray(previous_to_current,dtype=np.float64)
    if t.shape!=(3,3) or not np.isfinite(t).all() or not np.allclose(t[2],[0,0,1]) or not np.allclose(t[:2,:2].T@t[:2,:2],np.eye(2),atol=1e-5) or not np.isclose(np.linalg.det(t[:2,:2]),1,atol=1e-5):raise ValueError('Invalid SE2')
    xy=grid.points()[...,:2];old=(np.c_[xy.reshape(-1,2),np.ones(xy.shape[0]*xy.shape[1])]@np.linalg.inv(t).T)[:,:2].reshape(xy.shape)
    return sample(image,grid.xy_to_uv(old).astype(np.float32),nearest)
