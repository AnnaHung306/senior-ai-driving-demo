"""Memoize static camera/ground projection; preserve original fusion arithmetic."""
import copy
import numpy as np
from .bev_geometry import projection_maps,sample
from .bev_fusion import GROUND_IDS


class CachedRoadFusion:
    def __init__(self,cameras,grid,config):
        self.grid=grid;self.config=copy.deepcopy(config)
        self.maps={name:projection_maps(camera,grid,config['ground_z_m'],config['max_camera_range_m'],config['vehicle_mask']) for name,camera in cameras.items()}
    def __call__(self,observations):
        if set(observations)!=set(self.maps):raise ValueError('Cached calibration/camera set changed')
        total=np.zeros(self.grid.shape,np.float32);weighted=np.zeros(self.grid.shape+(2,),np.float32)
        support=np.zeros(self.grid.shape,np.uint8);coverage=np.zeros(self.grid.shape,np.uint8);maps={}
        for name,ob in observations.items():
            m=self.maps[name];maps[name]=m
            ids=sample(ob['ids'],m,True);surface=np.isin(ids,GROUND_IDS)&m[1]
            probability=sample(ob['ground_probabilities'],m);weight=m[2]*surface
            weighted+=probability*weight[...,None];total+=weight;support+=surface;coverage+=m[1]
        probabilities=np.divide(weighted,total[...,None],out=np.zeros_like(weighted),where=total[...,None]>0)
        return {'probabilities':probabilities,'observed':total>0,'support':support,'geometric_coverage':coverage,'maps':maps}
