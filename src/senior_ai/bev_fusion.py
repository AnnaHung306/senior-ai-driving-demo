"""Calibrated learned road fusion and ground-contact hypotheses, not 3D boxes."""
import numpy as np
from .bev_geometry import projection_maps,sample

GROUND_IDS=[1,2,10,16]

def fuse_road(observations,cameras,grid,config,only_camera=None):
    total=np.zeros(grid.shape,np.float32);weighted=np.zeros(grid.shape+(2,),np.float32);support=np.zeros(grid.shape,np.uint8);coverage=np.zeros(grid.shape,np.uint8)
    maps={}
    for name,observation in observations.items():
        if only_camera and name!=only_camera:continue
        m=projection_maps(cameras[name],grid,config['ground_z_m'],config['max_camera_range_m'],config['vehicle_mask']);maps[name]=m
        ids=sample(observation['ids'],m,True);surface=np.isin(ids,GROUND_IDS)&m[1]
        probability=sample(observation['ground_probabilities'],m);w=m[2]*surface
        weighted+=probability*w[...,None];total+=w;support+=surface;coverage+=m[1]
    probabilities=np.divide(weighted,total[...,None],out=np.zeros_like(weighted),where=total[...,None]>0)
    return {'probabilities':probabilities,'observed':total>0,'support':support,'geometric_coverage':coverage,'maps':maps}

def ground_objects(observations,cameras,config):
    accepted=[];rejected=[]
    for name,ob in observations.items():
        for index,obj in enumerate(ob['objects']):
            if obj['score']<config['object_min_score'] or obj['class_name']=='traffic_light':continue
            x0,y0,x1,y1=obj['bbox_xyxy'];u=(x0+x1)/2;v=y1+2
            reason=None
            if x0<=1 or x1>=639 or y0<=1 or v>=359:reason='truncated_or_no_contact_support'
            xyz,valid=cameras[name].ground([[u+.5,v+.5]],config['ground_z_m'],config['object_max_range_m'])
            if not valid[0]:reason='no_ground_intersection'
            if reason is None and ob['ids'][int(v),int(u)] not in GROUND_IDS:reason='contact_not_adjacent_to_predicted_ground'
            record={**obj,'camera':name,'detection_index':index}
            if reason:rejected.append({**record,'reason':reason});continue
            accepted.append({**record,'xy_m':xyz[0,:2].tolist(),'geometry':'bbox_bottom_ground_plane_hypothesis_not_3D_box'})
    return accepted,rejected

def deduplicate(objects,distance=1.5):
    clusters=[]
    for obj in sorted(objects,key=lambda x:(-x['score'],x['camera'],x['detection_index'])):
        match=None
        for cluster in clusters:
            if any(x['camera']==obj['camera'] for x in cluster) or cluster[0]['class_name']!=obj['class_name']:continue
            if all(np.linalg.norm(np.asarray(obj['xy_m'])-x['xy_m'])<=distance for x in cluster):match=cluster;break
        if match is None:clusters.append([obj])
        else:match.append(obj)
    result=[]
    for i,cluster in enumerate(clusters):
        weights=np.array([x['score'] for x in cluster]);xy=np.average([x['xy_m'] for x in cluster],axis=0,weights=weights)
        result.append({'frame_local_id':i,'class_name':cluster[0]['class_name'],'xy_m':xy.tolist(),'score_uncalibrated':float(weights.max()),'sources':cluster,'geometry':'assumed_ground_contact','maximum_source_separation_m':max([np.linalg.norm(np.asarray(a['xy_m'])-b['xy_m']) for a in cluster for b in cluster])})
    return result
