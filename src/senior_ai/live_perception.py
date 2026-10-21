"""Persistent frozen eight-RGB inference. Accepts arrays/calibration/time only."""
from pathlib import Path
import sys, time
import numpy as np
import torch, cv2
from .world_display import read, sha
from .bev_geometry import Grid, cameras_from_calibration
from .bev_fusion import fuse_road
from .bev_temporal import CausalRoadFusion
from .bev_path import lane_center
from .cuboid import CuboidNet, geometry_features, crop, decode, fuse_cuboids, prior_box
from .world_tracking import WorldTracker
from .motion_forecast import MotionTracker, appearance_histogram
from .light_model import LightStateNet, crop_rgb
from .light_labels import LIGHT_STATES
from .road_fusion_cache import CachedRoadFusion

ROOT=Path(__file__).resolve().parents[2]

def track_signal_rois(previous_gray,current_gray,rois,timestamp):
    """Track an observed image patch through braking; never invent a lamp ROI."""
    tracked=[]
    if previous_gray is None:return tracked
    for old in rois:
        if timestamp-old['last_detected_s']>30:continue
        x0,y0,x1,y1=old['bbox_xyxy']
        points=np.array([[x0-2,y0-2],[x1+2,y0-2],[x0-2,y1+2],[x1+2,y1+2],[(x0+x1)/2,(y0+y1)/2]],np.float32).reshape(-1,1,2)
        flow,status,_=cv2.calcOpticalFlowPyrLK(previous_gray,current_gray,points,None,winSize=(15,15),maxLevel=2)
        if flow is None or status.sum()<3:continue
        valid=status.ravel().astype(bool);delta=np.median((flow-points).reshape(-1,2)[valid],axis=0)
        if not np.isfinite(delta).all() or np.linalg.norm(delta)>25:continue
        box=(np.array([x0,y0,x1,y1])+np.tile(delta,2)).tolist()
        if box[0]<0 or box[1]<0 or box[2]>=640 or box[3]>=360:continue
        tracked.append({**old,'bbox_xyxy':box,'ROI_tracking':'current_RGB_optical_flow'})
    return tracked

class LivePerception:
    def __init__(self, calibration, device='cuda'):
        sys.path.insert(0,str(ROOT/'scripts'))
        from predict_sunny import load_model
        self.device=device;self.c=read(ROOT/'configs/phase4_bev_v1.json');self.c3=read(ROOT/'configs/phase5_world_v1.json')
        torch.set_num_threads(4);cv2.setNumThreads(1)
        for m in self.c['models'].values():
            if sha(ROOT/m['path'])!=m['sha256']:raise ValueError('Frozen checkpoint changed')
        self.seg,_,_=load_model(ROOT/self.c['models']['segmentation']['path'],device)
        self.det,_,_=load_model(ROOT/self.c['models']['detection']['path'],device)
        self.light=LightStateNet().to(device).eval()
        self.light.load_state_dict(torch.load(ROOT/self.c['models']['phase']['path'],map_location='cpu',weights_only=False)['model'])
        checkpoint=ROOT/'artifacts/runs/phase5-cuboid-v1-001/best.pt'
        if sha(checkpoint)!='1ad8ccbf8e5409b476219b8ecaf0924922a756c19b845043b9cbeae99fd32f6f':raise ValueError('Frozen cuboid checkpoint changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for p,h in saved['contract']['sources'].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen cuboid implementation changed')
        self.cuboid=CuboidNet().to(device).eval();self.cuboid.load_state_dict(saved['model'])
        self.priors=saved['class_size_priors'];self.supported=saved['supported_class_ids']
        self.cameras=cameras_from_calibration(calibration);self.grid=Grid(**self.c['grid'])
        self.temporal=CausalRoadFusion(self.grid,self.c);self.tracker=WorldTracker(self.c3['tracker'])
        self.motion=MotionTracker(read(ROOT/'configs/phase7_motion_v1_2.json'))
        self.signal_roi=[]
        self.previous_front_gray=None
        self.calibration_sha=__import__('hashlib').sha256(__import__('json').dumps(calibration,sort_keys=True).encode()).hexdigest()
        optimization=ROOT/'configs/phase10_optimization_v1.json'
        self.projection_cache=CachedRoadFusion(self.cameras,self.grid,self.c) if optimization.exists() and read(optimization).get('projection_cache_enabled') else None

    @torch.inference_mode()
    def infer(self, rgb, scene, frame, timestamp):
        started=time.perf_counter()
        if set(rgb)!=set(self.cameras) or any(a.shape!=(360,640,3) or a.dtype!=np.uint8 for a in rgb.values()):raise ValueError('Eight real uint8 RGB arrays required')
        names=list(rgb);t=torch.stack([torch.from_numpy(rgb[n].transpose(2,0,1).copy()) for n in names]).to(self.device).float()/255
        logits=self.seg(t)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite segmentation')
        probs=logits.softmax(1);ids=logits.argmax(1).cpu().numpy().astype(np.uint8)
        ground=torch.stack([probs[:,self.c['road_ids']].sum(1),probs[:,self.c['line_id']]],-1).cpu().numpy()
        segmentation_done=time.perf_counter()
        observations={};signals=[]
        for i,n in enumerate(names):
            detection=self.det([t[i]])[0];objects=[]
            if not torch.isfinite(detection['boxes']).all() or not torch.isfinite(detection['scores']).all():raise ValueError('Nonfinite detection')
            for b,l,s in zip(detection['boxes'].cpu().tolist(),detection['labels'].cpu().tolist(),detection['scores'].cpu().tolist()):
                d={'bbox_xyxy':b,'class_id':l,'score':s}
                if l==7 and s>=.5:
                    light=self.light(crop_rgb(rgb[n],b).to(self.device).float()[None]/255).softmax(1)[0]
                    d['phase']=LIGHT_STATES[int(light.argmax())]
                    if n=='front' and 200<(b[0]+b[2])/2<440:
                        signals.append({'phase':d['phase'],'score':s,'camera':n,'bbox_xyxy':b})
                objects.append(d)
            observations[n]={'rgb':rgb[n],'ids':ids[i],'ground_probabilities':ground[i],'objects':objects}
        detection_done=time.perf_counter()
        fused=self.projection_cache(observations) if self.projection_cache is not None else fuse_road(observations,self.cameras,self.grid,self.c)
        temporal,ego_motion=self.temporal.update(fused,observations,self.cameras,scene,timestamp,self.calibration_sha)
        current_gray=cv2.cvtColor(rgb['front'],cv2.COLOR_RGB2GRAY)
        retained=track_signal_rois(self.previous_front_gray,current_gray,self.signal_roi,timestamp)
        self.previous_front_gray=current_gray
        # A previously observed lamp may disappear from detector outputs when
        # its colour changes. Only retain its image ROI while RGB SE2 confirms
        # a stationary camera; classification always uses the CURRENT pixels.
        transform=np.asarray(ego_motion.get('previous_to_current',np.eye(3)))
        stationary=ego_motion.get('status')=='aligned' and np.linalg.norm(transform[:2,2])<.03 and abs(np.arctan2(transform[1,0],transform[0,0]))<.005
        if signals:
            self.signal_roi=[{**s,'last_detected_s':timestamp} for s in signals]
        else:
            self.signal_roi=retained
        if not signals and stationary:
            for old in retained:
                if timestamp-old['last_detected_s']>30:continue
                lp=self.light(crop_rgb(rgb['front'],old['bbox_xyxy']).to(self.device).float()[None]/255).softmax(1)[0]
                phase=LIGHT_STATES[int(lp.argmax())] if float(lp.max())>=.70 else 'unknown_or_unobservable'
                signals.append({**old,'phase':phase,'state_score_uncalibrated':float(lp.max()),'association':'stationary_RGB_retained_detection_ROI','ROI_age_s':timestamp-old['last_detected_s']})
        road_path=lane_center(temporal,self.grid,self.c['path'])
        fusion_done=time.perf_counter()
        candidates=[];features=[];crops=[]
        for n,ob in observations.items():
            for i,d in enumerate(ob['objects']):
                if d['score']<self.c3['minimum_detection_score'] or d['class_id'] not in self.supported:continue
                g=geometry_features(self.cameras[n],d['bbox_xyxy'],d['class_id'],self.c3['maximum_anchor_range_m'])
                if g is None:continue
                f,anchor=g;features.append(f);crops.append(crop(rgb[n],d['bbox_xyxy'],self.c3['crop_size']));candidates.append((n,i,d,anchor))
        predictions=self.cuboid(torch.from_numpy(np.stack(crops)).to(self.device).float()/255,torch.from_numpy(np.stack(features)).to(self.device)).cpu().numpy() if candidates else []
        single=[];priors=[]
        for (n,i,d,a),pred in zip(candidates,predictions):
            o=decode(pred,a,self.cameras[n],d['class_id'],self.priors,self.c3['center_residual_scale_m'])
            if np.linalg.norm(o['center_ego_m'][:2])<=30:
                single.append({**o,'camera':n,'detection_index':i,'bbox_xyxy':d['bbox_xyxy'],'score_uncalibrated':d['score']})
                priors.append({**prior_box(a,self.cameras[n],d['class_id'],self.priors),'camera':n,'detection_index':i,'bbox_xyxy':d['bbox_xyxy'],'score_uncalibrated':d['score']})
        measured,tracking=self.tracker.update(fuse_cuboids(single,self.c3['merge_distance_m']),scene,timestamp,self.calibration_sha,ego_motion)
        dynamic,dynamics=self.motion.update(measured,scene,timestamp,self.calibration_sha,ego_motion,[appearance_histogram(rgb,o['sources']) for o in measured])
        if self.device=='cuda':torch.cuda.synchronize()
        self.audit_output={'observations':observations,'single_3D':single,'prior_3D':priors}
        stage_ms={'input_and_segmentation':(segmentation_done-started)*1000,
                  'eight_detection_and_detected_lights':(detection_done-segmentation_done)*1000,
                  'BEV_temporal_and_retained_light':(fusion_done-detection_done)*1000,
                  'cuboid_tracking_and_motion':(time.perf_counter()-fusion_done)*1000}
        step=5;xy=self.grid.points()[::step,::step,:2];p=fused['probabilities'][::step,::step];seen=fused['observed'][::step,::step]
        view={'frame':int(frame),'timestamp_s':float(timestamp),'source':'live_eight_RGB_frozen_neural_models',
              'objects':dynamic,'signals':signals,'road_path':road_path,'RGB_ego_motion':ego_motion,
              'road_cells_xy_m':xy[seen&(p[...,0]>.4)].tolist(),'lane_cells_xy_m':xy[seen&(p[...,1]>.25)].tolist(),
              'tracking':tracking,'dynamics':dynamics,'inference_ms':(time.perf_counter()-started)*1000,'stage_wall_ms':stage_ms,
              'GT_other_actors_used':False,'learned_planner':False,'model_quality_passed':False}
        return view,fused

    def road_fractions(self, fused, local_path, config):
        points=np.asarray(local_path,float)
        # A destination can be closer than the camera's hood blind zone. Use
        # continuation of the final tangent for *road context only*, never as
        # a new destination or controller path, with unchanged evidence gate.
        if len(points)>=2 and points[-1,0]<12:
            tangent=points[-1]-points[-2];norm=np.linalg.norm(tangent)
            if norm>.01 and tangent[0]/norm>.7:
                extra=points[-1]+np.arange(1.,13.)[:,None]*tangent/norm
                points=np.vstack([points,extra])
        points=points[(points[:,0]>=3)&(points[:,0]<=12)]
        result={}
        for offset in config['lateral_offsets_m']:
            samples=np.repeat(points,3,axis=0)
            if not len(samples):result[str(offset)]=0.;continue
            samples[:,1]+=np.tile([-.65,0,.65],len(points))+offset*np.clip(samples[:,0]/6.,0,1)
            uv=self.grid.xy_to_uv(samples);col=np.rint(uv[:,0]).astype(int);row=np.rint(uv[:,1]).astype(int)
            valid=(row>=0)&(row<self.grid.shape[0])&(col>=0)&(col<self.grid.shape[1]);ok=np.zeros(len(samples),bool)
            ok[valid]=fused['observed'][row[valid],col[valid]]&(fused['probabilities'][row[valid],col[valid],0]>=config['minimum_road_probability'])
            result[str(offset)]=float(ok.mean())
        return result
