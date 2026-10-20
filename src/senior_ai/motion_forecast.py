"""Causal RGB-derived ground tracking and explicitly kinematic forecasts. No GT IO."""
import copy
import math
import numpy as np


def assignment(cost, unmatched=16.):
    """Rectangular Hungarian assignment with one independent dummy per row."""
    a=np.asarray(cost,dtype=float)
    if a.ndim!=2 or np.isnan(a).any():raise ValueError('Finite or gated-infinite 2D cost required')
    n,d=a.shape
    if not n or not d:return []
    a=np.concatenate([np.where(np.isfinite(a),a,1e9),np.full((n,n),unmatched)],axis=1)
    m=a.shape[1];u=np.zeros(n+1);v=np.zeros(m+1);p=np.zeros(m+1,int);way=np.zeros(m+1,int)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=np.full(m+1,np.inf);used=np.zeros(m+1,bool)
        while True:
            used[j0]=True;i0=p[j0];delta=np.inf;j1=0
            for j in range(1,m+1):
                if used[j]:continue
                current=a[i0-1,j-1]-u[i0]-v[j]
                if current<minimum[j]:minimum[j]=current;way[j]=j0
                if minimum[j]<delta:delta=minimum[j];j1=j
            for j in range(m+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    return [(int(p[j]-1),j-1) for j in range(1,d+1) if p[j] and a[p[j]-1,j-1]<unmatched]


def rigid_se2(value):
    t=np.asarray(value,float)
    if t.shape!=(3,3) or not np.isfinite(t).all() or not np.allclose(t[2],[0,0,1],atol=1e-8) or not np.allclose(t[:2,:2].T@t[:2,:2],np.eye(2),atol=1e-6) or not np.isclose(np.linalg.det(t[:2,:2]),1):
        raise ValueError('Rigid finite previous-to-current SE2 required')
    return t


def forbid_truth(value):
    if isinstance(value,dict):
        if set(value)&{'actor_id','world_pose','world_vertices_m','depth','ego_velocity_mps','velocity_mps','future_positions'}:raise ValueError('GT/future fields forbidden in motion inference')
        for x in value.values():forbid_truth(x)
    elif isinstance(value,(list,tuple)):
        for x in value:forbid_truth(x)


def appearance_histogram(rgb_by_camera, sources):
    result=[]
    for source in sources:
        image=rgb_by_camera[source['camera']];x0,y0,x1,y1=source['bbox_xyxy']
        # A central crop reduces background contamination, without using masks/GT.
        dx=(x1-x0)*.1;dy=(y1-y0)*.1
        patch=image[max(0,int(y0+dy)):min(image.shape[0],int(math.ceil(y1-dy))),max(0,int(x0+dx)):min(image.shape[1],int(math.ceil(x1-dx)))]
        if not patch.size:continue
        vector=np.concatenate([np.histogram(patch[...,c],bins=8,range=(0,256))[0] for c in range(3)]).astype(float)
        result.append(vector/vector.sum())
    return np.mean(result,axis=0) if result else None


def source_overlap(a,b):
    """Maximum same-camera 2D overlap; unavailable views cannot invent support."""
    overlaps=[]
    for x in a.get('sources',[]):
        for y in b.get('sources',[]):
            if x['camera']!=y['camera']:continue
            p=x['bbox_xyxy'];q=y['bbox_xyxy']
            intersection=max(0,min(p[2],q[2])-max(p[0],q[0]))*max(0,min(p[3],q[3])-max(p[1],q[1]))
            area=(p[2]-p[0])*(p[3]-p[1])+(q[2]-q[0])*(q[3]-q[1])-intersection
            overlaps.append(intersection/max(area,1e-12))
    return max(overlaps) if overlaps else None


def ego_forecast(velocity,yaw_rate,times):
    v=np.asarray(velocity,float);t=np.asarray(times,float);w=float(yaw_rate)
    if not np.isfinite(v).all() or v.shape!=(2,) or not np.isfinite(w):raise ValueError('Finite ego twist required')
    if abs(w)<1e-8:return t[:,None]*v
    s=np.sin(w*t)/w;c=(1-np.cos(w*t))/w
    return np.stack([s*v[0]-c*v[1],c*v[0]+s*v[1]],axis=1)


def swept_contact(relative, times, radius):
    """Continuous circle contact within piecewise-linear intervals; no tunneling."""
    first=None;clearance=float('inf')
    for i in range(len(times)-1):
        p=np.asarray(relative[i]);d=np.asarray(relative[i+1])-p;r=max(float(radius[i]),float(radius[i+1]))
        aa=float(d@d);bb=float(2*p@d);cc=float(p@p-r*r)
        closest=np.clip(-float(p@d)/aa,0,1) if aa>1e-15 else 0
        clearance=min(clearance,float(np.linalg.norm(p+closest*d)-r))
        if cc<=0:alpha=0.
        elif aa<1e-15:continue
        else:
            discriminant=bb*bb-4*aa*cc
            if discriminant<0:continue
            alpha=(-bb-math.sqrt(discriminant))/(2*aa)
            if alpha<0 or alpha>1:continue
        contact=float(times[i]+alpha*(times[i+1]-times[i]));first=contact if first is None else min(first,contact)
    return first,clearance


def collision_risk(obj, forecast, ego, config):
    if forecast['status']!='available' or ego['status']!='estimated':
        return {'status':'unknown','TTC_s':None,'reason':'insufficient_motion_or_ego_information','probability_calibrated':False}
    times=np.array(forecast['times_s']);own=np.array(ego['positions_ego_xy_m']);positions=np.array(forecast['positions_ego_xy_m'])
    radius=(math.hypot(*config['ego_size_lw_m'])+math.hypot(*obj['size_lwh_m'][:2]))/2+config['collision_margin_m']
    radii=np.full(len(times),radius);nominal,clearance=swept_contact(positions-own,times,radii)
    uncertain,_=swept_contact(positions-own,times,radii+np.array(forecast['uncertainty_radius_m']))
    brake,_=swept_contact(np.array(forecast['braking_sensitivity_ego_xy_m'])-own,times,radii)
    possible=min([t for t in [uncertain,brake] if t is not None],default=None)
    status='high' if nominal is not None and nominal<=2 else 'watch' if nominal is not None else 'potential' if possible is not None else 'not_triggered'
    return {'status':status,'TTC_s':nominal,'possible_TTC_s':possible,'minimum_nominal_clearance_m':clearance,
            'reason':'conservative_circles_under_constant_twist_and_object_CV; possible_includes_uncertainty_or_braking',
            'probability_calibrated':False,'not_triggered_means_safe':False}


class MotionTracker:
    def __init__(self,config):
        self.c=copy.deepcopy(config);self.tracks={};self.previous=None;self.counter=0
        self.times=np.arange(0,config['forecast_horizon_s']+config['forecast_step_s']/2,config['forecast_step_s'])
        if self.times[0]!=0 or len(self.times)<2 or len(self.times)>101:raise ValueError('Bounded forecast grid required')

    def _new(self,obj,feature,timestamp):
        self.counter+=1;key=obj['track_id'] if self.c.get('association_mode')=='verified_upstream_identity' else f'{self.scene}:T7-{self.counter:06d}'
        state=np.array([*obj['center_ego_m'][:2],0.,0.]);r=self.c['measurement_sigma_m']**2
        return key,{'x':state,'P':np.diag([r,r,self.c['initial_velocity_sigma_mps']**2,self.c['initial_velocity_sigma_mps']**2]),
            'object':copy.deepcopy(obj),'appearance':feature,'hits':1,'filter_hits':1,'first':timestamp,'last_measurement':timestamp,'history':[], 'ambiguous':False}

    def _forecast(self,track,obj,timestamp):
        c=self.c;v=track['x'][2:];speed=float(np.linalg.norm(v));sigma=float(np.sqrt(max(np.linalg.eigvalsh(track['P'][2:,2:]))))
        ready=track['filter_hits']>=c['minimum_hits'] and timestamp-track['first']>=c['minimum_history_s']-1e-6
        available=ready and sigma<=c['velocity_sigma_limit_mps'] and speed<=c['maximum_speed_mps'] and not track['ambiguous']
        if not ready:status='warming_up'
        elif not available:status='uncertain'
        elif speed+2*sigma<c['stationary_threshold_mps']:status='stationary'
        elif speed-2*sigma>c['stationary_threshold_mps']:status='moving'
        else:status='uncertain'
        motion={'status':status,'velocity_available':available,'velocity_ground_ego_mps':v.tolist() if available else None,
                'speed_mps':speed if available else None,'heading_rad':float(np.arctan2(v[1],v[0])) if available and speed>.1 else None,
                'velocity_covariance':track['P'][2:,2:].tolist(),'velocity_sigma_max_mps':sigma,'confidence_calibrated':False,
                'association_ambiguous':track['ambiguous'],'observed_history_s':float(timestamp-track['first']),'velocity_measurements':track['filter_hits']}
        forecast={'status':'available' if available else 'unavailable','model':'constant_velocity_with_braking_sensitivity_not_learned_intent',
                  'coordinate_frame':'fixed_current_ego','times_s':[],'positions_ego_xy_m':[],'uncertainty_radius_m':[],
                  'braking_sensitivity_ego_xy_m':[],'probability_calibrated':False}
        if available:
            p=np.array(obj['center_ego_m'][:2]);t=self.times
            forecast['times_s']=t.tolist();forecast['positions_ego_xy_m']=(p+t[:,None]*v).tolist()
            stopped=np.minimum(t,speed/c['braking_assumption_mps2']);distance=speed*stopped-.5*c['braking_assumption_mps2']*stopped**2
            forecast['braking_sensitivity_ego_xy_m']=(p+distance[:,None]*(v/max(speed,1e-12))).tolist()
            for h in t:
                F=np.concatenate([np.eye(2),h*np.eye(2)],axis=1)
                covariance=F@track['P']@F.T+np.eye(2)*(c['measurement_sigma_m']**2+c['acceleration_sigma_mps2']**2*h**4/4)
                forecast['uncertainty_radius_m'].append(float(2*np.sqrt(max(np.linalg.eigvalsh(covariance)))))
        return motion,forecast

    def update(self,objects,scene,timestamp,calibration_sha,motion,appearances=None):
        forbid_truth(objects);forbid_truth(motion)
        if not isinstance(scene,str) or not scene or not np.isfinite(timestamp):raise ValueError('Finite identified input required')
        for o in objects:
            p=np.array(o['center_ego_m'],float);size=np.array(o['size_lwh_m'],float)
            if o['class_id'] not in range(1,7) or p.shape!=(3,) or size.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(size).all() or np.any(size<=0) or not np.isfinite(o['yaw_rad']):raise ValueError('Invalid measured cuboid')
        features=appearances if appearances is not None else [None]*len(objects)
        if len(features)!=len(objects):raise ValueError('Appearance count mismatch')
        for f in features:
            if f is not None and (np.shape(f)!=(24,) or not np.isfinite(f).all() or np.any(np.asarray(f)<0) or not np.isclose(np.sum(f),1)):raise ValueError('Normalized RGB appearance required')
        self.scene=scene;reset='cold_start';dt=None;T=None
        if self.previous is not None:
            old_scene,old_t,old_calibration=self.previous;dt=float(timestamp-old_t)
            if scene!=old_scene or calibration_sha!=old_calibration:reset='stream_identity_changed'
            elif dt<=0 or dt>self.c['maximum_gap_s']:reset='timestamp_gap_or_reversal'
            elif motion.get('status')!='aligned':reset='untrusted_ego_motion'
            else:T=rigid_se2(motion['previous_to_current']);reset=None
        if reset:self.tracks.clear()
        ego={'status':'unavailable','velocity_ego_mps':None,'yaw_rate_radps':None,'times_s':[],'positions_ego_xy_m':[], 'source':'RGB_ground_SE2_only'}
        if T is not None:
            R=T[:2,:2];translation=T[:2,2];B=np.zeros((4,4));B[:2,:2]=R;B[2:,2:]=R
            F=np.eye(4);F[:2,2:]=dt*np.eye(2);G=np.concatenate([.5*dt*dt*np.eye(2),dt*np.eye(2)],axis=0)
            for track in self.tracks.values():
                track['x']=B@track['x'];track['x'][:2]+=translation;track['P']=B@track['P']@B.T
                for entry in track['history']:entry['xy']=R@entry['xy']+translation
                track['x']=F@track['x'];track['P']=F@track['P']@F.T+G@G.T*self.c['acceleration_sigma_mps2']**2
            velocity=-translation/dt;yaw_rate=-float(np.arctan2(R[1,0],R[0,0]))/dt
            ego={'status':'estimated','velocity_ego_mps':velocity.tolist(),'yaw_rate_radps':yaw_rate,'times_s':self.times.tolist(),
                 'positions_ego_xy_m':ego_forecast(velocity,yaw_rate,self.times).tolist(),'source':'RGB_ground_SE2_only'}
        for key in list(self.tracks):
            if timestamp-self.tracks[key]['last_measurement']>self.c['maximum_coast_s']+1e-9:del self.tracks[key]
        keys=list(self.tracks);cost=np.full((len(keys),len(objects)),np.inf);r=np.eye(2)*self.c['measurement_sigma_m']**2
        for i,key in enumerate(keys):
            tr=self.tracks[key]
            for j,o in enumerate(objects):
                if o['class_id']!=tr['object']['class_id']:continue
                delta=np.array(o['center_ego_m'][:2])-tr['x'][:2];maha=float(delta@np.linalg.solve(tr['P'][:2,:2]+r,delta))
                size=float(np.max(np.abs(np.log(np.array(o['size_lwh_m'])/tr['object']['size_lwh_m']))))
                if maha>self.c['mahalanobis_gate'] or np.linalg.norm(delta)>self.c['maximum_association_distance_m'] or size>math.log(2.5):continue
                appearance=0.
                if features[j] is not None and tr['appearance'] is not None:appearance=1-float(np.sqrt(features[j]*tr['appearance']).sum())
                spatial=maha
                if 'association_spatial_weight' in self.c:spatial=maha/self.c['mahalanobis_gate']*self.c['association_spatial_weight']
                overlap=source_overlap(tr['object'],o);bbox_cost=(1-overlap) if overlap is not None else 1.
                cost[i,j]=spatial+self.c['appearance_weight']*appearance+size+self.c.get('association_bbox_weight',0)*bbox_cost
        backbone=self.c.get('association_mode')=='verified_upstream_identity'
        if backbone:
            source_ids=[o.get('track_id') for o in objects]
            if len(set(source_ids))!=len(source_ids) or any(not isinstance(k,str) or not k.startswith(scene+':T') for k in source_ids):raise ValueError('Unique verified upstream IDs required')
            matched={j:o['track_id'] for j,o in enumerate(objects) if o['track_id'] in self.tracks}
            if any(self.tracks[key]['object']['class_id']!=objects[j]['class_id'] for j,key in matched.items()):raise ValueError('Verified upstream identity changed class')
        else:
            pairs=assignment(cost);matched={j:keys[i] for i,j in pairs}
        used=set(matched.values());output=[]
        for j,obj in enumerate(objects):
            key=matched.get(j)
            if key is None:key,tr=self._new(obj,features[j],timestamp)
            else:
                tr=self.tracks[key];row=keys.index(key)
                # Preserve established visual identity, but do not turn an outlier
                # into a confident velocity jump. Re-warm this velocity filter.
                rewarmed=backbone and not np.isfinite(cost[row,j])
                if rewarmed:
                    identity_hits=tr['hits'];_,tr=self._new(obj,features[j],timestamp);tr['hits']=identity_hits
                else:
                    H=np.concatenate([np.eye(2),np.zeros((2,2))],axis=1);P=tr['P'];K=np.linalg.solve(H@P@H.T+r,H@P).T
                    tr['x']+=K@(np.array(obj['center_ego_m'][:2])-H@tr['x']);I=np.eye(4)-K@H;tr['P']=I@P@I.T+K@r@K.T
                    tr['P']=(tr['P']+tr['P'].T)/2;tr['filter_hits']+=1
                tr['hits']+=1
                alternatives=[v for jj,v in enumerate(cost[row]) if jj!=j and np.isfinite(v)]
                tr['ambiguous']=False if backbone else bool(alternatives and min(alternatives)-cost[row,j]<.3)
                if features[j] is not None:tr['appearance']=features[j] if tr['appearance'] is None else .8*tr['appearance']+.2*features[j]
            tr['object']=copy.deepcopy(obj);tr['last_measurement']=timestamp
            tr['history'].append({'timestamp_s':float(timestamp),'xy':np.array(obj['center_ego_m'][:2],float)})
            tr['history']=tr['history'][-self.c['history_length']:];self.tracks[key]=tr
            measured={**copy.deepcopy(obj),'upstream_track_id':obj.get('track_id'),'track_id':key,'track_age_measurements':tr['hits'],'measurement_current':True}
            movement,forecast=self._forecast(tr,obj,timestamp)
            measured.update(motion=movement,forecast=forecast,history_ego_xy_m=[x['xy'].tolist() for x in tr['history']],lifecycle='confirmed' if tr['hits']>=self.c['minimum_hits'] else 'tentative')
            measured['risk']=collision_risk(measured,forecast,ego,self.c);output.append(measured)
        held=[]
        for key,tr in self.tracks.items():
            if key in used or tr['last_measurement']==timestamp:continue
            held.append({'track_id':key,'class_id':tr['object']['class_id'],'center_ego_m':[float(tr['x'][0]),float(tr['x'][1]),tr['object']['center_ego_m'][2]],
                         'size_lwh_m':tr['object']['size_lwh_m'],'age_since_measurement_s':float(timestamp-tr['last_measurement']),
                         'lifecycle':'coasting','measurement_current':False,'used_for_occupancy':False})
        self.previous=scene,float(timestamp),calibration_sha
        statuses=[o['risk']['status'] for o in output];priority=['high','watch','potential','unknown','not_triggered']
        risk=next((s for s in priority if s in statuses),'unknown')
        association='verified_RGB_upstream_identity_plus_causal_Kalman' if backbone else 'causal_SE2_Kalman_Hungarian_RGB_bbox_histogram' if self.c.get('association_bbox_weight') else 'causal_SE2_Kalman_Hungarian_RGB_histogram'
        return output,{'reset_reason':reset,'association':association,'held_unobserved_tracks':held,
                       'ego_motion':ego,'risk_summary':risk,'GT_identity_or_future_used':False,'held_tracks_used_for_occupancy':False,
                       'velocity_definition':'ground velocity in current ego axes; not relative velocity','forecast_coordinate_frame':'fixed_current_ego'}
