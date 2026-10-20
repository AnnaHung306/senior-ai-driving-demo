"""Visible lane center, explicit optional width prior; no map/GT route or controls."""
import numpy as np

def lane_center(bev,grid,config):
    xyz=grid.points();p=bev['probabilities'];line=(p[...,1]>=.25)&(p[...,0]>=.4)&bev['observed']
    points=[];bounds=[];previous=0.;previous_x=None
    for x in np.arange(config['x_min_m'],config['x_max_m']+.01,.5):
        yy=xyz[...,1][line&(np.abs(xyz[...,0]-x)<.3)]
        if len(yy)<4:continue
        ys=np.sort(yy);chunks=np.split(ys,np.where(np.diff(ys)>.25)[0]+1);centers=[float(np.median(c)) for c in chunks if len(c)>=2]
        pairs=[]
        for a in centers:
            for b in centers:
                if not config['minimum_width_m']<=b-a<=config['maximum_width_m']:continue
                mid=(a+b)/2
                if previous_x is None and not a<0<b:continue
                if previous_x is not None and (x-previous_x>config.get('maximum_mark_gap_m',3) or abs(mid-previous)>.5*(x-previous_x)+.2):continue
                pairs.append((abs(mid-previous),a,b))
        if not pairs:continue
        _,a,b=min(pairs);previous=(a+b)/2;previous_x=x;points.append([float(x),previous]);bounds.append([a,b])
    if (len(points)<8 or points[-1][0]-points[0][0]<config['minimum_span_m']) and config.get('allow_single_boundary_prior',False):
        return single_boundary_center(bev,grid,config)
    return finish(points,bounds,config,'paired_predicted_lane_marks_ground_plane')

def finish(points,bounds,config,source):
    if len(points)<8 or points[-1][0]-points[0][0]<config['minimum_span_m']:
        return {'status':'unavailable','reason':'insufficient_paired_visible_lane_evidence','points_ego_m':[],'control_issued':False}
    points=np.asarray(points);coeff=np.polyfit(points[:,0],points[:,1],2);residual=float(np.sqrt(np.mean((np.polyval(coeff,points[:,0])-points[:,1])**2)))
    if residual>.35:return {'status':'unavailable','reason':'inconsistent_lane_pair','points_ego_m':[],'control_issued':False}
    samples=np.arange(points[0,0],points[-1,0]+.01,.5)
    return {'status':'candidate','source':source,'points_ego_m':np.stack([samples,np.polyval(coeff,samples)],1).tolist(),
            'observed_pairs':len(points),'span_m':float(points[-1,0]-points[0,0]),'fit_rmse_m':residual,'median_width_m':float(np.median(np.diff(bounds,axis=1))),
            'control_issued':False,'limits':'Visible local center candidate; no destination routing, traffic rules or closed-loop proof'}

def single_boundary_center(bev,grid,config):
    xyz=grid.points();p=bev['probabilities'];line=(p[...,1]>=.25)&(p[...,0]>=.4)&bev['observed'];road=(p[...,0]>=.5)&bev['observed']
    width=config['width_prior_m'];points=[];bounds=[];last_x=None;last_mid=0.;last_boundary=None;last_side=None
    for x in np.arange(config['x_min_m'],config['x_max_m']+.01,.5):
        row=abs(xyz[...,0]-x)<.3;ys=np.sort(xyz[...,1][line&row]);chunks=np.split(ys,np.where(np.diff(ys)>.25)[0]+1)
        choices=[]
        for chunk in chunks:
            if len(chunk)<2:continue
            boundary=float(np.median(chunk))
            for side in [-1,1]:
                mid=boundary+side*width/2
                if last_x is None and abs(mid)>1.5:continue
                if last_x is not None and (x-last_x>config.get('maximum_mark_gap_m',3) or side!=last_side or abs(mid-last_mid)>.5*(x-last_x)+.2):continue
                zone=row&(abs(xyz[...,1]-mid)<width/2-.25)
                if not zone.any() or np.mean(road[zone])<config['minimum_road_support']:continue
                cost=abs(mid-last_mid)+(0 if last_boundary is None else .2*abs(boundary-last_boundary))
                choices.append((cost,mid,boundary,side))
        if not choices:continue
        _,mid,boundary,side=min(choices);points.append([float(x),mid]);bounds.append([mid-width/2,mid+width/2]);last_x=x;last_mid=mid;last_boundary=boundary;last_side=side
    result=finish(points,bounds,config,'single_predicted_boundary_with_fixed_width_prior')
    if result['status']=='candidate':result.update({'width_prior_m':width,'assumption':'One painted boundary, fixed lane width and predicted ground support; width is not measured in this mode'})
    return result
