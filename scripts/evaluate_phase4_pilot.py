"""Independent GT evaluator; consumes completed predictions, never supplies inference inputs."""
import argparse,json,sys,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.bev_geometry import Grid
from senior_ai.core import write_json
from select_phase4_pilot import gt_points,read,sha

def ratio(a,b):return float(a/b) if b else None
def road_counts(p,observed,truth,domain):
    predicted=(p[...,0]>=.5)&observed;return {'TP':int((predicted&truth&domain).sum()),'FP':int((predicted&~truth&domain).sum()),'FN':int((~predicted&truth&domain).sum()),'GT_visible_ground_cells':int(domain.sum()),'predicted_observed_cells_on_GT_domain':int((observed&domain).sum())}
def line_counts(p,observed,truth,domain):
    predicted=(p[...,1]>=.25)&observed&domain;target=truth&domain
    near_t=cv2.dilate(target.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
    near_p=cv2.dilate(predicted.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
    return {'GT_line_cells':int(target.sum()),'predicted_line_cells':int(predicted.sum()),'matched_predicted':int((predicted&near_t).sum()),'matched_GT':int((target&near_p).sum())}
def finalize(counts):
    p=ratio(counts['matched_predicted'],counts['predicted_line_cells']);r=ratio(counts['matched_GT'],counts['GT_line_cells'])
    return {**counts,'road_IoU':ratio(counts['TP'],counts['TP']+counts['FP']+counts['FN']),'GT_ground_coverage':ratio(counts['predicted_observed_cells_on_GT_domain'],counts['GT_visible_ground_cells']),
            'line_precision':p,'line_recall':r,'line_F1':2*p*r/(p+r) if p is not None and r is not None and p+r else 0.}

def reference(root,manifest,grid):
    counts=np.zeros(grid.shape+(3,),np.int32);ego=None;label=None;front_visible=np.zeros(grid.shape,bool)
    for name,xyz,semantic,ego,label in gt_points(root,manifest,1):
        valid=(abs(xyz[...,2])<=.2)&np.isin(semantic,[1,2,10,24,25]);uv=grid.xy_to_uv(xyz[...,:2]);col=np.floor(uv[...,0]+.5).astype(np.int32);row=np.floor(uv[...,1]+.5).astype(np.int32)
        valid&=(row>=0)&(row<grid.shape[0])&(col>=0)&(col<grid.shape[1]);valid&=~((abs(xyz[...,0])<2.5)&(abs(xyz[...,1])<1.1))
        if name=='front':front_visible[row[valid],col[valid]]=True
        ground_road=np.isin(semantic,[1,24]);line=semantic==24
        for k,mask in enumerate([valid&ground_road,valid&~ground_road,valid&line]):np.add.at(counts[...,k],(row[mask],col[mask]),1)
    domain=counts[...,:2].sum(-1)>0;road=counts[...,0]>counts[...,1];line=(counts[...,2]>0)&road
    return domain,road,line,ego,label,front_visible

def iou(a,b):
    left=max(a[0],b[0]);top=max(a[1],b[1]);right=min(a[2],b[2]);bottom=min(a[3],b[3]);area=max(0,right-left)*max(0,bottom-top)
    return area/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)

def object_metrics(root,metadata,world_from_ego,label):
    inverse=np.linalg.inv(world_from_ego);actors={};target_by_camera={};gt_kind={1:'car',2:'pedestrian',3:'bicycle',4:'motorcycle',5:'truck',6:'bus',8:'temporary_obstacle'}
    scene=metadata['scene_id'];frame=metadata['frame']
    for name in ['front','front_right','right','rear_right','rear','rear_left','left','front_left']:
        targets=read(root/f'labels/visible/{scene}/{name}/{frame:08}.json')['positive_objects'];target_by_camera[name]=targets
        for t in targets:
            if t['category_id'] in gt_kind:actors[t['actor_id']]=gt_kind[t['category_id']]
    truth={}
    for a in label['objects_3d']:
        if a['actor_id'] not in actors:continue
        xy=(inverse@np.r_[a['world_pose']['position_m'],1])[:2]
        if np.linalg.norm(xy)<=24:truth[a['actor_id']]={'xy_m':xy,'class_name':actors[a['actor_id']]}
    predicted=[o for o in metadata['objects'] if np.linalg.norm(o['xy_m'])<=24]
    pairs=sorted((np.linalg.norm(np.asarray(o['xy_m'])-t['xy_m']),i,key) for i,o in enumerate(predicted) for key,t in truth.items() if o['class_name']==t['class_name'])
    used_p=set();used_t=set();distances=[]
    for dist,i,key in pairs:
        if dist>2 or i in used_p or key in used_t:continue
        used_p.add(i);used_t.add(key);distances.append(float(dist))
    incorrect=0;duplicate_support=0;removed=metadata['projected_before_dedup']-metadata['objects_after_dedup']
    for obj in metadata['objects']:
        identities=set();matched=0
        for source in obj['sources']:
            options=[(iou(source['bbox_xyxy'],t['visible_bbox_xyxy']),t['actor_id']) for t in target_by_camera[source['camera']] if t['category_id']==source['class_id']]
            if options and max(options)[0]>=.5:identities.add(max(options)[1]);matched+=1
        if len(identities)>1:incorrect+=1
        if len(identities)==1 and matched>1:duplicate_support+=1
    return {'TP':len(used_p),'FP':len(predicted)-len(used_p),'FN':len(truth)-len(used_t),'matched_distances_m':distances,'duplicate_observations_removed':removed,'GT_supported_merged_clusters':duplicate_support,'mixed_GT_actor_merges':incorrect}

def path_error(root,metadata,ego,label):
    if metadata['path']['status']!='candidate':return None
    inverse=np.linalg.inv(ego);roads=read(root/label['roads'])['waypoints'];all_points=[]
    for wp in roads:
        p=(inverse@np.r_[wp['pose']['position_m'],1])[:2];yaw=np.deg2rad(wp['pose']['rotation_deg'][1]);heading=inverse[:2,:2]@np.array([np.cos(yaw),np.sin(yaw)])
        if heading[0]>.8 and not wp['is_junction']:all_points.append((wp,p))
    near=[x for x in all_points if -3<x[1][0]<6 and abs(x[1][1])<3]
    if not near:return None
    # OpenDRIVE road_id may change on one continuous lane. Trace heading-compatible
    # GT segments from ego, independently of the predicted path and without extrapolation.
    anchor=min(near,key=lambda x:np.linalg.norm(x[1]));trace=[anchor];previous=anchor
    while True:
        w,p=previous;yaw=np.deg2rad(w['pose']['rotation_deg'][1]);heading=inverse[:2,:2]@np.array([np.cos(yaw),np.sin(yaw)])
        candidates=[]
        for other,q in all_points:
            dx=q[0]-p[0]
            if not 1<dx<=7.5 or q[0]>25:continue
            lateral=abs(q[1]-(p[1]+dx*heading[1]/max(.1,heading[0])))
            if lateral<1:candidates.append((lateral+.02*dx,other,q))
        if not candidates:break
        _,w,q=min(candidates,key=lambda x:x[0]);previous=(w,q);trace.append(previous)
    points=np.array([p for w,p in trace]);key=(anchor[0]['road_id'],anchor[0]['lane_id'])
    if len(points)<3 or np.ptp(points[:,0])<5:return None
    points=points[np.argsort(points[:,0])];pred=np.asarray(metadata['path']['points_ego_m']);valid=(pred[:,0]>=points[:,0].min())&(pred[:,0]<=points[:,0].max())
    if valid.sum()<4:return None
    errors=pred[valid,1]-np.interp(pred[valid,0],points[:,0],points[:,1])
    return {'lateral_rmse_m':float(np.sqrt(np.mean(errors**2))),'lateral_absolute_p95_m':float(np.quantile(abs(errors),.95)),'road_id':int(key[0]),'lane_id':int(key[1]),'points':int(valid.sum())}

def main(args):
    run=args.run.resolve();root=args.root.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('New evaluation output required')
    state=read(run/'state.json')
    if state['status']!='completed_inference_pending_evaluation_and_visual_review':raise ValueError('Completed prediction required')
    out.mkdir(parents=True);config=read(run/'config.json');grid=Grid(**config['grid']);contract=read(run/'contract.json');selection=read(args.selection/'selection.json')
    if sha(root/'audit.json')!=selection['dataset_audit_sha256']:raise ValueError('Different GT dataset')
    if sha(ROOT/'scripts/select_phase4_pilot.py')!=selection['tool_sha256']:raise ValueError('GT projection helper differs from selection version')
    if sha(ROOT/'src/senior_ai/bev_geometry.py')!=contract['sources']['src/senior_ai/bev_geometry.py']:raise ValueError('GT metric grid/geometry differs from frozen inference')
    golden=read(ROOT/'artifacts/runs/sunny-segmentation-v6-001/provenance.json')['data']
    if sha(root/'metadata_inventory.json')!=golden['metadata_inventory_sha256'] or sha(root/'image_inventory.jsonl')!=golden['image_inventory_sha256']:raise ValueError('GT inventories differ from frozen parent audit')
    image_inventory={r['path']:r['sha256'] for r in [json.loads(l) for l in (root/'image_inventory.jsonl').read_text().splitlines()]};meta_inventory={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')};verified={}
    for rel,digest in contract['sources'].items():
        if sha(run/'source'/rel)!=digest:raise ValueError('Frozen inference snapshot changed')
    totals={name:Counter() for name in ['front','single','temporal']};common={name:Counter() for name in totals};scenes=defaultdict(lambda:{n:Counter() for n in totals});records=[];last={};object_total=Counter();distances=[]
    for item in state['records']:
        folder=run/item['folder'];metadata=read(folder/'frame.json');manifest=metadata['manifest'];input_frame=read(root/manifest);label_path=f"labels/frames/{item['scene_id']}/{item['frame']:08}.json";label=read(root/label_path)
        for relative in [manifest,input_frame['calibration'],label_path,label['roads']]+[f"labels/visible/{item['scene_id']}/{camera}/{item['frame']:08}.json" for camera in input_frame['cameras']]+[entry['image'] for modality in ['depth','semantic_segmentation'] for entry in label[modality].values()]:
            if relative not in verified:
                digest=sha(root/relative);expected=image_inventory.get(relative,meta_inventory.get(relative))
                if digest!=expected:raise ValueError('GT source changed: '+relative)
                verified[relative]=digest
        domain,truth,line,ego,label,front_domain=reference(root,manifest,grid);arrays=np.load(folder/'bev.npz');metrics={}
        for name in totals:
            observed=arrays['front_observed'] if name=='front' else arrays['observed'];counts={**road_counts(arrays[name],observed,truth,domain),**line_counts(arrays[name],observed,line,domain)}
            totals[name].update(counts);scenes[item['scene_id']][name].update(counts);metrics[name]=finalize(counts)
            common[name].update({**road_counts(arrays[name],observed,truth,domain&front_domain),**line_counts(arrays[name],observed,line,domain&front_domain)})
        motion_error=None
        if metadata['motion']['status']=='aligned' and item['scene_id'] in last:
            gt=np.linalg.inv(ego)@last[item['scene_id']];pred=np.asarray(metadata['motion']['previous_to_current']);a=np.arctan2(gt[1,0],gt[0,0]);b=np.arctan2(pred[1,0],pred[0,0]);angle=np.arctan2(np.sin(a-b),np.cos(a-b))
            motion_error={'translation_error_m':float(np.linalg.norm(gt[:2,3]-pred[:2,2])),'yaw_error_deg':float(abs(np.rad2deg(angle)))}
        last[item['scene_id']]=ego;objects=object_metrics(root,metadata,ego,label);distances+=objects.pop('matched_distances_m');object_total.update(objects)
        path=path_error(root,metadata,ego,label)
        record={**item,'metrics':metrics,'motion_error':motion_error,'path_GT_error':path,'object_metrics':objects};records.append(record)
        target=out/item['folder'];target.mkdir(parents=True);np.savez_compressed(target/'GT_reference.npz',visible_ground=domain,road=truth,line=line)
        panel=Image.new('RGB',(1200,540),(15,15,15));draw=ImageDraw.Draw(panel)
        rgb=np.zeros(grid.shape+(3,),np.uint8);rgb[domain]=[60,60,60];rgb[truth&domain]=[85,110,145];rgb[line&domain]=[255,225,40];panel.paste(Image.fromarray(rgb),(0,30))
        for x,name in [(400,'single'),(800,'temporal')]:
            p=arrays[name];rgb=np.full(grid.shape+(3,),20,np.uint8);rgb[domain]=[60,60,60];rgb[domain&(p[...,0]>=.5)&arrays['observed']]=[85,110,145];rgb[domain&(p[...,1]>=.25)&arrays['observed']]=[255,225,40];panel.paste(Image.fromarray(rgb),(x,30))
        for x,title in [(0,'GT DEPTH GROUND REFERENCE'),(400,'RGB 8CAM SINGLE ON GT DOMAIN'),(800,'RGB CAUSAL ON GT DOMAIN')]:draw.text((x+4,6),title,fill='white')
        panel.save(target/'evaluation.png')
    accepted=[r['motion_error'] for r in records if r['motion_error']];paths=[r['path_GT_error'] for r in records if r['path_GT_error']]
    report={'status':'completed_pilot_evaluation_pending_visual_review','scope':'Fixed32 original train/validation bundles, not independent heldout. GT visible near-plane ground only; unknown prediction counts against GT road. No GT used by inference.',
        'metrics':{n:finalize(v) for n,v in totals.items()},'common_front_GT_visible_domain':{n:finalize(v) for n,v in common.items()},'ablation_scope':'Full-surround domain comparison includes coverage gain from eight views; common-front GT-visible domain separately compares same-FOV quality.', 'by_scene':{s:{n:finalize(v) for n,v in modes.items()} for s,modes in scenes.items()},
        'motion':{'accepted':len(accepted),'possible':len(records)-len(scenes),'translation_mean_m':float(np.mean([v['translation_error_m'] for v in accepted])) if accepted else None,'yaw_mean_deg':float(np.mean([v['yaw_error_deg'] for v in accepted])) if accepted else None},
        'path':{'candidates':sum(r['path_status']=='candidate' for r in records),'bundles':len(records),'GT_comparable':len(paths),'lateral_RMSE_mean_m':float(np.mean([v['lateral_rmse_m'] for v in paths])) if paths else None},
        'objects':{**object_total,'precision_at_2m':ratio(object_total['TP'],object_total['TP']+object_total['FP']),'recall_at_2m':ratio(object_total['TP'],object_total['TP']+object_total['FN']),'matched_mean_distance_m':float(np.mean(distances)) if distances else None,'limits':'GT positive visible actors within24m, class-aware one-to-one2m match; ground-contact hypothesis vs actor origin, not3D box accuracy. Mixed-actor merge check only on IoU50-supported sources.'},
        'timing_ms':{key:{'p50':float(np.quantile([r[key] for r in records],.5)),'p95':float(np.quantile([r[key] for r in records],.95))} for key in ['inference_ms','fusion_ms']},
        'line_tolerance_m':.25,'line_raster_note':'5x5 elliptical dilation at.1m grid; offsets <=sqrt(5)*.1m, within declared.25m bound. GT forward depth raster, no filled holes.',
        'prediction_contract_sha256':sha(run/'contract.json'),'selection_sha256':sha(args.selection/'selection.json'),'evaluator_sha256':sha(Path(__file__)),'GT_helper_sha256':sha(ROOT/'scripts/select_phase4_pilot.py'),'verified_GT_files':len(verified),'records':records}
    write_json(out/'report.json',report);write_json(out/'GT_input_inventory.json',verified)
    print(json.dumps({k:v for k,v in report.items() if k in ['metrics','motion','path','objects','timing_ms']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
