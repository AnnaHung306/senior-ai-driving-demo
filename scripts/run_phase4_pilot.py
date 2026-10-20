"""Actual RGB-only frozen-model BEV replay. No labels/depth/world-pose imports."""
import argparse,json,hashlib,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.dataset import safe_path,read_sensor_frame
from senior_ai.core import write_json
from senior_ai.bev_geometry import Grid,cameras_from_calibration
from senior_ai.bev_fusion import fuse_road,ground_objects,deduplicate
from senior_ai.bev_temporal import CausalRoadFusion
from senior_ai.bev_path import lane_center
from senior_ai.sunny_data import DETECTION_CLASSES
from senior_ai.light_model import LightStateNet,crop_rgb
from senior_ai.light_labels import LIGHT_STATES
from predict_sunny import load_model

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def render_bev(bev,grid,path=None,objects=()):
    p=bev['probabilities'];rgb=np.full(grid.shape+(3,),24,np.uint8);seen=bev['observed'];rgb[seen]=[65,65,65];rgb[seen&(p[...,0]>.5)]=[85,110,145];rgb[seen&(p[...,1]>.25)]=[255,225,40]
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    if path and path['points_ego_m']:
        uv=grid.xy_to_uv(np.asarray(path['points_ego_m']));draw.line([tuple(x) for x in uv],fill=(0,255,255),width=2)
    for obj in objects:
        u,v=grid.xy_to_uv(obj['xy_m']);draw.ellipse((u-3,v-3,u+3,v+3),fill=(255,90,70));draw.text((u+3,v),obj['class_name'][:3],fill='white')
    u,v=grid.xy_to_uv([0,0]);draw.rectangle((u-9,v-23,u+9,v+23),outline='white',width=2)
    for x in [0,10,20,30]:
        _,row=grid.xy_to_uv([x,0]);draw.text((4,row),f'{x}m',fill='white')
    return im

def main(args):
    root=args.root.resolve();out=args.output.resolve();spec=read(args.input_list);config=read(args.config)
    if out.exists():raise ValueError('New run output required')
    if set(spec)!={'schema_version','scope','inventory_sha256','sequences'} or spec['schema_version']!=1:raise ValueError('Strict RGB sequence input schema required')
    inventory=read(args.inventory)
    if sha(args.inventory)!=spec['inventory_sha256']:raise ValueError('RGB inventory changed')
    actual={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file()}
    if actual!=set(inventory) or any(not x.startswith(('inputs/','calibration/')) for x in actual):raise ValueError('Staging must contain only frozen RGB inputs and calibration')
    for relative,digest in inventory.items():
        if sha(safe_path(root,relative))!=digest:raise ValueError('RGB staging bytes changed: '+relative)
    for m in config['models'].values():
        if sha(ROOT/m['path'])!=m['sha256']:raise ValueError('Baseline checkpoint changed')
    for seq in spec['sequences']:
        if set(seq)!={'scene_id','manifests'}:raise ValueError('GT fields forbidden in sequence input')
    out.mkdir(parents=True);args.created=True;source={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    for p in [Path(__file__),ROOT/'scripts/predict_sunny.py',ROOT/'scripts/perception_inputs.py',ROOT/'scripts/train_sunny.py',ROOT/'configs/phase4_bev_v1.json']:source[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    write_json(out/'contract.json',{'config_sha256':sha(args.config),'input_list_sha256':sha(args.input_list),'RGB_inventory_sha256':sha(args.inventory),'sources':source,'models':config['models'],'inference_file_scope':'Only staged RGB/calibration + fixed model files. GT is not in this process input root.','control_issued':False})
    write_json(out/'config.json',config)
    for name in source:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.device=='cuda':
        from train_sunny import require_idle_gpu
        require_idle_gpu()
    segment,kind,_=load_model(ROOT/config['models']['segmentation']['path'],args.device)
    if kind!='segmentation':raise ValueError('Segmentation task required')
    detector,kind,_=load_model(ROOT/config['models']['detection']['path'],args.device)
    if kind!='detection':raise ValueError('Detection task required')
    phase=LightStateNet().to(args.device).eval();phase.load_state_dict(torch.load(ROOT/config['models']['phase']['path'],map_location='cpu',weights_only=False)['model'])
    names={v:k for k,v in DETECTION_CLASSES.items()};grid=Grid(**config['grid']);records=[]
    with torch.inference_mode():
        for seq in spec['sequences']:
            temporal=CausalRoadFusion(grid,config);last=-float('inf')
            for index,manifest in enumerate(seq['manifests']):
                frame=read_sensor_frame(root,manifest)
                if frame['scene_id']!=seq['scene_id'] or frame['timestamp_s']<=last:raise ValueError('Mixed scene or nonmonotonic sequence')
                last=frame['timestamp_s'];folder=out/frame['scene_id']/str(frame['frame']);folder.mkdir(parents=True)
                cameras=cameras_from_calibration(read(safe_path(root,frame['calibration'])));observations={}
                if args.device=='cuda':torch.cuda.synchronize()
                started=time.perf_counter();rgb={name:np.asarray(Image.open(safe_path(root,c['image'])).convert('RGB')) for name,c in frame['cameras'].items()}
                if any(v.shape!=(360,640,3) for v in rgb.values()):raise ValueError('Frozen 640x360 RGB dimensions required')
                tensors=torch.stack([torch.from_numpy(v.transpose(2,0,1).copy()) for v in rgb.values()]).to(args.device).float()/255
                logits=segment(tensors)
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite segmentation')
                probs=logits.softmax(1);ids=logits.argmax(1).cpu().numpy().astype(np.uint8)
                ground=torch.stack([probs[:,config['road_ids']].sum(1),probs[:,config['line_id']]],-1).cpu().numpy()
                for i,name in enumerate(rgb):
                    detection=detector([tensors[i]])[0];objects=[]
                    if not torch.isfinite(detection['boxes']).all() or not torch.isfinite(detection['scores']).all():raise FloatingPointError('Nonfinite detections')
                    for box,label,score in zip(detection['boxes'].cpu().tolist(),detection['labels'].cpu().tolist(),detection['scores'].cpu().tolist()):
                        obj={'bbox_xyxy':box,'class_id':label,'class_name':names[label],'score':score}
                        if label==7 and score>=.5:
                            crop=crop_rgb(rgb[name],box).to(args.device).float()/255;lp=phase(crop[None]).softmax(1)[0];state=int(lp.argmax());obj['phase']=LIGHT_STATES[state]
                            if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite light state')
                        objects.append(obj)
                    observations[name]={'rgb':rgb[name],'ids':ids[i],'ground_probabilities':ground[i],'objects':objects}
                if args.device=='cuda':torch.cuda.synchronize()
                infer_ms=1000*(time.perf_counter()-started);fusion_start=time.perf_counter()
                fused=fuse_road(observations,cameras,grid,config);front=fuse_road(observations,cameras,grid,config,'front')
                smoothed,motion=temporal.update(fused,observations,cameras,frame['scene_id'],frame['timestamp_s'],sha(safe_path(root,frame['calibration'])))
                projected,rejected=ground_objects(observations,cameras,config);unique=deduplicate(projected,config['object_merge_distance_m']);path=lane_center(smoothed,grid,config['path'])
                fusion_ms=1000*(time.perf_counter()-fusion_start)
                np.savez_compressed(folder/'bev.npz',single=fused['probabilities'],temporal=smoothed['probabilities'],front=front['probabilities'],observed=fused['observed'],front_observed=front['observed'],support=fused['support'],geometric_coverage=fused['geometric_coverage'])
                np.savez_compressed(folder/'camera_observations.npz',**{n+'_ids':o['ids'] for n,o in observations.items()})
                write_json(folder/'detections.json',{n:o['objects'] for n,o in observations.items()})
                metadata={'source':'learned_RGB_prediction_with_calibrated_ground_plane','scene_id':frame['scene_id'],'frame':frame['frame'],'timestamp_s':frame['timestamp_s'],'manifest':manifest,'manifest_sha256':sha(safe_path(root,manifest)),
                    'motion':motion,'path':path,'objects':unique,'unprojected_objects':rejected,'projected_before_dedup':len(projected),'objects_after_dedup':len(unique),
                    'timing_ms':{'RGB_decode_and_neural_inference':infer_ms,'fusion_motion_path':fusion_ms},'control_issued':False}
                write_json(folder/'frame.json',metadata)
                panel=Image.new('RGB',(1280,910),(12,12,12));draw=ImageDraw.Draw(panel)
                for i,(n,image) in enumerate(rgb.items()):panel.paste(Image.fromarray(image).resize((320,180)),((i%4)*320,(i//4)*180));draw.text(((i%4)*320+3,(i//4)*180+3),n,fill='red')
                for x,b,label in [(0,front,'FRONT ONLY'),(440,fused,'8 CAM SINGLE'),(880,smoothed,'8 CAM CAUSAL')]:
                    panel.paste(render_bev(b,grid,path if x==880 else None,unique if x!=0 else ()),(x,390));draw.text((x+4,370),label,fill='white')
                draw.text((5,894),'UP=+x forward; RIGHT=+y; yellow=lane, cyan=path; dark=unknown; ground-contact objects, no 3D boxes',fill='white');panel.save(folder/'panel.png')
                record={'scene_id':frame['scene_id'],'frame':frame['frame'],'folder':str(folder.relative_to(out)),'motion_status':motion['status'],'path_status':path['status'],'inference_ms':infer_ms,'fusion_ms':fusion_ms};records.append(record)
                write_json(out/'progress.json',{'completed_bundles':len(records),'latest':record});print(json.dumps(record),flush=True)
    for relative,digest in source.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Implementation changed during run')
    write_json(out/'state.json',{'status':'completed_inference_pending_evaluation_and_visual_review','bundles':len(records),'sequences':len(spec['sequences']),'records':records,'control_issued':False,'GT_inputs_used':False,'source_integrity_exact':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input-list',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=p.parse_args();args.created=False
    try:main(args)
    except Exception as e:
        if args.created:write_json(args.output/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'no_auto_retry':True})
        raise
