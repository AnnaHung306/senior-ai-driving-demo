"""GT-only scene selection BEFORE perception. Never opens model weights or heldout."""
import json,hashlib,sys,argparse,shutil
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.bev_geometry import Camera,OPTICAL
from senior_ai.dataset import safe_path,read_sensor_frame
from senior_ai.core import write_json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def gt_points(root,manifest,stride=4):
    frame=read_sensor_frame(root,manifest);label=read(safe_path(root,f"labels/frames/{frame['scene_id']}/{frame['frame']:08}.json"));calibration=read(safe_path(root,frame['calibration']));definitions={c['name']:c for c in calibration['cameras']}
    front=Camera(definitions['front']);world_from_ego=np.asarray(label['capture_world_from_camera']['front']['rgb'])@front.invT;ego_from_world=np.linalg.inv(world_from_ego)
    for name in frame['cameras']:
        camera=Camera(definitions[name]);sem=np.asarray(Image.open(safe_path(root,label['semantic_segmentation'][name]['image'])))[::stride,::stride,0]
        packed=np.asarray(Image.open(safe_path(root,label['depth'][name]['image'])))[::stride,::stride,:3].astype(np.uint32)
        depth=(packed[...,0]+256*packed[...,1]+65536*packed[...,2])*(1000./16777215)
        y,x=np.indices(sem.shape);uv=np.stack([x*stride+.5,y*stride+.5,np.ones_like(x)],-1);rays=uv@camera.invK.T@OPTICAL
        T=ego_from_world@np.asarray(label['capture_world_from_camera'][name]['depth']);xyz=(rays*depth[...,None])@T[:3,:3].T+T[:3,3]
        yield name,xyz,sem,world_from_ego,label

def assess(root,manifest):
    walls=0;lines=0;ground=[]
    for name,xyz,sem,_,_ in gt_points(root,manifest,4):
        valid=np.isfinite(xyz).all(-1)&(np.linalg.norm(xyz[...,:2],axis=-1)<30)
        corridor=valid&(xyz[...,0]>-10)&(xyz[...,0]<25)&(abs(xyz[...,1])<6)&(xyz[...,2]>-.5)&(xyz[...,2]<4)
        walls+=int(((sem==4)&corridor).sum())
        road=valid&np.isin(sem,[1,24])&(xyz[...,0]>2)&(xyz[...,0]<20)&(abs(xyz[...,1])<6)
        ground.extend(abs(xyz[...,2][road]).tolist())
        lines+=int(((sem==24)&road).sum())
    z=float(np.quantile(ground,.9)) if ground else None
    reasons=[]
    if walls:reasons.append('visible_near_wall_in_ego_corridor')
    if lines<12:reasons.append('insufficient_visible_lane_mark_pixels')
    if z is None or z>.2:reasons.append('ground_plane_residual_over_0.2m')
    return {'wall_samples_stride4':walls,'line_samples_stride4':lines,'road_abs_z_p90_m':z,'excluded_reasons':reasons}

def main(args):
    root=args.root.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('New selector output required')
    out.mkdir(parents=True);candidates=[];selected=[]
    # Fixed three windows / group, independent of all new model results.
    inventory={x['path']:x['sha256'] for x in read(root/'metadata_inventory.json')}
    for split in ['train','validation']:
        index=root/f'training/{split}.jsonl'
        if sha(index)!=inventory[f'training/{split}.jsonl']:raise ValueError('Canonical split index changed')
        frames={}
        for line in index.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);frames.setdefault(row['scene_id'],{})[row['frame']]=row['input_manifest']
        for scene,records in sorted(frames.items()):
            ordered=[records[k] for k in sorted(records)];chosen=None
            for start in [20,60,100]:
                manifests=ordered[start:start+8]
                if len(manifests)!=8:continue
                # Check all8 frames, not just the appealing middle frame.
                checks=[assess(root,m) for m in manifests];ok=all(not x['excluded_reasons'] for x in checks)
                item={'scene':scene,'original_split':split,'start_index':start,'manifests':manifests,'checks':checks,'eligible':ok};candidates.append(item)
                if chosen is None and ok:chosen=item
            if chosen is not None:selected.append(chosen)
    # Bounded first2 train and first2 validation groups; preserve every rejected candidate.
    selected=[x for split in ['train','validation'] for x in [v for v in selected if v['original_split']==split][:2]]
    write_json(out/'selection.json',{'status':'selected_before_model_inference','tool_sha256':sha(Path(__file__)),'dataset_audit_sha256':sha(root/'audit.json'),'criteria':'stride4 visible rawWall4 must be zero in x[-10,25]m,y±6m,z[-.5,4]m; road-z p90<=.2m; >=12 raw24 line samples, all8frames. Visible sample screen, not complete asset clearance. Fixed starts20/60/100, first eligible per scene; first2 groups per split.',
        'scope':'Prototype original train/validation only, no development/lockbox, no independent generalization claim. Selection uses GT, inference will not.','candidates':candidates,'selected':selected})
    # Standalone inference staging has no labels/GT pose/depth fields or files.
    stage=out/'rgb_only';stage.mkdir();files={}
    for item in selected:
        for manifest in item['manifests']:
            frame=read_sensor_frame(root,manifest)
            for relative in [manifest,frame['calibration']]+[x['image'] for x in frame['cameras'].values()]:
                if relative in files:continue
                source=safe_path(root,relative)
                if relative in inventory and sha(source)!=inventory[relative]:raise ValueError('Selected metadata changed')
                target=safe_path(stage,relative);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);files[relative]=sha(target)
    image_inventory={x['path']:x['sha256'] for x in [json.loads(l) for l in (root/'image_inventory.jsonl').read_text().splitlines()]}
    for relative,digest in files.items():
        if relative.startswith('inputs/rgb/') and image_inventory[relative]!=digest:raise ValueError('Selected RGB differs from original inventory')
    write_json(out/'rgb_only_inventory.json',files)
    print(json.dumps({'selected_sequences':len(selected),'bundles':8*len(selected),'candidates':len(candidates),'selection':str(out/'selection.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
