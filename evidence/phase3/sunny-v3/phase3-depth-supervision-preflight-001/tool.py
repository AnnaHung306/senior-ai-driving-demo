"""Bound 64-view TRAIN-only depth feasibility check; no model or GPU use."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from senior_ai.sunny_data import segmentation_target,SEGMENTATION_NAMES

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
out=Path(__file__).parent
if (out/'report.json').exists():raise FileExistsError('Preserve completed feasibility evidence')
root=ROOT/'data/sunny-cross-town-v3-007'
parent=ROOT/'artifacts/runs/sunny-segmentation-v6-001'
bound=read(parent/'provenance.json')['data']
for name,key in [('audit.json','audit_sha256'),('visual_review.json','review_sha256'),('image_inventory.jsonl','image_inventory_sha256'),('metadata_inventory.json','metadata_inventory_sha256')]:
    assert sha(root/name)==bound[key]
meta={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
assert sha(root/'training/train.jsonl')==meta['training/train.jsonl']
inventory={r['path']:r for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
selected={}
for row in map(json.loads,(root/'training/train.jsonl').read_text().splitlines()):
    assert row['split']=='train'
    selected.setdefault((row['group_id'],row['camera']),row)
assert len(selected)==64 and len({k[0] for k in selected})==8
contract={'scope':'64 fixed first train views, one per original train group/camera. No model inference, parameter changes or heldout scoring. Not a complete depth-training pipeline or full-data depth-quality claim.',
    'main_data':bound,'parent_config_sha256':sha(parent/'config.json'),'tool_sha256':sha(Path(__file__)),
    'selection':'First canonical training-index view for each of eight train groups and eight cameras, before reading depth values.',
    'tentative_valid_mask':'semantic != raw0 and raw11/sky, finite, depth > .1 and < 120 metres; proposed pilot mask, not frozen production range or semantic quality gate.',
    'packed_RGB_formula':'(R + 256*G + 65536*B)*1000/16777215; geometry convention remains axial as in original audited native object checks.'}
(out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
records=[];values={k:[] for k in ['wall','fence','terrain_or_ground']};total_valid=0;total_pixels=0;checked_files={}
def checked(rel):
    p=(root/rel).resolve();assert p.is_relative_to(root)
    item=inventory[rel];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    checked_files[rel]=item['sha256']
    with Image.open(p) as im:
        assert im.mode=='RGB' and im.size==(640,360) and im.format=='PNG'
        return np.array(im,dtype=np.uint8)
for key,row in sorted(selected.items()):
    manifest=row['input_manifest'];assert sha(root/manifest)==meta[manifest]
    m=read(root/manifest);assert m['scene_id']==row['scene_id'] and m['frame']==row['frame']
    calibration=m['calibration'];assert sha(root/calibration)==meta[calibration]
    c=next(c for c in read(root/calibration)['cameras'] if c['name']==row['camera'])
    k=np.asarray(c['K']);assert k.shape==(3,3) and k[0,0]>0 and k[1,1]>0 and np.isfinite(k).all()
    checked(row['rgb']);semantic=checked(row['semantic']);instance=checked(row['instance'])
    depth_rel=row['semantic'].replace('labels/semantic_segmentation/','labels/depth/')
    packed=checked(depth_rel).astype(np.uint32)
    depth=(packed[...,0]+256*packed[...,1]+65536*packed[...,2])*(1000./16777215)
    raw=semantic[...,0];valid=(raw!=0)&(raw!=11)&np.isfinite(depth)&(depth>.1)&(depth<120.)
    target=segmentation_target(raw,instance,row['verified_obstacle_actor_ids'])
    total_valid+=int(valid.sum());total_pixels+=valid.size
    for name in values:values[name].append(depth[valid&(target==SEGMENTATION_NAMES.index(name))])
    records.append({'sample':row,'depth':depth_rel,'depth_sha256':inventory[depth_rel]['sha256'],'valid_pixels':int(valid.sum()),'nonfinite_depth_pixels':int((~np.isfinite(depth)).sum()),'calibration':calibration,'calibration_sha256':meta[calibration]})
summary={}
for name,parts in values.items():
    x=np.concatenate(parts);summary[name]={'valid_pixels':int(x.size),'depth_quantiles_m':dict(zip(['p01','p50','p99'],np.quantile(x,[.01,.5,.99]).tolist())) if x.size else None}
report={'status':'passed_selected_train_depth_feasibility_only','views':len(records),'train_groups':8,'cameras_per_group':8,'checked_image_files':len(checked_files),'valid_pixels':total_valid,'total_pixels':total_pixels,'valid_fraction':total_valid/total_pixels,'depth_by_selected_GT_class':summary,'records':records,'checked_image_sha256':checked_files,'contract_sha256':sha(out/'contract.json'),'limits':'Selected train subset only; no class-prediction/Wall-IoU experiment, no evidence of depth predicting semantic identity, no normal targets or augmented-intrinsics compiler validated.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['records','checked_image_sha256']},ensure_ascii=False))
