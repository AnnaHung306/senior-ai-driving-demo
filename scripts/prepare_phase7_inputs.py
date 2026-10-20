"""Fixed longer RGB-only windows; never selects by perception or reads GT."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.dataset import read_sensor_frame,safe_path
from senior_ai.core import write_json

def main():
    root=ROOT/'data/sunny-cross-town-v3-007';out=ROOT/'artifacts/runs/phase7-inputs-v1-001'
    if out.exists():raise FileExistsError(out)
    config=read(ROOT/'configs/phase7_motion_v1.json');meta={x['path']:x['sha256'] for x in read(root/'metadata_inventory.json')}
    golden=read(ROOT/'artifacts/runs/sunny-segmentation-v6-001/provenance.json')['data']
    assert sha(root/'metadata_inventory.json')==golden['metadata_inventory_sha256']
    assert sha(root/'image_inventory.jsonl')==golden['image_inventory_sha256']
    images={x['path']:x['sha256'] for x in map(json.loads,(root/'image_inventory.jsonl').read_text(encoding='utf8').splitlines())}
    available={};splits={}
    for split in ['train','validation']:
        path=f'training/{split}.jsonl';assert sha(root/path)==meta[path]
        for x in map(json.loads,(root/path).read_text(encoding='utf8').splitlines()):
            available.setdefault(x['scene_id'],{})[x['frame']]=x['input_manifest'];splits[x['scene_id']]=split
    out.mkdir(parents=True);stage=out/'rgb_only';files={};seq=[];counts=[]
    for scene in config['selection']['scenes']:
        manifests=[p for _,p in sorted(available[scene].items())][config['selection']['start']:config['selection']['stop']]
        assert len(manifests)==60;seq.append({'scene_id':scene,'manifests':manifests});timestamps=[]
        for path in manifests:
            f=read_sensor_frame(root,path);timestamps.append(f['timestamp_s'])
            for relative in [path,f['calibration']]+[v['image'] for v in f['cameras'].values()]:
                if relative in files:continue
                p=safe_path(root,relative);digest=sha(p);assert digest=={**meta,**images}[relative]
                dest=safe_path(stage,relative);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);files[relative]=digest
        counts.append({'scene':scene,'split':splits[scene],'frames':len(manifests),'duration_s':timestamps[-1]-timestamps[0]})
    write_json(out/'rgb_only_inventory.json',files)
    write_json(out/'sequence_inputs.json',{'schema_version':1,'scope':'Phase7 fixed long RGB+rig only; no GT or future observations in each causal inference step',
        'inventory_sha256':sha(out/'rgb_only_inventory.json'),'sequences':seq})
    write_json(out/'selection.json',{'status':'fixed_before_model_replay','scenes':counts,'config_sha256':sha(ROOT/'configs/phase7_motion_v1.json'),
        'source_sha256':sha(Path(__file__)),'original_metadata_inventory_sha256':sha(root/'metadata_inventory.json'),
        'scope':'Known train/validation, no new heldout or wall-free driving claim','GT_read':False,'images':1920})
    print(json.dumps({'status':'RGB_staged','frames':240,'images':1920,'files':len(files),'scenes':counts}))

if __name__=='__main__':main()
