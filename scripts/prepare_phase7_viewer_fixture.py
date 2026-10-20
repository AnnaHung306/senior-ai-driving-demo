"""A one-frame owned UI negative/recovery fixture; never modifies the real bundle."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.core import write_json

def main(a):
    source=ROOT/'artifacts/runs/phase7-viewer-bundle-v1-001';out=a.output
    if out.exists():raise FileExistsError(out)
    m=copy.deepcopy(read(source/'bundle.json'));row=m['frames'][6];f=read(source/row['file']);out.mkdir(parents=True)
    files=[row['file'],f['occupancy']['file']]+[c['file'] for c in f['cameras'].values()]
    for relative in files:
        p=out/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,p)
    f['future_observations_used']=not a.valid
    (out/row['file']).write_text(json.dumps(f,separators=(',',':'))+'\n',encoding='utf8')
    m['file_inventory']={relative:sha(out/relative) for relative in files};row['sha256']=m['file_inventory'][row['file']]
    m['frames']=[row];m['scenes']=[{'id':row['scene_id'],'frames':[row['key']],'duration_s':0.,'source_frequency_Hz':5.}]
    m.pop('synthetic_diagnostics');m['label']='Owned semantic-rejection fixture, not an inference run'
    write_json(out/'bundle.json',m);print({'fixture':str(out),'expected_client_acceptance':a.valid,'files':len(files)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--valid',action='store_true');main(p.parse_args())
