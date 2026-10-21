"""Frozen CPU light-head replay with observability-conditioned independent GT."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json
from senior_ai.instance_labels import decode_instance_rgb
from senior_ai.phase10_metrics import visible_targets
from senior_ai.light_labels import LIGHT_STATES,phase_target
from senior_ai.light_model import LightStateNet,crop_rgb
from senior_ai.detection_metrics import iou

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    cfg=read(ROOT/'configs/phase4_bev_v1.json')['models']['phase'];checkpoint=ROOT/cfg['path'];assert sha(checkpoint)==cfg['sha256']
    model=LightStateNet().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    logs=ROOT/'artifacts/reports/phase10-signal-observer-v1-001.jsonl';global_phases={}
    if logs.exists():
        for line in logs.read_text().splitlines():
            r=json.loads(line);global_phases[r['readback_frame']]={x['id']:x['state'] for x in r['lights']}
    crops=[];rows=[];excluded=0
    for folder in sorted((ROOT/'artifacts/runs').glob('phase10-drive-*')):
        if not (folder/'report.json').exists() or not (folder/'audit').exists():continue
        gts=[read(p) for p in sorted((folder/'audit').glob('[0-9]*.json'))];local={g['frame']+1:g.get('signal_phases_readback',{}) for g in gts}
        for g in gts:
            if not g['sampled_images']:continue
            before=local.get(g['frame']-1) or global_phases.get(g['frame']-1,{})
            after=local.get(g['frame']+1) or global_phases.get(g['frame']+1,{})
            before={int(k):v for k,v in before.items()};after={int(k):v for k,v in after.items()}
            for camera,item in g['cameras'].items():
                instance_path=folder/'audit'/f'{g["frame"]:08d}'/f'{camera}-instance.png'
                assert sha(instance_path)==item['instance_sha256'] and sha(folder/item['RGB']['file'])==item['RGB']['sha256']
                raw=np.array(Image.open(instance_path))
                tags,ids=decode_instance_rgb(raw);rgb=np.array(Image.open(folder/item['RGB']['file']))
                targets,_,_=visible_targets(raw,{x['actor_id'] for x in g['actors']})
                for target in targets:
                    if target['label']!=7:continue
                    identifier=target['rendered_instance_id']
                    if identifier not in before or identifier not in after or before[identifier]!=after[identifier]:excluded+=1;continue
                    gt=phase_target(rgb,(ids==identifier)&(tags==7),before[identifier])
                    row={'run_id':folder.name,'frame':g['frame'],'camera':camera,'rendered_lamp_id':identifier,
                         'truth_class':gt['light_state_class'],'truth_observability':gt,'gt_crop_index':len(crops),'det_crop_index':None}
                    crops.append(crop_rgb(rgb,target['box']))
                    candidates=[d for d in item['detections'] if d['label']==7 and d['score']>=.5]
                    overlap,d=max([(float(iou(t['box'],[target['box']])[0]),t) for t in candidates],key=lambda x:x[0],default=(0,None))
                    if overlap>=.5:row['det_crop_index']=len(crops);crops.append(crop_rgb(rgb,d['box']))
                    rows.append(row)
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(crops),64):predictions+=model(torch.stack(crops[start:start+64]).float()/255).argmax(1).tolist()
    ceiling=np.zeros((4,4),int);joint=np.zeros((4,5),int)
    for r in rows:
        gt=r['truth_class'];cp=predictions[r['gt_crop_index']];dp=predictions[r['det_crop_index']] if r['det_crop_index'] is not None else 4
        ceiling[gt,cp]+=1;joint[gt,dp]+=1;r.update(GT_crop_prediction=cp,detected_crop_prediction=dp)
    def metrics(m):
        totals=m.sum(1);return {'confusion':m.tolist(),'support':totals.tolist(),'per_class_recall':[(float(m[i,i]/totals[i]) if totals[i] else None) for i in range(4)],
            'macro_recall_supported':float(np.mean([m[i,i]/totals[i] for i in range(4) if totals[i]])) if totals.sum() else None}
    report={'status':'completed_frozen_CPU_light_replay','samples':len(rows),'excluded_no_bracketing_constant_phase':excluded,
        'state_names':LIGHT_STATES,'GT_crop_ceiling':metrics(ceiling),'detector_plus_state_joint':metrics(joint),
        'head_checkpoint_sha256':cfg['sha256'],'signal_log_sha256':sha(logs) if logs.exists() else None,
        'tool_sha256':sha(Path(__file__)),'formal_quality_passed':False,
        'limits':['CPU replay of frozen head; not asserted bit-exact with original CUDA inference.',
          'Only lamps with matching raw-instance ID and unchanged phase bracketing capture tick are scored.',
          'Uses pre-existing >=2 emissive-pixel observability target; back-facing/unresolvable = unknown.',
          'Joint confusion has extra final column for missing detector; macro over supported classes only, absent classes unmeasured.']}
    if crops:
        np.savez_compressed(a.output/'head_inputs.npz',RGB_uint8_NCHW=np.stack([x.numpy() for x in crops]))
        report['head_inputs_sha256']=sha(a.output/'head_inputs.npz')
    write_json(a.output/'report.json',report);write_json(a.output/'samples.json',rows);print(report)

if __name__=='__main__':main()
