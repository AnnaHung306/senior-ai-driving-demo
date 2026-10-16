"""One-time crop and real-detector-box heldout phase evaluation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from train_light_state import ROOT,LightCrops,LightStateNet,evaluate,quality
from train_sunny import require_idle_gpu,configure_cuda
from senior_ai.core import write_json
from senior_ai.sunny_data import png,sha,verify
from senior_ai.light_model import crop_rgb
from senior_ai.light_labels import LIGHT_STATES
from senior_ai.detection_metrics import iou
from senior_ai.dataset import safe_path


@torch.inference_mode()
def joint(model,root,frames):
    matrix=np.zeros((4,5),dtype=np.int64);records=[]
    for row in frames:
        targets=[t for t in row['targets'] if t['label']==7];matched=set()
        predictions=[p for p in row['predictions'] if p['label']==7 and p['score']>=.5]
        if not targets and not predictions: continue
        rgb=png(safe_path(root,row['rgb']))
        for prediction in sorted(predictions,key=lambda p:-p['score']):
            crop=crop_rgb(rgb,prediction['box']).cuda().float()/255;pred=int(model(crop[None]).argmax(1)[0])
            overlaps=iou(prediction['box'],[t['box'] for t in targets]);options=[i for i,v in enumerate(overlaps) if v>=.5 and i not in matched]
            if options:
                best=max(options,key=lambda i:overlaps[i]);matched.add(best);truth=targets[best]['light_state_class'];matrix[truth,pred]+=1
                records.append({'scene_id':row['scene_id'],'camera':row['camera'],'frame':row['frame'],'rgb':row['rgb'],
                                'box':prediction['box'],'target':truth,'prediction':pred,'matched':True})
            else:
                records.append({'scene_id':row['scene_id'],'camera':row['camera'],'frame':row['frame'],'rgb':row['rgb'],
                                'box':prediction['box'],'prediction':pred,'matched':False})
        for index,t in enumerate(targets):
            if index not in matched:
                truth=t['light_state_class'];matrix[truth,4]+=1
                records.append({'scene_id':row['scene_id'],'camera':row['camera'],'frame':row['frame'],'rgb':row['rgb'],
                                'box':t['box'],'target':truth,'prediction':None,'matched':False})
    classes=[{'name':name,'gt_objects':int(matrix[c].sum()),'correct_state_recall_including_missed_detections':float(matrix[c,c]/matrix[c].sum()) if matrix[c].sum() else None} for c,name in enumerate(LIGHT_STATES)]
    values=[r['correct_state_recall_including_missed_detections'] for r in classes if r['gt_objects']]
    return {'confusion_true_rows_phase_prediction_columns_plus_missed':matrix.tolist(),'classes':classes,
            'macro_correct_state_recall':float(np.mean(values)) if values else None,
            'unmatched_detector_light_predictions':sum('target' not in r for r in records)},records


def main(args):
    state=json.loads((args.run/'state.json').read_text(encoding='utf-8'))
    if state['status']!='completed' or not state['validation_quality']['passed']: raise ValueError('Validation-qualified phase checkpoint required')
    if sha(args.run/'best.pt')!=state['best_sha256']: raise ValueError('Phase checkpoint changed')
    saved=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False);config=saved['config']
    if verify(args.root)!=saved['data']: raise ValueError('Phase data changed')
    for relative,digest in saved['source'].items():
        if sha(ROOT/relative)!=digest: raise ValueError('Frozen phase source changed')
    if (args.run/'lockbox_started.json').exists(): raise FileExistsError('Phase heldout already opened')
    for split in ['development','lockbox']:
        if not (args.detector_run/(split+'_frames.jsonl')).exists(): raise ValueError('Completed detector heldout predictions required')
    write_json(args.run/'lockbox_started.json',{'best_sha256':state['best_sha256'],'detector_run':str(args.detector_run)})
    require_idle_gpu();configure_cuda();model=LightStateNet().cuda().eval();model.load_state_dict(saved['model']);result={}
    for split in ['development','lockbox']:
        data=LightCrops(args.root,split,config,allow_lockbox=split=='lockbox')
        if not data.approved: raise ValueError('Heldout phase coverage missing')
        cropped,crop_frames=evaluate(model,data,config['batch_size']);write_json(args.run/(split+'_crops.json'),cropped)
        (args.run/(split+'_crop_frames.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in crop_frames),encoding='utf-8')
        detector_frames=[json.loads(line) for line in (args.detector_run/(split+'_frames.jsonl')).read_text(encoding='utf-8').splitlines()]
        combined,joint_frames=joint(model,args.root,detector_frames);write_json(args.run/(split+'_joint.json'),combined)
        (args.run/(split+'_joint_frames.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in joint_frames),encoding='utf-8')
        result[split]={'crop_quality':quality(cropped,config),'joint_quality':(combined['macro_correct_state_recall'] or 0)>=config['minimum_joint_macro_correct_state_recall']}
    final={'status':'passed' if result['lockbox']['crop_quality']['passed'] and result['lockbox']['joint_quality'] else 'quality_failed',
           'quality':result,'checkpoint_sha256':state['best_sha256'],'scope':'Observable lamp color eligibility; unknown class for back-facing/unresolvable lamps; ClearNoon only'}
    write_json(args.run/'final_evaluation.json',final);print(json.dumps(final));return 0 if final['status']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--detector-run',type=Path,required=True);sys.exit(main(p.parse_args()))
