"""One-time crop and real-detector-box heldout phase evaluation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw
from train_light_state import ROOT,LightCrops,LightStateNet,evaluate,quality
from train_sunny import require_idle_gpu,configure_cuda
from senior_ai.core import write_json
from senior_ai.sunny_data import png,sha,verify
from senior_ai.light_model import crop_rgb
from senior_ai.light_labels import LIGHT_STATES
from senior_ai.detection_metrics import iou
from senior_ai.dataset import safe_path
from audit_light_supplement import verify_supplement


@torch.inference_mode()
def joint_scores(matrix):
    classes=[{'name':name,'gt_objects':int(matrix[c].sum()),'correct_state_recall_including_missed_detections':float(matrix[c,c]/matrix[c].sum()) if matrix[c].sum() else None} for c,name in enumerate(LIGHT_STATES)]
    values=[r['correct_state_recall_including_missed_detections'] for r in classes if r['gt_objects']]
    return {'confusion_true_rows_phase_prediction_columns_plus_missed':matrix.tolist(),'classes':classes,
            'macro_correct_state_recall':float(np.mean(values)) if values else None,
            'all_phase_classes_present':bool((matrix.sum(1)>0).all()),
            'red_as_green_rate_all_GT_red':float(matrix[1,3]/matrix[1].sum()) if matrix[1].sum() else None,
            'red_as_green_rate_matched_red':float(matrix[1,3]/matrix[1,:4].sum()) if matrix[1,:4].sum() else None}


def joint_quality(result,config):
    checks={'all_classes':result['all_phase_classes_present'],
        'macro_correct_state_recall':(result['macro_correct_state_recall'] or 0)>=config['minimum_joint_macro_correct_state_recall'],
        'red_as_green':result['red_as_green_rate_matched_red'] is not None and result['red_as_green_rate_matched_red']<=config['max_red_as_green_rate']}
    return {'passed':all(checks.values()),'checks':checks}


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
    return {**joint_scores(matrix),'unmatched_detector_light_predictions':sum('target' not in r for r in records)},records


@torch.inference_mode()
def supplement_detector(checkpoint,root):
    from predict_sunny import load_model
    detector,task,_=load_model(checkpoint,'cuda')
    if task!='detection':raise ValueError('Detector checkpoint required')
    frames=[]
    for line in (root/'training/lockbox.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);rgb=png(safe_path(root,row['rgb']));image=torch.from_numpy(rgb.transpose(2,0,1).copy()).cuda().float()/255
        p=detector([image])[0];labels=json.loads(safe_path(root,row['detection']).read_text(encoding='utf-8'))
        frames.append({**row,'targets':[{'label':7,'box':p['visible_bbox_xyxy'],'light_state_class':p['light_state_class']} for p in labels['positive_objects']],
            'predictions':[{'box':b,'label':int(l),'score':float(s)} for b,l,s in zip(p['boxes'].cpu().tolist(),p['labels'].cpu().tolist(),p['scores'].cpu().tolist())]})
    del detector
    return frames


def error_gallery(root,supplement_root,crop_frames,joint_frames,output):
    """Real wrong/missed examples, stratified by true class and original route."""
    selected=[];seen=set()
    candidates=[{**r,'error_kind':'crop_misclassification'} for r in crop_frames if r['target']!=r['prediction']]
    candidates.extend({**r,'error_kind':'missed_detection' if r['prediction'] is None else 'joint_misclassification'} for r in joint_frames if 'target' in r and r['target']!=r['prediction'])
    for r in candidates:
        key=(r['error_kind'],r['scene_id'],r['target'],r['prediction'])
        if key not in seen:seen.add(key);selected.append(r)
    selected=sorted(selected,key=lambda r:(r['target'],r['scene_id'],r['error_kind']))[:32]
    output.mkdir(exist_ok=False);catalog=[]
    for i,r in enumerate(selected):
        data_root=Path(r['data_root']) if 'data_root' in r else supplement_root if r.get('controlled_supplement') else root
        rgb=Image.open(safe_path(data_root,r['rgb'])).convert('RGB');context=rgb.copy();b=r['box'];ImageDraw.Draw(context).rectangle(b,outline='yellow',width=2)
        context=context.resize((320,180));crop=rgb.crop(tuple(b));crop.thumbnail((144,180));scale=min(144/crop.width,180/crop.height);crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(480,220));panel.paste(context,(0,40));panel.paste(crop,(328,40));d=ImageDraw.Draw(panel)
        prediction='missed' if r['prediction'] is None else LIGHT_STATES[r['prediction']]
        d.text((4,2),f'GT {LIGHT_STATES[r["target"]]} / predicted {prediction}',fill='white');d.text((4,18),r['error_kind']+' '+r['scene_id'],fill='white')
        path=output/f'{i:02}.png';panel.save(path);catalog.append({**r,'image':path.name,'image_sha256':sha(path)})
    write_json(output/'catalog.json',{'total_crop_errors':sum(r['target']!=r['prediction'] for r in crop_frames),'selected':catalog,
        'selection':'First per error kind, source route, true and predicted phase; sorted by phase/route, max32. This is a gallery, not an independent dataset.'})


def main(args):
    state=json.loads((args.run/'state.json').read_text(encoding='utf-8'))
    if state['status']!='completed' or not state['validation_quality']['passed']: raise ValueError('Validation-qualified phase checkpoint required')
    if sha(args.run/'best.pt')!=state['best_sha256']: raise ValueError('Phase checkpoint changed')
    if sha(args.run/'validation.json')!=state['validation_sha256']:raise ValueError('Phase validation changed')
    saved=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False);config=saved['config']
    if verify(args.root)!=saved['data']: raise ValueError('Phase data changed')
    for relative,digest in saved['source'].items():
        if sha(ROOT/relative)!=digest: raise ValueError('Frozen phase source changed')
    if (args.run/'lockbox_started.json').exists(): raise FileExistsError('Phase heldout already opened')
    validation_joint=args.run/'validation_joint_quality.json'
    if not validation_joint.exists():raise ValueError('Validation detector-box phase check required before heldout')
    validation_joint=json.loads(validation_joint.read_text(encoding='utf-8'))
    if not validation_joint['passed'] or validation_joint['phase_sha256']!=state['best_sha256'] or validation_joint['detector_sha256']!=sha(args.detector_run/'best.pt'):
        raise ValueError('Joint validation failed or model pair changed')
    if validation_joint['joint_report_sha256']!=sha(args.run/'validation_joint.json'):raise ValueError('Joint validation report changed')
    supplement_signature=None
    if config.get('heldout_supplement_plan'):
        if args.supplement_root is None:raise ValueError('Frozen phase supplement required')
        if sha(args.supplement_root/'plan.json')!=config['heldout_supplement_plan_sha256']:raise ValueError('Different supplement plan')
        supplement_signature=verify_supplement(args.supplement_root,saved['data']['audit_sha256'])
    for split in ['development','lockbox']:
        if not (args.detector_run/(split+'_frames.jsonl')).exists(): raise ValueError('Completed detector heldout predictions required')
    detector_state=json.loads((args.detector_run/'state.json').read_text(encoding='utf-8'))
    detector_final=json.loads((args.detector_run/'final_evaluation.json').read_text(encoding='utf-8'))
    if detector_final['status']!='passed' or sha(args.detector_run/'best.pt')!=detector_state['best_sha256']:raise ValueError('Qualified detector required')
    # Coverage is checked before the irreversible evaluation marker. No model
    # predictions are made in this preflight and coverage never lowers a gate.
    for split in ['development','lockbox']:
        coverage=LightCrops(args.root,split,config,allow_lockbox=split=='lockbox',supplement_root=args.supplement_root if split=='lockbox' else None)
        if not coverage.approved:raise ValueError('Heldout phase coverage missing')
        del coverage
    write_json(args.run/'lockbox_started.json',{'best_sha256':state['best_sha256'],'detector_run':str(args.detector_run),
        'detector_sha256':detector_state['best_sha256'],'supplement':supplement_signature})
    require_idle_gpu();configure_cuda();model=LightStateNet().cuda().eval();model.load_state_dict(saved['model']);result={}
    for split in ['development','lockbox']:
        data=LightCrops(args.root,split,config,allow_lockbox=split=='lockbox',supplement_root=args.supplement_root if split=='lockbox' else None)
        if not data.approved: raise ValueError('Heldout phase coverage missing')
        cropped,crop_frames=evaluate(model,data,config['batch_size']);write_json(args.run/(split+'_crops.json'),cropped)
        (args.run/(split+'_crop_frames.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in crop_frames),encoding='utf-8')
        detector_frames=[json.loads(line) for line in (args.detector_run/(split+'_frames.jsonl')).read_text(encoding='utf-8').splitlines()]
        combined,joint_frames=joint(model,args.root,detector_frames)
        if split=='lockbox' and args.supplement_root is not None:
            write_json(args.run/'lockbox_dynamic_joint.json',combined)
            supplemental_frames=supplement_detector(args.detector_run/'best.pt',args.supplement_root)
            (args.run/'lockbox_supplement_detector_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in supplemental_frames),encoding='utf-8')
            extra,extra_frames=joint(model,args.supplement_root,supplemental_frames);write_json(args.run/'lockbox_controlled_joint.json',extra)
            matrix=np.array(combined['confusion_true_rows_phase_prediction_columns_plus_missed'])+np.array(extra['confusion_true_rows_phase_prediction_columns_plus_missed'])
            combined={**joint_scores(matrix),'unmatched_detector_light_predictions':combined['unmatched_detector_light_predictions']+extra['unmatched_detector_light_predictions']}
            joint_frames.extend({**r,'controlled_supplement':True} for r in extra_frames)
        write_json(args.run/(split+'_joint.json'),combined)
        (args.run/(split+'_joint_frames.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in joint_frames),encoding='utf-8')
        error_gallery(args.root,args.supplement_root,crop_frames,joint_frames,args.run/(split+'_phase_errors'))
        result[split]={'crop_quality':quality(cropped,config),'joint_quality':joint_quality(combined,config)}
    final={'status':'passed' if result['lockbox']['crop_quality']['passed'] and result['lockbox']['joint_quality']['passed'] else 'quality_failed',
           'quality':result,'checkpoint_sha256':state['best_sha256'],'supplement':supplement_signature,
           'scope':'Observable lamp color eligibility; unknown class for back-facing/unresolvable lamps; ClearNoon only. Dynamic and controlled stationary phase tests reported separately; pooled phase acceptance includes the pre-frozen coverage supplement.'}
    write_json(args.run/'final_evaluation.json',final);print(json.dumps(final));return 0 if final['status']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--detector-run',type=Path,required=True);p.add_argument('--supplement-root',type=Path);sys.exit(main(p.parse_args()))
