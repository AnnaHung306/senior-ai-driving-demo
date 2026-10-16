"""Validation only: evaluate the selected RGB phase head using actual detector boxes."""
import argparse
import json
from pathlib import Path
import torch
from evaluate_light_state import joint,joint_quality,error_gallery
from train_light_state import ROOT,LightStateNet
from train_sunny import configure_cuda,require_idle_gpu
from senior_ai.sunny_data import sha
from senior_ai.core import write_json


def main(args):
    phase_state=json.loads((args.run/'state.json').read_text(encoding='utf-8'));detector_state=json.loads((args.detector_run/'state.json').read_text(encoding='utf-8'))
    if phase_state['status']!='completed' or detector_state['status']!='completed':raise ValueError('Complete both training runs first')
    if not phase_state['validation_quality']['passed']:raise ValueError('GT-crop phase validation must qualify')
    for run,state in [(args.run,phase_state),(args.detector_run,detector_state)]:
        if sha(run/'best.pt')!=state['best_sha256']:raise ValueError('Checkpoint changed')
        if (run/'lockbox_started.json').exists():raise ValueError('Joint validation must precede heldout opening')
        if sha(run/'validation.json')!=state['validation_sha256'] or sha(run/'validation_frames.jsonl')!=state['validation_frames_sha256']:
            raise ValueError('Validation results changed after training')
    saved=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False)
    for relative,digest in saved['source'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Phase evaluation source differs')
    detector_data=json.loads((args.detector_run/'provenance.json').read_text(encoding='utf-8'))['data']
    if saved['data']!=detector_data or saved['data']['audit_sha256']!=sha(args.root/'audit.json'):raise ValueError('Models trained on different audit')
    require_idle_gpu();configure_cuda();model=LightStateNet().cuda().eval();model.load_state_dict(saved['model'])
    frames=[json.loads(line) for line in (args.detector_run/'validation_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    if any(r['split']!='validation' for r in frames):raise ValueError('Validation split only')
    result,records=joint(model,args.root,frames);write_json(args.run/'validation_joint.json',result)
    (args.run/'validation_joint_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    receipt={**joint_quality(result,saved['config']),'phase_sha256':phase_state['best_sha256'],'detector_sha256':detector_state['best_sha256'],
        'joint_report_sha256':sha(args.run/'validation_joint.json'),
        'data_audit_sha256':saved['data']['audit_sha256'],'source_sha256':sha(Path(__file__)),'lockbox_opened':False}
    write_json(args.run/'validation_joint_quality.json',receipt)
    crop_frames=[json.loads(line) for line in (args.run/'validation_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    output=args.run/'validation_phase_errors'
    if not output.exists():error_gallery(args.root,None,crop_frames,records,output)
    print(json.dumps(receipt));return 0 if receipt['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--detector-run',type=Path,required=True);p.add_argument('--root',type=Path,required=True);raise SystemExit(main(p.parse_args()))
