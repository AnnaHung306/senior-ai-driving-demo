"""Check and render unused-town probe RGB/GT; never create training approval."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from prepare_unused_scenery import validate_expansion_plan
from audit_unused_scenery import local_camera
from validate_dataset import read_png
from senior_ai.sunny_data import sha,segmentation_target
from senior_ai.segmentation_supplement import resolved_class_alignment
from senior_ai.dataset import safe_path
from senior_ai.core import write_json
from predict_sunny import PALETTE


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main(args):
    root=args.root.resolve();out=args.output.resolve();plan=read(root/'plan.json')
    if out.exists() or out.is_relative_to(root):raise ValueError('New separate preview output required')
    validate_expansion_plan(plan,archive_root=root,check_runtime=False)
    report=read(root/args.town/'report.json');rows=read(root/args.town/'records.json');rig=plan['rig']
    if report['status']!='passed' or not report['cleanup'] or not report['probe_only'] or report['plan_sha256']!=sha(root/'plan.json'):
        raise ValueError('Completed explicitly probe-only capture required')
    for name,digest in report['source_files'].items():
        if sha(root/args.town/'source_snapshot'/name)!=digest:raise ValueError('Capture source snapshot differs')
    if report['source_files']['src/senior_ai/sunny_data.py']!=sha(ROOT/'src/senior_ai/sunny_data.py'):raise ValueError('Mapping implementation differs')
    cameras={c['name']:c for c in rig['cameras']};anchors=plan['towns'][args.town]['anchors'];n=len(rows)//8
    if not 0<n<len(anchors) or len(rows)!=n*8 or len(rows)!=report['records']:raise ValueError('Strict complete probe subset required')
    expected={(a['spawn_index'],c):np.asarray(a['native_transform_matrix'])@local_camera(camera) for a in anchors[:n] for c,camera in cameras.items()}
    if {(r['anchor_spawn_index'],r['camera']) for r in rows}!=set(expected) or len(rows)!=len(expected):raise ValueError('Probe anchor-camera matrix differs')
    counts=np.zeros(17,dtype=np.int64);raw_counts=np.zeros(29,dtype=np.int64);files=[];cases=[];pose_error=0.;disagreement=0.
    for index,row in enumerate(rows):
        matrix=expected[(row['anchor_spawn_index'],row['camera'])];images={}
        if set(row['images'])!={'rgb','semantic','depth','instance'} or row['verified_obstacle_actor_ids']!=[]:raise ValueError('Four-modal static probe required')
        if row.get('readback_frame',row['frame']+1)!=row['frame']+1:raise ValueError('Success readback frame differs from F+1')
        for modality,item in row['images'].items():
            p=safe_path(root,item['path']);image,_,_=read_png(p,rig['width'],rig['height']);images[modality]=image
            if item['frame']!=row['frame'] or abs(item['timestamp_s']-row['timestamp_s'])>1e-4:raise ValueError('Probe frame/timestamp mismatch')
            error=float(np.abs(np.asarray(item['T_world_from_camera'])-matrix).max());pose_error=max(pose_error,error)
            if error>=1e-4:raise ValueError('Actual camera transform mismatch')
            files.append({'path':item['path'],'sha256':sha(p),'bytes':p.stat().st_size})
        raw=images['semantic'][...,0]
        if images['semantic'][...,1:].any() or raw.max()>28 or images['instance'][...,0].max()>28:raise ValueError('Raw class encoding differs')
        aligned=resolved_class_alignment(raw,images['instance'][...,0]);disagreement=max(disagreement,aligned['resolved_disagreement'])
        if aligned['resolved_disagreement']>.02:raise ValueError('Resolved class alignment exceeds frozen2%')
        packed=images['depth'].astype(np.uint32);depth=(packed[...,0]+256*packed[...,1]+65536*packed[...,2])*(1000/16777215)
        if not np.isfinite(depth).all() or depth.min()<0 or depth.max()>1000:raise ValueError('Packed axial depth differs')
        target=segmentation_target(raw,images['instance'],[]);support=np.bincount(target[target!=255],minlength=17);counts+=support;raw_counts+=np.bincount(raw.ravel(),minlength=29)
        cases.append({'index':index,'sample':row,'support':support.tolist(),'raw_wall_pixels':int((raw==4).sum()),
            'raw_fence_pixels':int((raw==5).sum()),'non_sky_other_pixels':int(((target==0)&(raw!=11)).sum())})
    selected=[]
    for anchor in anchors[:n]:
        choices=[r for r in cases if r['sample']['anchor_spawn_index']==anchor['spawn_index']]
        for field in ['raw_wall_pixels','raw_fence_pixels','non_sky_other_pixels']:
            best=max(choices,key=lambda r:(r[field],-r['index']))
            if best[field]>0 and best['index'] not in selected:selected.append(best['index'])
        best=max(choices,key=lambda r:(r['support'][16],-r['index']))
        if best['index'] not in selected:selected.append(best['index'])
    out.mkdir(parents=True);pages=[];catalog=[]
    for start in range(0,len(selected),4):
        indices=selected[start:start+4];page=Image.new('RGB',(1280,384*len(indices)))
        for j,index in enumerate(indices):
            entry=cases[index];row=entry['sample'];rgb=Image.open(root/row['images']['rgb']['path']).convert('RGB')
            raw=read_png(root/row['images']['semantic']['path'],640,360)[0][...,0];inst=read_png(root/row['images']['instance']['path'],640,360)[0]
            target=segmentation_target(raw,inst,[]);color=np.zeros((*target.shape,3),dtype=np.uint8);valid=target!=255;color[valid]=PALETTE[target[valid]]
            y=j*384;page.paste(rgb,(0,y+24));page.paste(Image.fromarray(color),(640,y+24))
            ImageDraw.Draw(page).text((4,y+4),f"PROBE ONLY {row['town']} anchor{row['anchor_spawn_index']} {row['camera']} RGB / GT wall{entry['raw_wall_pixels']} fence{entry['raw_fence_pixels']}",fill='white')
            catalog.append({**entry,'page':f'page-{start//4:02d}.png'})
        p=out/f'page-{start//4:02d}.png';page.save(p);pages.append({'path':p.name,'sha256':sha(p),'indices':indices})
    write_json(out/'report.json',{'status':'probe_alignment_checked_not_training_approved','training_approved':False,'eligible_for_training':False,
        'town':args.town,'plan_sha256':sha(root/'plan.json'),'capture_report_sha256':sha(root/args.town/'report.json'),'tool_sha256':sha(Path(__file__)),
        'views':len(rows),'checked_PNGs':len(files),'segmentation_pixels':counts.tolist(),'raw_class_pixels':raw_counts.tolist(),
        'max_pose_error':pose_error,'max_resolved_class_disagreement':disagreement,'explicit_success_readback_records':sum('readback_frame' in r for r in rows),
        'scope':'Probe only. Exact frame/timestamp/camera/CRC/encoding checks. Legacy success readback is method-bound by captured callback source, not an explicit success counter. No material or human training approval before viewing pages.',
        'files':files,'catalog':catalog,'pages':pages})
    print(json.dumps({'views':len(rows),'wall_pixels':int(counts[12]),'fence_pixels':int(counts[13]),'pages':len(pages),'training_approved':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--town',choices=['Town06','Town07'],required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
