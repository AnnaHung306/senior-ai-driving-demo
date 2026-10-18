"""Render audited saved GPU predictions, with no new model inference or heldout."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from PIL import Image,ImageDraw
from senior_ai.core import write_json
from senior_ai.sunny_data import DETECTION_CLASSES,sha
from senior_ai.dataset import safe_path
from senior_ai.detection_metrics import evaluate_detection,iou


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    run=args.run.resolve();root=args.root.resolve();out=args.output.resolve();state=read(run/'state.json');proof=read(args.integrity)
    if out.exists() or out.is_relative_to(root):raise ValueError('New gallery outside immutable dataset required')
    if state['status']!='completed' or not proof['artifact_integrity_passed'] or not proof['full_and_by_scene_metrics_recomputed_exact']:
        raise ValueError('Completed detector and full canonical integrity proof required')
    if Path(proof['run']).resolve()!=run or sha(run/'validation_frames.jsonl')!=state['validation_frames_sha256'] or proof['hashes']['validation_frames.jsonl']!=state['validation_frames_sha256'] or proof['hashes']['best.pt']!=state['best_sha256']:
        raise ValueError('Gallery proof/checkpoint/report mismatch')
    if sha(root/'audit.json')!=read(run/'provenance.json')['data']['audit_sha256']:raise ValueError('Main data changed')
    frames=[json.loads(line) for line in (run/'validation_frames.jsonl').read_text().splitlines()]
    inventory={r['path']:r['sha256'] for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
    names={v:k for k,v in DETECTION_CLASSES.items()};scored=[]
    for frame in frames:
        per=evaluate_detection([frame],DETECTION_CLASSES)['per_class'];tp=sum(c['true_positives'] for c in per.values());fp=sum(c['false_positives'] for c in per.values());fn=sum(c['false_negatives'] for c in per.values())
        score=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1
        scored.append({'frame':frame,'per_class':per,'F1':score,'tp':tp,'fp':fp,'fn':fn})
    selected={}
    def add(row,reason):selected.setdefault(row['frame']['index'],{'record':row,'reasons':[]})['reasons'].append(reason)
    for name in DETECTION_CLASSES:
        choices=[r for r in scored if r['per_class'][name]['ground_truth_objects']]
        if choices:add(min(choices,key=lambda r:(r['per_class'][name]['recall_at_score'],r['F1'],r['frame']['index'])),'worst supported class recall/'+name)
    for scene in sorted({f['scene_id'] for f in frames}):
        choices=[r for r in scored if r['frame']['scene_id']==scene]
        add(min(choices,key=lambda r:(r['F1'],r['frame']['index'])),'worst whole-frame F1/'+scene)
        bikes=[r for r in choices if r['per_class']['bicycle']['ground_truth_objects']]
        if bikes:
            add(max(bikes,key=lambda r:(r['per_class']['bicycle']['false_negatives'],-r['frame']['index'])),'most bicycle misses/'+scene)
            tiny=[r for r in bikes if any(t['label']==3 and (t['box'][2]-t['box'][0])*(t['box'][3]-t['box'][1])<1024 for t in r['frame']['targets']) and r['per_class']['bicycle']['false_negatives']]
            if tiny:add(min(tiny,key=lambda r:(min((t['box'][2]-t['box'][0])*(t['box'][3]-t['box'][1]) for t in r['frame']['targets'] if t['label']==3),r['frame']['index'])),'small bicycle missed/'+scene)
    false_bikes=[r for r in scored if not r['per_class']['bicycle']['ground_truth_objects'] and r['per_class']['bicycle']['false_positives']]
    if false_bikes:add(max(false_bikes,key=lambda r:(r['per_class']['bicycle']['false_positives'],-r['frame']['index'])),'bicycle false positives without bicycle GT')
    out.mkdir(parents=True);catalog=[];panels=[]
    for rank,case in enumerate(selected.values()):
        record=case['record'];frame=record['frame'];path=safe_path(root,frame['rgb'])
        if sha(path)!=inventory[frame['rgb']]:raise ValueError('Gallery RGB bytes changed')
        rgb=Image.open(path).convert('RGB');gt=rgb.copy();pred=rgb.copy();gd=ImageDraw.Draw(gt);pd=ImageDraw.Draw(pred)
        for t in frame['targets']:
            color='cyan' if t['label']==3 else 'lime';b=t['box'];gd.rectangle(b,outline=color,width=2);gd.text((b[0],max(0,b[1]-11)),names[t['label']],fill=color)
        for b in frame['ignore_boxes']:gd.rectangle(b,outline='gray',width=1)
        for p in frame['predictions']:
            if p['score']>=.5:
                b=p['box'];pd.rectangle(b,outline='yellow',width=2);pd.text((b[0],max(0,b[1]-11)),f"{names[p['label']]} {p['score']:.2f}",fill='yellow')
        panel=Image.new('RGB',(1920,384));panel.paste(rgb,(0,24));panel.paste(gt,(640,24));panel.paste(pred,(1280,24))
        ImageDraw.Draw(panel).text((4,4),f"{frame['scene_id']}/{frame['camera']} F{frame['frame']} RGB / GT cyan=bike gray=ignore / SAVED GPU score>=.5 TP{record['tp']} FP{record['fp']} FN{record['fn']}",fill='white')
        target=out/f'case-{rank:02d}.png';panel.save(target);panels.append(panel)
        zooms=[]
        for index,t in enumerate([t for t in frame['targets'] if t['label']==3]):
            b=t['box'];x,y=(b[0]+b[2])/2,(b[1]+b[3])/2
            box=(int(max(0,min(576,x-32))),int(max(0,min(296,y-32))))
            crop=(box[0],box[1],box[0]+64,box[1]+64)
            zoom=Image.new('RGB',(640,320));zoom.paste(gt.crop(crop).resize((320,320),Image.Resampling.NEAREST),(0,0));zoom.paste(pred.crop(crop).resize((320,320),Image.Resampling.NEAREST),(320,0))
            z=out/f'case-{rank:02d}-bike-{index}-display-zoom.png';zoom.save(z);zooms.append({'path':z.name,'sha256':sha(z),'GT_box':b,'source_crop_xyxy':crop})
        catalog.append({'rank':rank,'index':frame['index'],'sample':{k:frame[k] for k in ['scene_id','group_id','camera','frame','rgb']},'selection_reasons':case['reasons'],'counts':{k:record[k] for k in ['tp','fp','fn','F1','per_class']},'panel':target.name,'panel_sha256':sha(target),'display_only_zooms':zooms})
    pages=[]
    for start in range(0,len(panels),3):
        subset=panels[start:start+3];page=Image.new('RGB',(1920,384*len(subset)))
        for i,panel in enumerate(subset):page.paste(panel,(0,i*384))
        p=out/f'page-{start//3:02d}.png';page.save(p);pages.append({'path':p.name,'sha256':sha(p),'case_ranks':list(range(start,start+len(subset)))})
    write_json(out/'catalog.json',{'checkpoint_sha256':state['best_sha256'],'validation_frames_sha256':state['validation_frames_sha256'],
        'integrity_report_sha256':sha(args.integrity),'tool_sha256':sha(Path(__file__)),'scope':'Saved original validation GPU predictions only; no new inference or heldout. Biased diagnostic selection; nearest-neighbor zooms are display only. Not yet human visual approval.',
        'display_score_threshold':.5,'cases':catalog,'pages':pages})
    print(json.dumps({'cases':len(catalog),'pages':len(pages),'output':str(out),'human_reviewed':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--integrity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
