"""Assemble saved prediction/evaluation figures; no new neural inference."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image,ImageDraw
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(args):
    out=args.output
    if out.exists():raise ValueError('New gallery required')
    out.mkdir(parents=True);report=read(args.evaluation/'report.json');records=report['records'];pages=[]
    for kind,source,size,per_page,columns in [('prediction',args.run,(640,455),4,2),('GT_evaluation',args.evaluation,(600,270),8,2)]:
        for start in range(0,len(records),per_page):
            items=records[start:start+per_page];page=Image.new('RGB',(size[0]*columns,size[1]*((len(items)+columns-1)//columns)));bindings=[]
            for i,item in enumerate(items):
                path=source/item['folder']/('panel.png' if kind=='prediction' else 'evaluation.png')
                with Image.open(path) as im:page.paste(im.resize(size),(i%columns*size[0],i//columns*size[1]))
                draw=ImageDraw.Draw(page);pos=(i%columns*size[0]+3,i//columns*size[1]+3);draw.rectangle((pos[0],pos[1],pos[0]+265,pos[1]+13),fill='black');draw.text(pos,f"{item['scene_id']} F{item['frame']} {kind}",fill='white')
                bindings.append({'frame':item['frame'],'scene':item['scene_id'],'source':str(path),'sha256':sha(path)})
            name=f'{kind}-{start//per_page:02}.png';page.save(out/name);pages.append({'kind':kind,'path':name,'sha256':sha(out/name),'sources':bindings})
    groups=defaultdict(list)
    for r in records:groups[r['scene_id']].append(r)
    chosen={}
    for items in groups.values():chosen[(items[4]['scene_id'],items[4]['frame'])]=['fixed_middle_frame']
    for r in sorted(records,key=lambda r:r['path_GT_error']['lateral_rmse_m'] if r['path_GT_error'] else -1,reverse=True)[:2]:chosen.setdefault((r['scene_id'],r['frame']),[]).append('largest_path_GT_error')
    for r in sorted(records,key=lambda r:r['object_metrics']['FP'],reverse=True)[:2]:chosen.setdefault((r['scene_id'],r['frame']),[]).append('largest_object_false_positive_count')
    cases=[]
    for key,reasons in chosen.items():
        r=next(r for r in records if (r['scene_id'],r['frame'])==key);a=args.run/r['folder']/'panel.png';b=args.evaluation/r['folder']/'evaluation.png'
        cases.append({'scene':key[0],'frame':key[1],'reasons':reasons,'prediction_panel':str(a),'prediction_sha256':sha(a),'GT_panel':str(b),'GT_sha256':sha(b)})
    (out/'catalog.json').write_text(json.dumps({'status':'pending_assistant_visual_review','scope':'All32 saved bundles in contact pages, fixed middle cases and validation error extremes. No independent new trials.','pages':pages,'full_resolution_cases':cases,'run_contract_sha256':sha(args.run/'contract.json'),'evaluation_report_sha256':sha(args.evaluation/'report.json')},indent=2)+'\n',encoding='utf-8');print(json.dumps({'pages':len(pages),'full_resolution_cases':len(cases)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
