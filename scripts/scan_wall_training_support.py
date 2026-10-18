"""Train-only, model-free shape/neutral-colour hints for wall error diagnosis.

Apparent height and neutral pixels are candidate proxies, not material labels.
Reads only the original train index, never validation/development/lockbox images.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def rgb(path):
    with Image.open(path) as im:
        if im.mode!='RGB' or im.size!=(640,360):raise ValueError('Rig/PNG mode differs')
        return np.array(im)


def main(args):
    if args.output.exists():raise FileExistsError('Use a new survey version')
    root=args.root.resolve();audit=read(root/'audit.json');review=read(root/'visual_review.json')
    if audit['status']!='passed' or review['audit_sha256']!=sha(root/'audit.json') or not review['training_approved']:raise ValueError('Bound main approval required')
    if sha(root/'image_inventory.jsonl')!=audit['image_inventory_sha256'] or sha(root/'metadata_inventory.json')!=audit['metadata_inventory_sha256']:raise ValueError('Inventory changed')
    metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
    if sha(root/'training/train.jsonl')!=metadata['training/train.jsonl']:raise ValueError('Train index changed')
    samples=[json.loads(line) for line in (root/'training/train.jsonl').read_text(encoding='utf-8').splitlines()]
    if any(r['split']!='train' or r['scene_id'].split('-')[0] not in {'town01','town03'} for r in samples):raise ValueError('Original train boundary only')
    inventory={r['path']:r for r in map(json.loads,(root/'image_inventory.jsonl').read_text(encoding='utf-8').splitlines())}
    rig=read(root/'rig.json');focal=rig['width']/(2*math.tan(math.radians(rig['horizontal_fov_deg'])/2))
    def checked(relative):
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or sha(p)!=inventory[relative]['sha256']:raise ValueError('Selected bytes changed')
        return rgb(p)
    def shape(mask,z):
        mask=mask&(z>.5)&(z<40)
        ys,xs=np.nonzero(mask)
        if len(ys)<512:return None
        y0,y1=np.percentile(ys,[5,95]);x0,x1=np.percentile(xs,[5,95]);median=float(np.median(z[mask]))
        return {'pixels':len(ys),'width_90pct_px':float(x1-x0),'height_90pct_px':float(y1-y0),
            'apparent_height_m_proxy':float((y1-y0)*median/focal),'median_axial_depth_m':median,'median_y':float(np.median(ys))}
    def one(pair):
        index,row=pair;raw=checked(row['semantic'])[...,0]
        wall=raw==4;negative=np.isin(raw,[3,8,9,10,11,17,20,22,23,25,26,27,28]);result={'index':index,'sample':row,'wall_pixels':int(wall.sum())}
        if wall.sum()<512 and negative.sum()<5000:return result
        packed=checked(row['semantic'].replace('semantic_segmentation','depth')).astype(np.uint32)
        z=(packed[...,0]+256*packed[...,1]+65536*packed[...,2])*(1000/16777215)
        wall_shape=shape(wall,z)
        if wall_shape:
            result['wall_shape']=wall_shape
            result['low_wall_hint']=.4<=wall_shape['apparent_height_m_proxy']<=2.5 and wall_shape['width_90pct_px']>=60 and wall_shape['median_y']>=120
        if negative.sum()>=5000:
            image=checked(row['rgb']);neutral=(image.max(2)-image.min(2)<=30)&(image.mean(2)>=40)&(image.mean(2)<=220)
            n=shape(negative&neutral,z)
            if n and n['pixels']>=3000 and n['height_90pct_px']>=80 and n['apparent_height_m_proxy']>=2:
                result['neutral_nonwall_hint']=n
        return result
    args.output.mkdir(parents=True);records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(one,enumerate(samples)):
            records.append(record)
            if len(records)%1000==0:print(json.dumps({'checked_train_views':len(records)}),flush=True)
    low=[r for r in records if r.get('low_wall_hint')];negative=[r for r in records if r.get('neutral_nonwall_hint')]
    selected=[]
    for kind,candidates in [('low_wall_hint',low),('neutral_nonwall_hint',negative)]:
        candidates=sorted(candidates,key=lambda r:(-r['wall_shape']['pixels'] if kind=='low_wall_hint' else -r[kind]['pixels'],r['index']))
        seen=set();subset=[]
        for row in candidates:
            key=(row['sample']['group_id'],row['sample']['camera'])
            if key not in seen:seen.add(key);subset.append(row)
            if len(subset)==12:break
        for start in range(0,len(subset),3):
            canvas=Image.new('RGB',(1280,384*len(subset[start:start+3])))
            for j,row in enumerate(subset[start:start+3]):
                image=checked(row['sample']['rgb']);raw=checked(row['sample']['semantic'])[...,0];overlay=image.copy()
                mask=raw==4 if kind=='low_wall_hint' else np.isin(raw,[3,8,9,10,11,17,20,22,23,25,26,27,28])
                overlay[mask]=[255,0,255] if kind=='low_wall_hint' else [0,255,255]
                y=j*384;canvas.paste(Image.fromarray(image),(0,y+24));canvas.paste(Image.fromarray(overlay),(640,y+24))
                ImageDraw.Draw(canvas).text((4,y+4),f'{kind} index{row["index"]} {row["sample"]["scene_id"]}/{row["sample"]["camera"]}',fill='white')
                selected.append({'kind':kind,'record':row,'page':f'{kind}-{start//3}.png'})
            canvas.save(args.output/f'{kind}-{start//3}.png')
    with (args.output/'records.jsonl.gz').open('wb') as stream:
        with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as z:z.write(''.join(json.dumps(r)+'\n' for r in records).encode())
    result={'status':'completed_train_diagnostic_only','training_approved':False,'main_audit_sha256':sha(root/'audit.json'),'train_views':len(samples),
        'low_wall_hint_views':len(low),'low_wall_hint_groups':dict(Counter(r['sample']['group_id'] for r in low)),
        'neutral_nonwall_hint_views':len(negative),'neutral_nonwall_hint_groups':dict(Counter(r['sample']['group_id'] for r in negative)),
        'selected':selected,'records_archive_sha256':sha(args.output/'records.jsonl.gz'),'tool_sha256':sha(Path(__file__)),
        'limits':'Depth-times-image-height is an apparent-height proxy across possibly multiple surfaces, not a measured object height. Neutral colour does not prove rock or masonry; selected RGB must be actually viewed. No new data or independent groups.'}
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='selected'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
