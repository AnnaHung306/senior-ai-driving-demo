"""Add supported-road and high-confidence wrong-class examples to the first gallery."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    out=ROOT/'artifacts/runs/phase10-visual-supplement-v1-001';out.mkdir(parents=True,exist_ok=False)
    evaluation=ROOT/'artifacts/runs/phase10-evaluation-final-v1-002'
    wrong=read(ROOT/'artifacts/reports/phase10-detection-confusion-final.json')['wrong_class_examples'];examples=read(evaluation/'examples.json')
    selected=[];used=set()
    for row in sorted([x for x in wrong if x['true_class'] in [1,3]],key=lambda r:-r['score']):
        key=(row['run_id'],row['camera'])
        if key in used:continue
        used.add(key);selected.append({**row,'kind':'class_confusion','caption':f'GT class {row["true_class"]}, highest-score match class {row["predicted_class"]} @ {row["score"]:.3f}'})
        if len(selected)==4:break
    for row in sorted(examples,key=lambda r:r['road_iou'] if r['road_iou'] is not None else 2):
        key=(row['run_id'],row['camera'])
        if key in used:continue
        raw=np.array(Image.open(ROOT/row['sample_path']/f'{row["camera"]}-instance.png'))[:,:,0]
        if int((raw==1).sum())<5000:continue
        used.add(key);selected.append({**row,'kind':'supported_road','caption':f'GT road pixels {int((raw==1).sum())}; road IoU {row["road_iou"]:.4f}'})
        if len(selected)==7:break
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    pages=[]
    for i,row in enumerate(selected):
        run=ROOT/'artifacts/runs'/row['run_id'];gt=read(run/'audit'/f'{row["frame"]:08d}.json');camera=gt['cameras'][row['camera']]
        rgb=Image.open(run/camera['RGB']['file']).convert('RGB');left=rgb.copy();right=rgb.copy();dl=ImageDraw.Draw(left);dr=ImageDraw.Draw(right)
        if row['kind']=='class_confusion':dl.rectangle(row['true_box'],outline='#71ffa0',width=3);dr.rectangle(row['predicted_box'],outline='#ffbe7a',width=3)
        else:
            raw=np.array(Image.open(run/'audit'/f'{row["frame"]:08d}'/f'{row["camera"]}-instance.png'))[:,:,0];pred=np.array(Image.open(run/'audit'/f'{row["frame"]:08d}'/f'{row["camera"]}-predicted_seg.png'))
            for image,mask in [(left,raw==1),(right,pred==1)]:
                a=np.array(image);a[mask]=(a[mask]*.5+np.array([80,210,180])*.5).astype(np.uint8);image.paste(Image.fromarray(a))
        panel=Image.new('RGB',(1280,445),'#111b26');draw=ImageDraw.Draw(panel)
        draw.text((10,7),f'{row["run_id"]} F{row["frame"]} {row["camera"]}',font=font,fill='white');draw.text((10,37),row['caption'],font=small,fill='#eac38e')
        draw.text((10,65),'Independent GT',font=small,fill='#71ffa0');draw.text((650,65),'Neural prediction',font=small,fill='#ffbe7a');panel.paste(left,(0,85));panel.paste(right,(640,85))
        name=f'case-{i:02d}.jpg';panel.save(out/name,quality=92);row['file']=name;pages.append(panel)
    for start in range(0,len(pages),3):
        page=Image.new('RGB',(1280,445*len(pages[start:start+3])),'#111b26')
        for j,p in enumerate(pages[start:start+3]):page.paste(p,(0,j*445))
        page.save(out/f'contact-{start//3}.jpg',quality=92)
    (out/'catalog.json').write_text(json.dumps({'status':'pending_actual_visual_review','cases':selected,
        'selection_note':'Original first gallery includes road-empty negatives with tiny FP union. Supplemental supported-road views require >=5000 GT road pixels; wrong-class cases remain unchanged.'},indent=2)+'\n',encoding='utf8')
    print({'supplemental_panels':len(selected)})

if __name__=='__main__':main()
