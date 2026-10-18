"""Train-only selected-context CPU fit diagnostic, not population accuracy."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
from PIL import Image,ImageDraw
import torch
from predict_sunny import load_model,PALETTE
from senior_ai.sunny_data import png,sha,segmentation_target,SEGMENTATION_NAMES
from senior_ai.segmentation_metrics import confusion,metrics
from senior_ai.core import write_json

def read(p):return json.loads(p.read_text(encoding='utf-8'))

out=Path(__file__).resolve().parent
if (out/'report.json').exists():raise FileExistsError('Preserve completed diagnostic')
root=ROOT/'data/sunny-cross-town-v3-007'
checkpoint=ROOT/'artifacts/runs/sunny-segmentation-v6-001/best.pt'
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
parent=read(checkpoint.parent/'state.json')
assert parent['status']=='completed' and sha(checkpoint)==parent['best_sha256'] and not parent['lockbox_opened']
model,task,saved=load_model(checkpoint,'cpu')
assert task=='segmentation' and sha(root/'audit.json')==saved['provenance']['data']['audit_sha256']
for relative,digest in saved['provenance']['source']['files'].items():assert sha(ROOT/relative)==digest
train=[json.loads(line) for line in (root/'training/train.jsonl').read_text().splitlines()]
metadata={r['path']:r['sha256'] for r in read(root/'metadata_inventory.json')}
assert sha(root/'metadata_inventory.json')==saved['provenance']['data']['metadata_inventory_sha256']
assert sha(root/'training/train.jsonl')==metadata['training/train.jsonl']
inventory={r['path']:r['sha256'] for r in map(json.loads,(root/'image_inventory.jsonl').read_text().splitlines())}
assert sha(root/'image_inventory.jsonl')==saved['provenance']['data']['image_inventory_sha256']
appearance=ROOT/'artifacts/runs/wall-training-appearance-survey-001/report.json'
shape=ROOT/'artifacts/runs/wall-training-shape-survey-001/report.json'
cases={r['index']:{**r,'selection':'appearance/'+r['appearance_hint']} for r in read(appearance)['selected']}
for item in read(shape)['selected']:
    if item['kind']=='neutral_nonwall_hint':
        r=item['record'];cases.setdefault(r['index'],{**r,'selection':'neutral_nonwall_hint'})
write_json(out/'contract.json',{'checkpoint_sha256':sha(checkpoint),'tool_sha256':sha(Path(__file__)),
    'selection_reports':{str(p):sha(p) for p in [appearance,shape]},'indices':sorted(cases),
    'parent_state_sha256':sha(checkpoint.parent/'state.json'),'reused_selection_contract_sha256':sha(ROOT/'artifacts/runs/wall-training-fit-001/contract.json'),
    'scope':'Seen original train selected contexts only, CPU FP32, not official GPU gates, population fit or unseen generalization.'})
assert sorted(cases)==read(ROOT/'artifacts/runs/wall-training-fit-001/contract.json')['indices']
records=[];panels={};total=np.zeros((17,17),dtype=np.int64)
for index,case in sorted(cases.items()):
    row=train[index];assert row==case['sample'] and row['split']=='train'
    for k in ['rgb','semantic','instance']:assert sha(root/row[k])==inventory[row[k]]
    rgb=png(root/row['rgb']);truth=segmentation_target(png(root/row['semantic'])[...,0],png(root/row['instance']),row['verified_obstacle_actor_ids'])
    with torch.inference_mode():prediction=model(torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255)[0].argmax(0).numpy()
    m=confusion(truth,prediction,17);total+=m;v=metrics(m,SEGMENTATION_NAMES)
    records.append({'index':index,'sample':row,'selection':case['selection'],'wall':v['classes'][12],
        'wall_false_positive_pixels':int(m[:,12].sum()-m[12,12]),'wall_false_negative_pixels':int(m[12].sum()-m[12,12]),'confusion':m.tolist()})
    gt=np.zeros_like(rgb);valid=truth!=255;gt[valid]=PALETTE[truth[valid]]
    panel=Image.new('RGB',(1920,384));draw=ImageDraw.Draw(panel)
    for j,pixels in enumerate([rgb,gt,PALETTE[prediction]]):panel.paste(Image.fromarray(pixels),(640*j,24))
    draw.text((4,4),f"train index{index} {row['scene_id']}/{row['camera']} {case['selection']} RGB / GT / CPU PRED",fill='white')
    panels[index]=panel
selected=[]
for key in ['wall_false_positive_pixels','wall_false_negative_pixels']:
    for r in sorted(records,key=lambda r:(-r[key],r['index']))[:3]:
        if r['index'] not in selected:selected.append(r['index'])
pages=[]
for start in range(0,len(selected),3):
    subset=selected[start:start+3];page=Image.new('RGB',(1920,384*len(subset)))
    for j,index in enumerate(subset):page.paste(panels[index],(0,384*j))
    p=out/f'page-{start//3}.png';page.save(p);pages.append({'path':p.name,'sha256':sha(p)})
report={'status':'completed_train_selected_fit_diagnostic_only','views':len(records),'aggregate':metrics(total,SEGMENTATION_NAMES),
    'records':records,'pages':pages,'scope':'Biased selected seen train examples; not full train accuracy, val, acceptance, or independent trials.'}
write_json(out/'report.json',report)
print(json.dumps({'views':len(records),'selected_train_wall':report['aggregate']['classes'][12],'pages':len(pages)}))
