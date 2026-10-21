"""Build paired RGB/instance-GT/prediction panels and route/performance figures."""
import argparse,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/tmp/mpl-phase10'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE=np.array([[40,45,50],[90,108,128],[173,133,101],[237,126,165],[240,180,100],[93,164,224],[175,148,223],[93,201,203],[229,185,102],[111,212,169],[235,223,148],[245,168,91],[121,105,97],[118,135,102],[173,190,202],[224,127,88],[98,132,88]],np.uint8)
RAW={1:1,2:2,12:3,13:4,14:5,15:6,16:7,18:8,19:9,24:10,7:11,4:12,5:13,6:14,10:16,25:16}

def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);report=read(a.evaluation/'report.json');examples=read(a.evaluation/'examples.json');images=read(a.evaluation/'per_image.json')
    selected=[];keys=set()
    for group in [sorted(examples,key=lambda x:x['road_iou'] if x['road_iou'] is not None else 2),
                  sorted(examples,key=lambda x:-x['detection_FP']),sorted(examples,key=lambda x:-x['detection_FN'])]:
        for x in group:
            key=(x['run_id'],x['frame'],x['camera'])
            if key not in keys:keys.add(key);selected.append(x)
            if len(selected)%3==0:break
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    catalog=[]
    for rank,x in enumerate(selected):
        rgb=Image.open(ROOT/'artifacts/runs'/x['run_id']/x['RGB']).convert('RGB');sample=ROOT/x['sample_path'];raw=np.array(Image.open(sample/f'{x["camera"]}-instance.png'))[:,:,0]
        lut=np.zeros(29,np.uint8)
        for k,v in RAW.items():lut[k]=v
        target=PALETTE[lut[raw]];target[np.isin(raw,[0,21])]=[255,0,255]
        pred=PALETTE[np.array(Image.open(sample/f'{x["camera"]}-predicted_seg.png'))]
        im=next(r for r in images if all(r[k]==x[k] for k in ['run_id','frame','camera']))
        gt_rgb=rgb.copy();pred_rgb=rgb.copy();dg=ImageDraw.Draw(gt_rgb);dp=ImageDraw.Draw(pred_rgb)
        for t in im['targets']:dg.rectangle(t['box'],outline='#74ff9d',width=2);dg.text(t['box'][:2],str(t['label']),font=small,fill='#74ff9d')
        for d in im['predictions']:
            if d['score']>=.5:dp.rectangle(d['box'],outline='#ffbe7a',width=2);dp.text(d['box'][:2],f'{d["label"]} {d["score"]:.2f}',font=small,fill='#ffbe7a')
        panel=Image.new('RGB',(1920,826),'#111b26');draw=ImageDraw.Draw(panel)
        draw.text((12,9),f'{x["run_id"]}  F{x["frame"]}  {x["camera"]}',font=font,fill='white')
        for col,(label,image) in enumerate([('Source RGB',rgb),('Visible instance GT (green)',gt_rgb),('Detector >=.5 (orange)',pred_rgb)]):
            draw.text((col*640+10,44),label,font=small,fill='white');panel.paste(image,(col*640,70))
        draw.text((10,440),'Raw-instance semantic GT; magenta = ignored unknown/dynamic',font=small,fill='#9ee2cd')
        draw.text((650,440),'Neural segmentation; no GT fed to planner',font=small,fill='#efc783')
        draw.text((1290,440),f'Road IoU: {x["road_iou"]} | FP:{x["detection_FP"]} FN:{x["detection_FN"]}',font=small,fill='white')
        panel.paste(Image.fromarray(target),(0,466));panel.paste(Image.fromarray(pred),(640,466))
        name=f'accuracy-{rank:02d}.jpg';panel.save(a.output/name,quality=90);catalog.append({**x,'file':name,'selection':'lowest road IoU / most FP / most FN, deterministic union'})
    plt.style.use('dark_background')
    runs=report['runs'];fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    timed=[r for r in runs if r['sim_s']>0];labels=[r['run_id'].removeprefix('phase10-drive-') for r in timed];colors=['#8cdbbd' if r['passed'] else '#e79b83' for r in timed]
    axes[0].barh(labels,[r['wall_s']/r['sim_s'] for r in timed],color=colors);axes[0].set_xlabel('Wall / simulation time (lower is faster); zero-duration setup fault = N/A');axes[0].set_title('Green = engineering arrival gates, orange = other result including intentional stops',fontsize=10)
    performance=report['performance']['all'];names=['ticks_ms','RGB_PNG_and_SHA_ms','inference_ms','planning_control_ms','evaluation_observer_ms','JSON_and_publish_ms']
    axes[1].barh(names,[performance[n]['p50'] or 0 for n in names],color='#8dbbd8');axes[1].set_xlabel('Stage median milliseconds; medians do not add to total quantile');fig.savefig(a.output/'performance.png',dpi=140);plt.close(fig)
    classes=report['segmentation']['per_class'];fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    supported=[c for c in classes if c['support_pixels'] and c['id']!=12];ax.barh([c['name'] for c in supported],[100*c['IoU'] for c in supported],color='#8bd5bc');ax.set_xlim(0,100);ax.set_xlabel('IoU %, observed classes only; Wall excluded from project focus');ax.set_title('Fresh same-town diagnostic pixels; classes with no support are not perfect scores');fig.savefig(a.output/'semantic-IoU.png',dpi=140);plt.close(fig)
    telemetry=read(a.evaluation/'timing.json');components=['RGB_PNG_and_SHA_ms','inference_ms','JSON_and_publish_ms','total_cycle_ms'];fig,ax=plt.subplots(figsize=(12,5),layout='constrained');y=np.arange(len(components))
    for offset,name,label,color in [(-.18,'phase10-drive-straight-1-audit-v1-001','Original / GT observer','#e6b381'),(.18,'phase10-drive-straight-1-performance-v1-001','Cached projection + fast PNG / no GT','#8bd5bc')]:
        selected=[t for t in telemetry if t['run_id']==name];ax.barh(y+offset,[np.median([t[k] for t in selected]) for k in components],height=.33,label=label,color=color)
    ax.set_yticks(y,components);ax.set_xlabel('Median milliseconds (host wall clock)');ax.legend(fontsize=8);ax.set_title('Same geometric route; observer/workload differ. Paired microbenchmarks isolate optimizations.',fontsize=10);fig.savefig(a.output/'performance-comparison.png',dpi=140);plt.close(fig)
    forecast=read(a.evaluation/'forecast_samples.json');two=[x for x in forecast if x['horizon_s']==2];fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    ax.scatter([r['constant_position_FDE_m'] for r in two],[r['FDE_m'] for r in two],color='#d8accc');ax.plot([0,5],[0,5],color='#8bd5bc');ax.set_xlabel('Constant-position FDE2 m');ax.set_ylabel('CV forecast FDE2 m');ax.set_title(f'Conditional paired samples n={len(two)}; coverage denominator reported separately');fig.savefig(a.output/'forecast-vs-persistence.png',dpi=140);plt.close(fig)
    for index,r in enumerate(runs):
        folder=ROOT/'artifacts/runs'/r['run_id'];rep=read(folder/'report.json');records=[read(folder/x['file']) for x in rep['records']]
        if not records:continue
        fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained');path=np.array(rep['route']['points_xy_m']);actual=np.array([v['ego_pose_navigation_prior'][:2] for v in records]);ax[0].plot(path[:,0],path[:,1],label='Map route prior');ax[0].plot(actual[:,0],actual[:,1],label='Ego localization prior');ax[0].set_aspect('equal');ax[0].legend(fontsize=7)
        t=[v['simulation_s'] for v in records];ax[1].plot(t,[v['speed_mps'] for v in records],label='Actual ego speed');ax[1].plot(t,[v['plan']['target_speed_mps'] for v in records],label='Planner target');ax[1].set_xlabel('Simulation seconds');ax[1].set_ylabel('m/s');ax[1].legend(fontsize=7);fig.suptitle(r['run_id']+' | '+r['status']);fig.savefig(a.output/f'route-{index:02d}.png',dpi=120);plt.close(fig)
    thumbnails=[]
    for p in sorted(a.output.glob('*.jpg')):
        im=Image.open(p);im.thumbnail((960,450));thumbnails.append((p.name,im.copy()))
    for i in range(0,len(thumbnails),3):
        page=Image.new('RGB',(960,3*460),'#111b26');d=ImageDraw.Draw(page)
        for j,(name,im) in enumerate(thumbnails[i:i+3]):page.paste(im,(0,j*460+22));d.text((8,j*460+2),name,font=small,fill='white')
        page.save(a.output/f'contact-{i//3}.jpg',quality=90)
    (a.output/'catalog.json').write_text(json.dumps({'status':'pending_actual_visual_review','cases':catalog},indent=2)+'\n',encoding='utf8')
    print({'accuracy_panels':len(catalog),'routes':len(runs),'output':str(a.output)})

if __name__=='__main__':main()
