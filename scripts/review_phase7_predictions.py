"""Evaluation-only GT trajectory plots; never served by the World viewer.

Uses an existing plotting Python with NumPy/Matplotlib. Does not load ML weights.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/tmp/phase7-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    output=ROOT/'artifacts/runs/phase7-visual-QA-v1-001'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);root=ROOT/'data/sunny-cross-town-v3-007';run=ROOT/'artifacts/runs/phase7-motion-v1-004'
    evaluation=ROOT/'artifacts/runs/phase7-evaluation-v1-004';report=read(evaluation/'report.json');details=read(evaluation/'forecast_details.json')['2'];chosen=[]
    for scene in ['town01-00','town01-01','town04-00','town04-01']:chosen.append(('scene_worst_FDE2',max([d for d in details if d['scene']==scene],key=lambda d:d['FDE'])))
    val=sorted([d for d in details if d['scene'].startswith('town04-')],key=lambda d:d['FDE'])
    chosen.extend([('validation_median_FDE2',val[len(val)//2]),('validation_best_FDE2',val[0])])
    for category in [3,4]:
        for d in details:
            m=read(run/d['scene']/str(d['frame'])/'motion.json');o=next(o for o in m['objects'] if o['track_id']==d['track_id'])
            if o['class_id']==category:chosen.append(('two_wheeler_example',d));break
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    receipts=[];verified={}
    def load(relative):
        assert sha(root/relative)==report['GT_inventory'][relative];verified[relative]=report['GT_inventory'][relative];return read(root/relative)
    for i,(reason,d) in enumerate(chosen):
        scene=d['scene'];frame=d['frame'];m=read(run/scene/str(frame)/'motion.json');o=next(o for o in m['objects'] if o['track_id']==d['track_id'])
        world=read(ROOT/'artifacts/runs/phase7-world-v1-001'/scene/str(frame)/'world.json');sensor=load(world['parent_manifest']);cal=load(sensor['calibration'])
        T=np.array(next(c for c in cal['cameras'] if c['name']=='front')['T_ego_from_camera_local'])
        label=load(f'labels/frames/{scene}/{frame:08}.json');ego=np.array(label['capture_world_from_camera']['front']['rgb'])@np.linalg.inv(T);inverse=np.linalg.inv(ego)
        gt=[];ego_actual=[]
        for k in range(11):
            future=load(f'labels/frames/{scene}/{frame+4*k:08}.json');actor=next(a for a in future['objects_3d'] if a['actor_id']==d['actor_id'])
            center=np.mean(actor['world_vertices_m'],axis=0);gt.append((inverse[:3,:3]@center+inverse[:3,3])[:2])
            ef=np.array(future['capture_world_from_camera']['front']['rgb'])@np.linalg.inv(T);ego_actual.append((inverse@ef)[:2,3])
        gt=np.array(gt);pred=np.array(o['forecast']['positions_ego_xy_m']);history=np.array(o['history_ego_xy_m']);ego_pred=np.array(m['tracking']['ego_motion']['positions_ego_xy_m'])
        error=np.linalg.norm(pred[1:11]-gt[1:],axis=1);assert abs(float(error.mean())-d['ADE'])<1e-10 and abs(float(error[-1])-d['FDE'])<1e-10
        fig,axes=plt.subplots(1,2,figsize=(11,5.3),gridspec_kw={'width_ratios':[1.3,1]});ax=axes[0]
        ax.plot(history[:,1],history[:,0],'.-',color='#8094a7',label='Past RGB measurements')
        ax.plot(pred[:,1],pred[:,0],'o--',color='#218b9a',markersize=3,label='CV forecast, 0..3 s')
        ax.plot(gt[:,1],gt[:,0],'o-',color='#23824e',markersize=4,label='GT actual, 0..2 s')
        ax.scatter([o['center_ego_m'][1]],[o['center_ego_m'][0]],marker='x',s=70,color='#bc4d39',label='Current RGB center',zorder=5)
        ax.plot(ego_pred[:,1],ego_pred[:,0],'--',color='#999999',label='Assumed ego twist')
        ego_actual=np.array(ego_actual);ax.plot(ego_actual[:,1],ego_actual[:,0],':',color='#444444',label='GT ego, 0..2 s')
        ax.scatter([0],[0],marker='s',s=65,color='#334b61');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        ax.set_xlabel('Current ego y: right (m)');ax.set_ylabel('Current ego x: forward (m)');ax.legend(fontsize=8,loc='best')
        axes[1].plot(np.arange(1,11)*.2,error,'o-',color='#b45441',markersize=4);axes[1].axhline(2,color='#8b8b8b',linestyle='--',linewidth=1,label='2 m diagnostic threshold')
        axes[1].set_xlabel('Forecast horizon (s)');axes[1].set_ylabel('Euclidean position error (m)');axes[1].set_xlim(0,2.05);axes[1].set_ylim(bottom=0);axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
        fig.suptitle(f"{scene} / F{frame} / class {o['class_id']} / {d['track_id']}\n{reason}: ADE2 {d['ADE']:.3f} m | FDE2 {d['FDE']:.3f} m",fontsize=11)
        fig.text(.02,.015,'Evaluation-only GT; fixed CURRENT ego axes. CV is not learned intent. Same known train/validation, no heldout.',fontsize=8,color='#555555')
        fig.tight_layout(rect=[0,.04,1,.9]);file=f'case-{i:02}.png';fig.savefig(output/file,dpi=140);plt.close(fig)
        receipts.append({'file':file,'reason':reason,**d,'plot_sha256':sha(output/file),'motion_sha256':sha(run/scene/str(frame)/'motion.json'),
            'GT_future_ego_xy_m':gt.tolist(),'predicted_ego_xy_m':pred.tolist(),'pose_plot_matches_metric':True})
    for page in range((len(receipts)+3)//4):
        im=Image.new('RGB',(1540,850),'#f1f3f6');draw=ImageDraw.Draw(im)
        for j,r in enumerate(receipts[page*4:(page+1)*4]):
            image=Image.open(output/r['file']).convert('RGB');image.thumbnail((760,405));x=(j%2)*770;y=(j//2)*425
            im.paste(image,(x,y));draw.text((x+8,y+408),r['file']+' '+r['reason'],fill='#293b4e')
        im.save(output/f'contact-{page}.png')
    (output/'catalog.json').write_text(json.dumps({'status':'generated_pending_assistant_visual_review','cases':receipts,'GT_files_reverified':verified,
        'plot_runtime_python':sys.version,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'tool_sha256':sha(Path(__file__))},indent=2)+'\n',encoding='utf8')
    print({'status':'generated_evaluation_only_QA','cases':len(receipts),'contact_pages':(len(receipts)+3)//4})

if __name__=='__main__':main()
