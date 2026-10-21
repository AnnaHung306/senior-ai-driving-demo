"""Evaluation-only route/control plots, explicit map/ego localization prior."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/tmp/phase89-mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'artifacts/runs/phase89-visual-QA-v1-001'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);rows=[]
    names=['phase89-drive-unseen-straight1-v1-001','phase89-drive-unseen-straight2-v1-001','phase89-drive-signal-probe-v1-001',
           'phase89-drive-blocker-probe-v1-001','phase89-drive-blocker-probe-v1-002','phase89-drive-pedestrian-probe-v1-001','phase89-drive-sensor-fault-v1-001']
    for name in names:
        folder=ROOT/'artifacts/runs'/name;p=folder/'report.json'
        if not p.is_file():continue
        r=read(p)
        if not r.get('records'):continue
        records=[read(folder/x['file']) for x in r['records']];planned=np.array(r['route']['points_xy_m']);actual=np.array([x['ego_pose_navigation_prior'][:2] for x in records])
        t=np.array([x['simulation_s'] for x in records]);speed=np.array([x['speed_mps'] for x in records])*3.6;target=np.array([x['plan']['target_speed_mps'] for x in records])*3.6
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');axes[0].plot(planned[:,0],planned[:,1],'--',color='#7c8b99',label='Map route prior');axes[0].plot(actual[:,0],actual[:,1],color='#218d7c',label='Actual ego localization')
        axes[0].scatter(*actual[0],color='#278ab9',label='Start');axes[0].scatter(*planned[-1],marker='x',color='#ad5e49',label='Goal');axes[0].axis('equal');axes[0].set(xlabel='Map x (m)',ylabel='Map y (m)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
        axes[1].plot(t,speed,label='Actual ego speed');axes[1].plot(t,target,'--',label='Target speed')
        for state,color in [('SIGNAL_STOP','#cc6d43'),('OBSTACLE_STOP','#a64f5b'),('FAILSAFE','#9572b4'),('YIELD_OFFSET','#479580')]:
            selected=np.array([x['plan']['state']==state for x in records])
            if selected.any():axes[1].scatter(t[selected],speed[selected],s=12,color=color,label=state)
        axes[1].set(xlabel='Simulation time (s)',ylabel='Speed (km/h)');axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
        fig.suptitle(name+' | '+r['status'],fontsize=12)
        fig.text(.01,-.04,'Map/ego localization prior explicitly used. This is a closed-loop engineering plot, not a visual-localization accuracy or generalization score.',fontsize=9)
        target_path=out/f'case-{len(rows):02d}.png';fig.savefig(target_path,dpi=130,bbox_inches='tight');plt.close(fig)
        rows.append({'file':target_path.name,'run':name,'report_sha256':sha(p),'plot_sha256':sha(target_path),'frames':len(records),
                     'status':r['status'],'max_cross_track_m':max(x['navigation']['cross_track_m'] for x in records)})
    for page in range((len(rows)+3)//4):
        canvas=Image.new('RGB',(1560,640),'white')
        for i,r in enumerate(rows[page*4:(page+1)*4]):
            im=Image.open(out/r['file']).convert('RGB');im.thumbnail((775,315));canvas.paste(im,((i%2)*780,(i//2)*320))
        canvas.save(out/f'contact-{page}.png')
    (out/'catalog.json').write_text(json.dumps({'status':'generated_pending_assistant_visual_review','cases':rows,'source_sha256':sha(Path(__file__))},indent=2)+'\n',encoding='utf8')
    print({'plots':len(rows),'output':str(out)})
if __name__=='__main__':main()
