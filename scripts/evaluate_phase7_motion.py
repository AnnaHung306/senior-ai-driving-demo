"""GT-only evaluation. Never imported by causal motion or display generation."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from senior_ai.world_display import read,sha
from senior_ai.bev_geometry import cameras_from_calibration
from senior_ai.motion_forecast import assignment,swept_contact
from senior_ai.core import write_json
from phase5_gt import audited_inventory

def ratio(a,b):return float(a/b) if b else None
def mean(values):return float(np.mean(values)) if values else None

def match(pred,truth):
    a=np.full((len(pred),len(truth)),np.inf)
    for i,p in enumerate(pred):
        for j,t in enumerate(truth):
            d=np.linalg.norm(np.array(p['center_ego_m'][:2])-t['center_ego_m'][:2])
            if p['class_id']==t['class_id'] and d<=2:a[i,j]=d
    return assignment(a,3.)

def tracking_scores(frames):
    total=Counter();last={};history={};identity=Counter();switch_cases=[]
    for f in frames:
        scene=f['scene'];truth=f['truth'];pred=f['pred'];pairs=f['pairs'];total['GT']+=len(truth);total['predictions']+=len(pred)
        total['TP']+=len(pairs);total['FP']+=len(pred)-len(pairs);total['FN']+=len(truth)-len(pairs)
        current={}
        for i,j in pairs:
            actor=(scene,truth[j]['actor_id']);track=pred[i]['track_id'];identity[(scene,track,actor[1])]+=1;current[actor]=track
            if actor in last:
                total['consecutive_matched_transitions']+=1
                if last[actor]!=track:total['ID_switches']+=1;switch_cases.append({'scene':scene,'frame':f['frame'],'actor_id':actor[1],'previous':last[actor],'current':track})
            if actor in history and actor not in last:total['fragmentations']+=1
            history[actor]=track
        last=current
    IDTP=0
    for scene in dict.fromkeys(f['scene'] for f in frames):
        tracks=sorted({k[1] for k in identity if k[0]==scene});actors=sorted({k[2] for k in identity if k[0]==scene});counts=np.zeros((len(tracks),len(actors)))
        for i,t in enumerate(tracks):
            for j,a in enumerate(actors):counts[i,j]=identity[(scene,t,a)]
        IDTP+=sum(counts[i,j] for i,j in assignment(-counts))
    return {**dict(total),'IDTP':int(IDTP),'IDFP':int(total['predictions']-IDTP),'IDFN':int(total['GT']-IDTP),
            'IDF1':ratio(2*IDTP,total['predictions']+total['GT']),'MOTA':1-(total['FP']+total['FN']+total['ID_switches'])/total['GT'] if total['GT'] else None,
            'precision_at_2m':ratio(total['TP'],total['predictions']),'recall_at_2m':ratio(total['TP'],total['GT']),
            'switch_definition':'ID change across consecutive successfully matched sampled frames; fragmentation after unmatched gap separate',
            'ID_switch_cases':switch_cases}

def main(a):
    if a.output.exists():raise FileExistsError('New GT evaluation required')
    root=a.dataset;_,inventory=audited_inventory(root,ROOT);verified={};cache={}
    def load(rel):
        if rel not in cache:
            if sha(root/rel)!=inventory[rel]:raise ValueError('Changed GT input: '+rel)
            verified[rel]=inventory[rel];cache[rel]=read(root/rel)
        return cache[rel]
    state=read(a.motion/'state.json');groups=defaultdict(list);frames=[]
    for row in state['records']:
        folder=Path(row['folder']);overlay=read(a.motion/folder/'motion.json');world=read(a.world/folder/'world.json')
        assert overlay['world_source_sha256']==sha(a.world/folder/'world.json')
        manifest=load(world['parent_manifest']);cameras=cameras_from_calibration(load(manifest['calibration']))
        scene=world['scene_id'];number=world['frame'];label=load(f'labels/frames/{scene}/{number:08}.json')
        ego=np.array(label['capture_world_from_camera']['front']['rgb'])@cameras['front'].invT;inverse=np.linalg.inv(ego)
        positive={}
        for name in cameras:
            for o in load(f'labels/visible/{scene}/{name}/{number:08}.json')['positive_objects']:
                if o['category_id'] in range(1,7):positive[o['actor_id']]=o['category_id']
        all_actors={};truth=[]
        for actor in label['objects_3d']:
            vertices=np.asarray(actor['world_vertices_m'],float);center=vertices.mean(0)
            box={'actor_id':actor['actor_id'],'world_center':center,'center_ego_m':(inverse[:3,:3]@center+inverse[:3,3]).tolist(),
                 'size_lwh_m':(2*np.array(actor['extent_m'])).tolist(),'velocity_ground_ego_mps':(inverse[:3,:3]@actor['velocity_mps'])[:2].tolist()}
            all_actors[actor['actor_id']]=box
            if actor['actor_id'] in positive and np.linalg.norm(box['center_ego_m'][:2])<=25:truth.append({**box,'class_id':positive[actor['actor_id']]})
        pred=[o for o in overlay['objects'] if np.linalg.norm(o['center_ego_m'][:2])<=25];base=[o for o in world['objects'] if np.linalg.norm(o['center_ego_m'][:2])<=25]
        f={'scene':scene,'frame':number,'timestamp':world['timestamp_s'],'truth':truth,'all_actors':all_actors,'ego':ego,'inverse':inverse,
           'pred':pred,'pairs':match(pred,truth),'base':base,'base_pairs':match(base,truth),'overlay':overlay,'folder':folder.as_posix()}
        frames.append(f);groups[scene].append(f)
    horizon={h:{'eligible_GT':0,'right_censored_GT':0,'missing_future_actor_GT':0,'covered_GT':0,'ADE':[],'FDE':[],'persistence_ADE':[],'persistence_FDE':[],
                 'conditional_misses_max2m':0,'unmatched_forecasts':0,'details':[],'eligibility':[]} for h in [1,2,3]}
    velocity=[];ego_error=[];risk=Counter();risk_possible=Counter();object_details=[]
    for scene,seq in groups.items():
        for index,f in enumerate(seq):
            current_to_pred={j:i for i,j in f['pairs']};matched_pred={i for i,j in f['pairs']}
            gt_ego_velocity=(f['inverse'][:3,:3]@load(f"labels/frames/{scene}/{f['frame']:08}.json")['ego_velocity_mps'])[:2]
            estimated=f['overlay']['tracking']['ego_motion']
            if estimated['status']=='estimated':ego_error.append(float(np.linalg.norm(np.array(estimated['velocity_ego_mps'])-gt_ego_velocity)))
            for j,t in enumerate(f['truth']):
                o=f['pred'][current_to_pred[j]] if j in current_to_pred else None
                if o and o['motion']['velocity_available']:
                    p=np.array(o['motion']['velocity_ground_ego_mps']);v=np.array(t['velocity_ground_ego_mps'])
                    velocity.append({'scene':scene,'frame':f['frame'],'class_id':t['class_id'],'actor_id':t['actor_id'],
                        'vector_error_mps':float(np.linalg.norm(p-v)),'speed_error_mps':float(abs(np.linalg.norm(p)-np.linalg.norm(v)))})
                for h,score in horizon.items():
                    steps=h*5
                    if index+steps>=len(seq):score['right_censored_GT']+=1;score['eligibility'].append({'scene':scene,'status':'right_censored'});continue
                    future=seq[index+1:index+steps+1]
                    if any(t['actor_id'] not in ff['all_actors'] for ff in future):score['missing_future_actor_GT']+=1;score['eligibility'].append({'scene':scene,'status':'actor_missing'});continue
                    score['eligible_GT']+=1
                    score['eligibility'].append({'scene':scene,'status':'eligible'})
                    target=np.array([f['inverse'][:3,:3]@ff['all_actors'][t['actor_id']]['world_center']+f['inverse'][:3,3] for ff in future])[:,:2]
                    if o and o['forecast']['status']=='available':
                        predicted=np.array(o['forecast']['positions_ego_xy_m'])[1:steps+1];assert predicted.shape==target.shape
                        error=np.linalg.norm(predicted-target,axis=1);stay=np.linalg.norm(np.array(o['center_ego_m'][:2])-target,axis=1)
                        score['covered_GT']+=1;score['ADE'].append(float(error.mean()));score['FDE'].append(float(error[-1]));score['persistence_ADE'].append(float(stay.mean()));score['persistence_FDE'].append(float(stay[-1]));score['conditional_misses_max2m']+=bool(error.max()>2)
                        score['details'].append({'scene':scene,'frame':f['frame'],'actor_id':t['actor_id'],'track_id':o['track_id'],'ADE':float(error.mean()),'FDE':float(error[-1]),'max_error_m':float(error.max())})
                    if h==3:
                        tt=np.arange(0,steps+1)*.2;own=np.array([f['inverse'][:3,:3]@ff['ego'][:3,3]+f['inverse'][:3,3] for ff in [f]+future])[:,:2]
                        target_full=np.concatenate([np.array(t['center_ego_m'][:2])[None],target])
                        radius=(np.hypot(*read(a.motion/'config.json')['ego_size_lw_m'])+np.hypot(*t['size_lwh_m'][:2]))/2+.25
                        contact,_=swept_contact(target_full-own,tt,np.full(len(tt),radius));positive=contact is not None
                        alarm=o is not None and o['risk']['status'] in ['high','watch'];possible=o is not None and o['risk']['status'] in ['high','watch','potential']
                        risk['TP' if alarm and positive else 'FP' if alarm else 'FN' if positive else 'TN']+=1
                        risk_possible['TP' if possible and positive else 'FP' if possible else 'FN' if positive else 'TN']+=1
                        risk['unknown_prediction']+=o is None or o['risk']['status']=='unknown'
            for h,score in horizon.items():score['unmatched_forecasts']+=sum(i not in matched_pred and o['forecast']['status']=='available' for i,o in enumerate(f['pred']))
            risk['unmatched_nominal_alarms']+=sum(i not in matched_pred and o['risk']['status'] in ['high','watch'] for i,o in enumerate(f['pred']))
            risk_possible['unmatched_alarms']+=sum(i not in matched_pred and o['risk']['status'] in ['high','watch','potential'] for i,o in enumerate(f['pred']))
    def summarize_motion(rows,baseline=False):
        return tracking_scores([{**f,'pred':f['base'] if baseline else f['pred'],'pairs':f['base_pairs'] if baseline else f['pairs']} for f in rows])
    summary={}
    for h,v in horizon.items():
        summary[str(h)]={k:x for k,x in v.items() if k not in ['ADE','FDE','persistence_ADE','persistence_FDE','details','eligibility']}
        summary[str(h)].update(ADE_m=mean(v['ADE']),FDE_m=mean(v['FDE']),persistence_ADE_m=mean(v['persistence_ADE']),persistence_FDE_m=mean(v['persistence_FDE']),
             forecast_coverage=ratio(v['covered_GT'],v['eligible_GT']),conditional_miss_rate_max2m=ratio(v['conditional_misses_max2m'],v['covered_GT']),
             all_GT_miss_rate_including_unavailable=ratio(v['eligible_GT']-v['covered_GT']+v['conditional_misses_max2m'],v['eligible_GT']))
    tracked=summarize_motion(frames);baseline=summarize_motion(frames,True);cfg=read(a.motion/'config.json');gates=cfg['exploratory_quality_gates'];by_split={}
    for split in ['train','validation']:
        belongs=lambda scene:scene.startswith('town04-') if split=='validation' else not scene.startswith('town04-')
        selected=[f for f in frames if belongs(f['scene'])];v=[x for x in velocity if belongs(x['scene'])];hs={}
        for h,x in horizon.items():
            values=[d for d in x['details'] if belongs(d['scene'])];eligible=Counter(d['status'] for d in x['eligibility'] if belongs(d['scene']))
            misses=sum(d['max_error_m']>2 for d in values)
            hs[str(h)]={'eligible_GT':eligible['eligible'],'right_censored_GT':eligible['right_censored'],'missing_future_actor_GT':eligible['actor_missing'],
                'covered_GT':len(values),'forecast_coverage':ratio(len(values),eligible['eligible']),'ADE_m':mean([d['ADE'] for d in values]),'FDE_m':mean([d['FDE'] for d in values]),
                'conditional_miss_rate_max2m':ratio(misses,len(values)),'all_GT_miss_rate_including_unavailable':ratio(eligible['eligible']-len(values)+misses,eligible['eligible'])}
        by_split[split]={'tracking':summarize_motion(selected),'baseline':summarize_motion(selected,True),'velocity':{'paired_samples':len(v),'vector_MAE_mps':mean([d['vector_error_mps'] for d in v]),'speed_MAE_mps':mean([d['speed_error_mps'] for d in v])},'horizons_s':hs}
    val=by_split['validation'];quality={'validation_velocity':val['velocity']['vector_MAE_mps'] is not None and val['velocity']['vector_MAE_mps']<=gates['speed_MAE_mps'],
        'validation_ADE1':val['horizons_s']['1']['ADE_m'] is not None and val['horizons_s']['1']['ADE_m']<=gates['ADE1_m'],
        'validation_FDE2':val['horizons_s']['2']['FDE_m'] is not None and val['horizons_s']['2']['FDE_m']<=gates['FDE2_m'],
        'validation_ID_switches_not_above_baseline':val['tracking'].get('ID_switches',0)<=val['baseline'].get('ID_switches',0)}
    report={'status':'completed_independent_known_domain_evaluation','tracking':tracked,'baseline_tracking_same_detections':baseline,
       'by_scene':{s:{'tracking':summarize_motion(v),'baseline':summarize_motion(v,True)} for s,v in groups.items()},'by_split':by_split,
       'horizons_s':summary,'velocity':{'paired_samples':len(velocity),'vector_MAE_mps':mean([v['vector_error_mps'] for v in velocity]),'speed_MAE_mps':mean([v['speed_error_mps'] for v in velocity]),'target':'captured_actor_ground_velocity_world_rotated_into_current_ego'},
       'ego_velocity_vector_MAE_mps':mean(ego_error),'risk_GT_conservative_circle_proximity_3s':dict(risk),'risk_with_possible_envelope':dict(risk_possible),
       'quality_gates':quality,'exploratory_quality_passed':all(quality.values()),'formal_quality_passed':False,'GT_files_verified':len(verified),'GT_inventory':verified,
       'motion_contract_sha256':sha(a.motion/'contract.json'),'tool_sha256':sha(Path(__file__)),
       'limits':['Known model-selection train/validation, not heldout','Conditional metrics never replace all-GT coverage/miss','Right-censored last frames reported, not counted successful',
                 'Risk GT is conservative-circle proximity, not CARLA collision events or calibrated crash probability','No best-of-K selection; main score uses CV only, braking branch sensitivity only']}
    a.output.mkdir(parents=True);write_json(a.output/'report.json',report);write_json(a.output/'paired_velocity.json',velocity)
    write_json(a.output/'forecast_details.json',{str(k):v['details'] for k,v in horizon.items()})
    print({'status':report['status'],'tracking_ID_switches':tracked.get('ID_switches',0),'baseline_ID_switches':baseline.get('ID_switches',0),'velocity':report['velocity'],'horizons':summary,'quality_gates':quality})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'data/sunny-cross-town-v3-007')
    for k in ['motion','world','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
