"""Independent post-run accuracy/performance evaluation; GT never reaches drive()."""
import argparse,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json
from senior_ai.phase10_metrics import semantic_confusion,visible_targets,summary,calibration_bins
from senior_ai.sunny_data import SEGMENTATION_NAMES,DETECTION_CLASSES
from senior_ai.detection_metrics import evaluate_detection,iou
from senior_ai.cuboid import box_iou,wrap
from senior_ai.motion_forecast import assignment
from senior_ai.visibility_policy import ignore_background_proposals


def pairs(pred,truth,by_class=True):
    cost=np.full((len(pred),len(truth)),np.inf)
    for i,p in enumerate(pred):
        for j,g in enumerate(truth):
            d=float(np.linalg.norm(np.array(p['center_ego_m'][:2])-g['center_ego_m'][:2]))
            if d<=2 and (not by_class or p['class_id']==g['class_id']):cost[i,j]=d
    return assignment(cost,3.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    conf=np.zeros((17,17),np.int64);ignored_pixels=0;images=[];examples=[];run_rows=[];timing=[];sources={};ambiguities=Counter()
    localization=[];prior_rows=[];motion=[];velocity_rows=[];future_rows=[];calibration=[];counts=Counter();ignore_counts=Counter();track_history={};switches=[];reacquisitions=[]
    for folder in sorted((ROOT/'artifacts/runs').glob('phase10-drive-*')):
        if not (folder/'report.json').exists():continue
        report=read(folder/'report.json');contract=read(folder/'contract.json');record_by_frame={}
        if 'records' not in report:continue
        sources[str((folder/'report.json').relative_to(ROOT))]=sha(folder/'report.json')
        sources[str((folder/'contract.json').relative_to(ROOT))]=sha(folder/'contract.json')
        rows=[read(folder/r['file']) for r in report['records']]
        for entry,row in zip(report['records'],rows):
            assert sha(folder/entry['file'])==entry['sha256'];record_by_frame[row['frame']]=row
        run_rows.append({'run_id':folder.name,'status':report['status'],'passed':report['engineering_route_passed'],
            'frames':len(rows),'wall_s':report['wall_s'],'sim_s':report['simulation_s'],'checks':report['checks'],
            'distance_m':report['route']['length_m'],'events':report['events'],
            'goal_distance_m':rows[-1]['navigation']['goal_distance_m'] if rows else None,
            'max_cross_track_m':max([r['navigation']['cross_track_m'] for r in rows],default=None),
            'state_counts':dict(Counter(r['plan']['state'] for r in rows)),
            'cleanup_speed_mps':report.get('cleanup_final_speed_mps'),
            'observer_enabled':contract.get('evaluation_only_observer') is not None,
            'startup':read(folder/'startup-timing.json') if (folder/'startup-timing.json').exists() else None})
        if (folder/'timing.jsonl').exists():
            for line in (folder/'timing.jsonl').read_text().splitlines():
                t=json.loads(line);timing.append({'run_id':folder.name,'GT_observer':contract.get('evaluation_only_observer') is not None,**t})
        if not (folder/'audit').exists():continue
        gt_rows=[read(p) for p in sorted((folder/'audit').glob('[0-9]*.json'))]
        gt_by_frame={r['frame']:r for r in gt_rows}
        for g in gt_rows:
            if not g['sampled_images']:continue
            r=record_by_frame[g['frame']];sample=folder/'audit'/f'{g["frame"]:08d}';visible={};frame_tp=0;frame_fn=0;frame_fp=0
            assert abs(g['timestamp_s']-r['timestamp_s'])<1e-5
            actor_by_id={v['actor_id']:v for v in g['actors']}
            for camera,item in g['cameras'].items():
                ip=sample/f'{camera}-instance.png';sp=sample/f'{camera}-predicted_seg.png'
                assert sha(ip)==item['instance_sha256'] and sha(sp)==item['predicted_seg_sha256']
                assert sha(folder/item['RGB']['file'])==item['RGB']['sha256']
                raw=np.array(Image.open(ip));seg=np.array(Image.open(sp));cm,ig=semantic_confusion(raw[:,:,0],seg);conf+=cm;ignored_pixels+=ig
                targets,ignore,ambiguous=visible_targets(raw,set(actor_by_id));ambiguities.update(x['reason'] for x in ambiguous)
                image={'run_id':folder.name,'frame':g['frame'],'camera':camera,'targets':targets,
                       'predictions':item['detections'],'ignore_boxes':ignore};images.append(image)
                m=evaluate_detection([image],DETECTION_CLASSES)
                for result in m['per_class'].values():
                    frame_tp+=result['true_positives'];frame_fp+=result['false_positives'];frame_fn+=result['false_negatives']
                for target in targets:
                    if target['actor_id'] is not None:visible[target['actor_id']]=target['label']
                # Reliability of score>=.5, class-aware greedy; no ignored-background score claims.
                used=set()
                for d in sorted(item['detections'],key=lambda d:-d['score']):
                    if d['score']<.5:continue
                    candidates=[(float(iou(d['box'],[t['box']])[0]),j) for j,t in enumerate(targets) if j not in used and d['label']==t['label']]
                    overlap,j=max(candidates,default=(0,None));correct=overlap>=.5
                    if correct:used.add(j)
                    calibration.append({'score':d['score'],'correct':bool(correct),'run_id':folder.name})
                denom=cm.sum(0)+cm.sum(1)-cm.diagonal();road=float(cm[1,1]/denom[1]) if denom[1] else None
                examples.append({'run_id':folder.name,'frame':g['frame'],'camera':camera,'RGB':item['RGB']['file'],
                    'road_iou':road,'detection_FP':sum(v['false_positives'] for v in m['per_class'].values()),
                    'detection_FN':sum(v['false_negatives'] for v in m['per_class'].values()),
                    'sample_path':str(sample.relative_to(ROOT))})
            truth=[{**actor_by_id[identifier],'class_id':label} for identifier,label in visible.items()
                   if np.linalg.norm(actor_by_id[identifier]['center_ego_m'][:2])<=30]
            for target in truth:
                for horizon in [1.,2.]:
                    later=next((x for x in gt_rows if abs(x['timestamp_s']-g['timestamp_s']-horizon)<.051),None)
                    if later is None:counts[f'all_GT_right_censored_{horizon}']+=1
                    elif any(x['actor_id']==target['actor_id'] for x in later['actors']):counts[f'all_GT_future_eligible_{horizon}']+=1
                    else:counts[f'all_GT_future_actor_absent_{horizon}']+=1
            pred=r['perception']['objects'];paired=pairs(pred,truth)
            counts.update(GT=len(truth),predictions=len(pred),TP=len(paired),FP=len(pred)-len(paired),FN=len(truth)-len(paired))
            paired_predictions={i for i,_ in paired};unknown_predictions=0
            for index,obj in enumerate(pred):
                if index in paired_predictions:continue
                source_unknown=[]
                for source in obj.get('sources',[]):
                    im=next(x for x in images[-8:] if x['camera']==source['camera']);box=source['bbox_xyxy']
                    known=max(iou(box,[t['box'] for t in im['targets']]),default=0.)>=.5
                    source_unknown.append(not known and ignore_background_proposals([box],[0],im['ignore_boxes'])[0]==-1)
                if source_unknown and all(source_unknown):unknown_predictions+=1
            ignore_counts.update(GT=len(truth),TP=len(paired),FN=len(truth)-len(paired),
                unassessed_predictions_with_only_ignored_foreground_sources=unknown_predictions,
                FP=len(pred)-len(paired)-unknown_predictions,predictions_scored=len(pred)-unknown_predictions)
            class_free=pairs(pred,truth,False);counts['class_agnostic_TP']+=len(class_free)
            for i,j in paired:
                obj,t=pred[i],truth[j];center=float(np.linalg.norm(np.array(obj['center_ego_m'])-t['center_ego_m']))
                upright=t.get('upright_for_yaw_IoU',True)
                _,i3=box_iou(obj,t) if upright else (None,None)
                counts['cuboid_TP_IoU25']+=bool(i3 is not None and i3>=.25)
                counts['matched_non_upright_IoU_unassessed']+=not upright
                localization.append({'run_id':folder.name,'frame':g['frame'],'class_id':t['class_id'],'actor_id':t['actor_id'],
                    'center_3D_error_m':center,'IoU3D':i3,'yaw_error_deg':float(abs(wrap(obj['yaw_rad']-t['yaw_rad']))*180/np.pi) if upright else None,
                    'size_abs_error_m':np.abs(np.array(obj['size_lwh_m'])-t['size_lwh_m']).tolist()})
                key=(folder.name,t['actor_id']);track=obj['track_id']
                if key in track_history and track_history[key][0]!=track:
                    target=switches if g['frame']-track_history[key][1]==10 else reacquisitions
                    target.append({'run_id':folder.name,'frame':g['frame'],'actor_id':t['actor_id']})
                track_history[key]=(track,g['frame'])
                mo=obj.get('motion',{})
                if mo.get('velocity_available'):
                    velocity=mo['velocity_ground_ego_mps'];error=float(np.linalg.norm(np.array(velocity[:2])-t['velocity_ego_mps'][:2]));motion.append(error)
                    velocity_rows.append({'run_id':folder.name,'frame':g['frame'],'actor_id':t['actor_id'],'class_id':t['class_id'],
                        'vector_error_mps':error,'true_ground_speed_mps':float(np.linalg.norm(t['velocity_ego_mps'][:2]))})
                forecast=obj.get('forecast',{});times=np.array(forecast.get('times_s',[]));future=forecast.get('positions_ego_xy_m',[])
                for horizon in [1.,2.]:
                    future_gt=[row for row in gt_rows if abs(row['timestamp_s']-g['timestamp_s']-horizon)<.051]
                    if not future_gt:
                        counts[f'right_censored_{horizon}']+=1;continue
                    target=next((x for x in future_gt[0]['actors'] if x['actor_id']==t['actor_id']),None)
                    if target is None:counts[f'future_actor_absent_{horizon}']+=1;continue
                    counts[f'eligible_matched_{horizon}']+=1
                    k=np.where(np.isclose(times,horizon))[0]
                    if forecast.get('status')!='available' or not len(k):continue
                    transform=np.linalg.inv(np.array(g['ego_world_from_reference']))
                    xy=(transform@np.r_[target['center_world_m'],1])[:2]
                    errors=[]
                    for ti,position in zip(times,future):
                        if ti>horizon:continue
                        later=next((x for x in gt_rows if abs(x['timestamp_s']-g['timestamp_s']-ti)<.051),None)
                        later_actor=next((x for x in later['actors'] if x['actor_id']==t['actor_id']),None) if later else None
                        if later_actor is None:errors=[];break
                        truth_xy=(transform@np.r_[later_actor['center_world_m'],1])[:2]
                        errors.append(float(np.linalg.norm(np.asarray(position)-truth_xy)))
                    future_rows.append({'run_id':folder.name,'frame':g['frame'],'horizon_s':horizon,
                        'FDE_m':float(np.linalg.norm(np.array(future[int(k[0])])-xy)),
                        'ADE_m':float(np.mean(errors)) if errors else None,
                        'constant_position_FDE_m':float(np.linalg.norm(np.array(obj['center_ego_m'][:2])-xy))})
            # Same candidate crops, 2D GT association, no nearest-3D selection bias.
            assert len(g['single_3D'])==len(g['prior_3D'])
            for learned,prior in zip(g['single_3D'],g['prior_3D']):
                assert learned['camera']==prior['camera'] and learned['detection_index']==prior['detection_index'] and learned['bbox_xyxy']==prior['bbox_xyxy']
                im=next(x for x in images[-8:] if x['camera']==learned['camera'])
                values=[(float(iou(learned['bbox_xyxy'],[t['box']])[0]),t) for t in im['targets'] if t['actor_id'] in actor_by_id and t['label']==learned['class_id']]
                if not values:continue
                overlap,t=max(values,key=lambda x:x[0])
                if overlap<.5:continue
                gt=actor_by_id[t['actor_id']]
                if not gt.get('upright_for_yaw_IoU',True):continue
                prior_rows.append({'run_id':folder.name,'frame':g['frame'],'actor_id':t['actor_id'],
                    'learned_center_error_m':float(np.linalg.norm(np.array(learned['center_ego_m'])-gt['center_ego_m'])),
                    'prior_center_error_m':float(np.linalg.norm(np.array(prior['center_ego_m'])-gt['center_ego_m'])),
                    'learned_IoU3D':box_iou(learned,gt)[1],'prior_IoU3D':box_iou(prior,gt)[1]})
    per_class=[]
    for i,name in enumerate(SEGMENTATION_NAMES):
        union=int(conf[i].sum()+conf[:,i].sum()-conf[i,i]);per_class.append({'id':i,'name':name,'support_pixels':int(conf[i].sum()),
            'predicted_pixels':int(conf[:,i].sum()),'IoU':float(conf[i,i]/union) if union else None})
    focus=[x['IoU'] for x in per_class if x['id'] not in [0,12] and x['support_pixels'] and x['IoU'] is not None]
    detection=evaluate_detection(images,DETECTION_CLASSES) if images else None
    strict_images=[{**im,'ignore_boxes':[]} for im in images]
    performance={}
    for label,selected in [('all',timing),('with_GT',[t for t in timing if t['GT_observer']]),('without_GT',[t for t in timing if not t['GT_observer']])]:
        performance[label]={k:summary([t[k] for t in selected]) for k in (timing[0].keys() if timing else []) if k.endswith('_ms')}
    report={'status':'completed_bounded_Phase10_post_run_audit','runs':run_rows,'sampled_eight_RGB_bundles':len(images)//8,
        'segmentation':{'confusion_true_rows_pred_columns':conf.tolist(),'per_class':per_class,'focus_mIoU_excluding_other_and_wall':float(np.mean(focus)) if focus else None,'ignored_pixels':ignored_pixels},
        'detection':detection,'detection_strict_no_ignore':evaluate_detection(strict_images,DETECTION_CLASSES) if images else None,
        'ambiguous_ignore_reasons':dict(ambiguities),'score_reliability_unignored_strict':calibration_bins(calibration),
        'world3D_counts':dict(counts),'world3D_ignoreAware_actor_GT_supplement':dict(ignore_counts),
        'world3D_center_error_m':summary([v['center_3D_error_m'] for v in localization]),
        'world3D_IoU':summary([v['IoU3D'] for v in localization if v['IoU3D'] is not None]),'world3D_yaw_error_deg':summary([v['yaw_error_deg'] for v in localization if v['yaw_error_deg'] is not None]),
        'tracking_sampled_consecutive_ID_switches':len(switches),'tracking_gap_reacquisition_ID_changes':len(reacquisitions),'velocity_vector_error_mps':summary(motion),
        'velocity_dynamic_GT_over_half_mps':summary([v['vector_error_mps'] for v in velocity_rows if v['true_ground_speed_mps']>.5]),
        'velocity_stationary_GT_under_point2_mps':summary([v['vector_error_mps'] for v in velocity_rows if v['true_ground_speed_mps']<.2]),
        'forecast_horizons':{str(h):{metric:summary([f[metric] for f in future_rows if f['horizon_s']==h and f[metric] is not None]) for metric in ['FDE_m','ADE_m','constant_position_FDE_m']} for h in [1.,2.]},
        'same_2D_TP_crop_3D_ablation':{k:summary([v[k] for v in prior_rows]) for k in ['learned_center_error_m','prior_center_error_m','learned_IoU3D','prior_IoU3D']},
        'performance':performance,'sources':sources,'tool_sha256':sha(Path(__file__)),'formal_quality_passed':False,
        'limits':['Fresh same-town audit routes, not old development/lockbox or heldout assets.',
          'GT observer overhead is separated; semantic per-image accuracy does not establish BEV depth/occupancy accuracy.',
          'Visible-box minimum16px/side4 diagnostic; no depth-based projection-support gate, not directly comparable to old dataset policy.',
          'Dynamic GT actor visibility requires raw ID and capture snapshot. Traffic-light rendered instance masks only support 2D, not signal-state labels.',
          'Static environment cars/trucks can have semantic masks without CARLA vehicle actors. Strict actor-only unmatched counts are not all certified false positives; ignore-aware supplemental counts retain these unassessed sources.',
          'Sampled tracking switches are not full-rate MOT; FDE conditional coverage/right-censoring explicit, no all-GT success claim.',
          'No-support classes, no matched dynamics, no future coverage remain unmeasured, never perfect.']}
    write_json(a.output/'report.json',report);write_json(a.output/'per_image.json',images);write_json(a.output/'examples.json',examples)
    write_json(a.output/'matched_3D.json',localization);write_json(a.output/'crop_ablation.json',prior_rows);write_json(a.output/'timing.json',timing)
    write_json(a.output/'forecast_samples.json',future_rows)
    write_json(a.output/'velocity_samples.json',velocity_rows)
    print({'runs':len(run_rows),'images':len(images),'counts3D':dict(counts),'output':str(a.output)})

if __name__=='__main__':main()
