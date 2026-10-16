"""Visible-box AP with 101-point interpolated recall and explicit ignore policy.

AP@[.50:.95] uses 10 IoU thresholds, at most 100 predictions/image, all areas.
It is not advertised as the complete COCO evaluator (no area/crowd breakdown).
"""
import numpy as np
from .visibility_policy import ignore_background_proposals


def iou(box, boxes):
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4); box=np.asarray(box,dtype=float)
    intersection=np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])).prod(1)
    return intersection/np.maximum(1e-12,np.prod(box[2:]-box[:2])+(boxes[:,2:]-boxes[:,:2]).prod(1)-intersection)


def evaluate_detection(images, class_ids, score_threshold=.5):
    result={}; thresholds=np.linspace(.5,.95,10)
    for name,category in class_ids.items():
        targets_per_image={i:[g for g in image['targets'] if g['label']==category] for i,image in enumerate(images)}
        def size_key(g):
            b=g['box']; area=(b[2]-b[0])*(b[3]-b[1]); return 'small_lt_32_squared' if area<32**2 else ('medium_32_to_96_squared' if area<96**2 else 'large_ge_96_squared')
        strata={key:{'ground_truth':0,'matched_at_score':0} for key in ['small_lt_32_squared','medium_32_to_96_squared','large_ge_96_squared','truncated','not_truncated']}
        for values in targets_per_image.values():
            for g in values:
                strata[size_key(g)]['ground_truth']+=1;strata['truncated' if g.get('truncated',False) else 'not_truncated']['ground_truth']+=1
        gt_total=sum(sum(g['label']==category for g in image['targets']) for image in images)
        predictions=[]
        for index,image in enumerate(images):
            for rank,p in enumerate(sorted(image['predictions'],key=lambda p:-p['score'])[:100]):
                if p['label']==category: predictions.append((p['score'],index,rank,p['box']))
        predictions.sort(key=lambda v:(-v[0],v[1],v[2]))
        cached=[]
        for score,index,rank,box in predictions:
            image=images[index]
            overlaps=iou(box,[g['box'] for g in image['targets'] if g['label']==category])
            known=iou(box,[g['box'] for g in image['targets']])
            ignored=ignore_background_proposals([box],[0],image.get('ignore_boxes',[]))[0]==-1
            cached.append((score,index,overlaps,float(max(known,default=0)),ignored))
        aps=[]; point=None
        for threshold in thresholds:
            matched={i:set() for i in range(len(images))}; tp=[]; fp=[]; scores=[]; ignored_count=0
            for score,index,overlaps,known_overlap,ignored in cached:
                eligible=[i for i,v in enumerate(overlaps) if v>=threshold and i not in matched[index]]
                if eligible:
                    best=max(eligible,key=lambda i:overlaps[i]); matched[index].add(best)
                    tp.append(1); fp.append(0); scores.append(score)
                    if threshold==thresholds[0] and score>=score_threshold:
                        g=targets_per_image[index][best]
                        strata[size_key(g)]['matched_at_score']+=1
                        strata['truncated' if g.get('truncated',False) else 'not_truncated']['matched_at_score']+=1
                elif known_overlap<threshold and ignored:
                    ignored_count+=1
                else:
                    tp.append(0); fp.append(1); scores.append(score)
            ctp=np.cumsum(tp); cfp=np.cumsum(fp)
            recall=ctp/max(1,gt_total); precision=ctp/np.maximum(1,ctp+cfp)
            envelope=np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
            ap=float(np.mean([max(envelope[recall>=r],default=0.) for r in np.linspace(0,1,101)])) if gt_total else None
            aps.append(ap)
            if threshold==thresholds[0]:
                selected=np.asarray(scores)>=score_threshold; t=int(np.asarray(tp)[selected].sum()); f=int(np.asarray(fp)[selected].sum())
                point={'precision_at_score':t/(t+f) if t+f else 0.,'recall_at_score':t/gt_total if gt_total else None,
                       'true_positives':t,'false_positives':f,'false_negatives':gt_total-t,'ignored_predictions_all_scores':ignored_count}
        for bucket in strata.values(): bucket['recall']=bucket['matched_at_score']/bucket['ground_truth'] if bucket['ground_truth'] else None
        result[name]={'ground_truth_objects':gt_total,'AP50':aps[0], 'AP50_95':float(np.mean(aps)) if gt_total else None,**point,
                      'recall_strata_IoU50':strata}
    valid=[c for c in result.values() if c['ground_truth_objects']]
    return {'protocol':'visible-tight-xyxy-101recall-IoU50:95-max100-allarea-ignore-v1', 'score_threshold':score_threshold,
            'images':len(images),'per_class':result,'all_classes_present':len(valid)==len(class_ids),
            'mAP50':float(np.mean([c['AP50'] for c in valid])) if valid else None,
            'mAP50_95':float(np.mean([c['AP50_95'] for c in valid])) if valid else None}
