"""Separate class confusion from visible-box localization using saved predictions."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
from senior_ai.core import write_json
from senior_ai.detection_metrics import iou
from senior_ai.visibility_policy import ignore_background_proposals

def main():
    source=ROOT/'artifacts/runs/phase10-evaluation-final-v1-002/per_image.json';out=ROOT/'artifacts/reports/phase10-detection-confusion-final.json'
    if out.exists():raise FileExistsError(out)
    matrix=np.zeros((9,9),int);wrong=[];ignored=0;gt_count=0;localized=0
    for image in read(source):
        truth=image['targets'];matched=set();gt_count+=len(truth)
        for p in sorted(image['predictions'],key=lambda p:-p['score']):
            if p['score']<.5:continue
            values=iou(p['box'],[t['box'] for t in truth]);candidates=[j for j,v in enumerate(values) if v>=.5 and j not in matched]
            if candidates:
                j=max(candidates,key=lambda j:values[j]);matched.add(j);t=truth[j];matrix[t['label']-1,p['label']-1]+=1;localized+=1
                if t['label']!=p['label']:wrong.append({'run_id':image['run_id'],'frame':image['frame'],'camera':image['camera'],'true_class':t['label'],'predicted_class':p['label'],'score':p['score'],'IoU':float(values[j]),'true_box':t['box'],'predicted_box':p['box']})
            elif max(values,default=0)<.5 and ignore_background_proposals([p['box']],[0],image['ignore_boxes'])[0]==-1:ignored+=1
            else:matrix[8,p['label']-1]+=1
        for j,t in enumerate(truth):
            if j not in matched:matrix[t['label']-1,8]+=1
    write_json(out,{'status':'completed_class_agnostic_localization_and_class_confusion_supplement',
        'class_order':['car','pedestrian','bicycle','motorcycle','truck','bus','traffic_light','temporary_obstacle','background_or_missing'],
        'confusion_true_rows_pred_columns':matrix.tolist(),'GT_objects':gt_count,'class_agnostic_localized_IoU50':localized,
        'wrong_class_localized':len(wrong),'ignored_unknown_foreground_predictions':ignored,'wrong_class_examples':wrong,
        'source_sha256':sha(source),'tool_sha256':sha(Path(__file__)),
        'limits':'Greedy score-ordered class-agnostic IoU50 diagnostic; does not replace class-aware AP or certify ignored/static geometry.'})
    print({'GT':gt_count,'localized':localized,'wrong_class':len(wrong),'car_confusion_row':matrix[0].tolist()})

if __name__=='__main__':main()
