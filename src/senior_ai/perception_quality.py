"""Frozen validation/heldout gates shared with validation-only model selection."""
from .sunny_data import SEGMENTATION_NAMES


def quality(task,result,gates):
    if task=='detection':
        checks={'all_classes_present':result['all_classes_present'],'mAP50':(result['mAP50'] or 0)>=gates['detection_mAP50']}
        for name,v in result['per_class'].items():
            checks[f'{name}/recall']=(v['recall_at_score'] or 0)>=gates['detection_per_class_recall_at_score_0_5']
            if 'detection_per_class_precision_at_score_0_5' in gates:
                checks[f'{name}/precision']=(v['precision_at_score'] or 0)>=gates['detection_per_class_precision_at_score_0_5']
    else:
        checks={'focus_mIoU':(result['focus_macro_iou'] or 0)>=gates['segmentation_focus_mIoU'],
                'all_focus_present':result['focus_supported_classes']==len(SEGMENTATION_NAMES)-1}
        for row in result['classes'][1:]:checks[row['name']+'/IoU']=(row['iou'] or 0)>=gates['segmentation_per_focus_class_IoU']
        if 'minimum_visible_margin_f1' in gates:checks['visible_margin_f1']=(result['visible_margin']['f1'] or 0)>=gates['minimum_visible_margin_f1']
    return {'passed':all(checks.values()),'checks':checks}


def selection_key(score,qualified,policy):
    if policy not in {'score_only','qualified_first_then_score'}:raise ValueError('Unknown validation selection policy')
    return (int(qualified) if policy=='qualified_first_then_score' else 0,score)
