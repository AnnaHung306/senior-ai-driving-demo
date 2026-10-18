"""Use the detector's positive-first ignore contract for joint phase scoring."""
from .detection_metrics import iou
from .visibility_policy import ignore_background_proposals

JOINT_PROTOCOL='RGB-phase-score0.5-IoU50-positive-first-ignore-v2'


def ignored_unmatched_light(box,all_targets,ignore_boxes):
    # A wrong-class/duplicate hypothesis over ANY known positive remains FP.
    # Only unmatched background can be ignored, exactly as visible-box AP50.
    known=max(iou(box,[t['box'] for t in all_targets]),default=0.)
    return known<.5 and ignore_background_proposals([box],[0],ignore_boxes)[0]==-1
