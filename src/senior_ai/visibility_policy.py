"""Versioned visible boxes and explicit ignore handling, never hidden GT inference."""
import numpy as np
from .instance_labels import mask_bbox


def classify_candidate(candidate, policy):
    if not candidate['visible_pixels']:
        return 'not_visible', []
    reasons = []
    box = candidate['visible_bbox_xyxy']
    if candidate['visible_pixels'] < policy['min_visible_pixels']:
        reasons.append('too_few_visible_pixels')
    if min(box[2] - box[0], box[3] - box[1]) < policy['min_bbox_side_pixels']:
        reasons.append('too_small_visible_bbox')
    if candidate['unexpected_semantic_tags']:
        reasons.append('ambiguous_instance_taxonomy')
    if candidate['instance_semantic_disagreement_pixels'] / candidate['visible_pixels'] > policy['max_instance_semantic_disagreement_fraction']:
        reasons.append('semantic_alignment')
    if candidate['amodal_projection']['near_plane_clipped']:
        reasons.append('near_plane')
    projection_key = policy.get('geometry_projection', 'amodal_projection')
    if 'geometry_tolerance_m' in policy and candidate.get('geometry_tolerance_m') != policy['geometry_tolerance_m']:
        raise ValueError('Policy/audit geometry tolerance differs')
    projected = candidate[projection_key]['bbox_xyxy']
    margin = policy['projection_margin_pixels']
    if projected is None or any([box[0] < projected[0] - margin, box[1] < projected[1] - margin,
                                  box[2] > projected[2] + margin, box[3] > projected[3] + margin]):
        reasons.append('mask_outside_projected_geometry')
    if candidate['depth_geometry']['axial_in_box_fraction'] < policy['min_axial_depth_in_box_fraction']:
        reasons.append('depth_geometry')
    body = {'bicycle': '19', 'motorcycle': '18'}.get(candidate['kind'])
    if body and candidate['semantic_pixels_by_tag'].get(body, 0) < policy['min_vehicle_body_pixels_for_two_wheelers']:
        reasons.append('two_wheeler_body_not_visible')
    return ('ignore' if reasons else 'positive'), reasons


def export_view(candidates, classes, ids, policy):
    known = {c['actor_id'] for c in candidates}
    unknown = np.isin(classes, policy['foreground_raw_ids']) & ~np.isin(ids, list(known))
    ignore = unknown.copy()
    positives, ignored, regions, invisible = [], [], [], 0
    for c in candidates:
        status, reasons = classify_candidate(c, policy)
        if status == 'not_visible':
            invisible += 1
            continue
        record = {**c, 'status': status, 'reasons': reasons,
                  'category_id': policy['class_ids'][c['kind']]}
        if status == 'positive':
            positives.append(record)
        else:
            mask = ids == c['actor_id']
            ignore |= mask
            ignored.append(record)
            regions.append({'bbox_xyxy': mask_bbox(mask), 'instance_id': c['actor_id'], 'reason': reasons})
    for instance in np.unique(ids[unknown]):
        mask = unknown & (ids == instance)
        regions.append({'bbox_xyxy': mask_bbox(mask), 'instance_id': int(instance), 'reason': ['unresolved_or_unsupported_foreground'],
                        'semantic_tags': [int(v) for v in np.unique(classes[mask])]})
    return {'source': 'simulator_gt_visible_instance_export', 'policy_version': policy['version'],
            'positive_objects': positives, 'ignored_objects': ignored, 'ignore_regions': regions,
            'ignore_pixels': int(ignore.sum()), 'not_visible_objects': invisible}, ignore.astype(np.uint8)


def ignore_background_proposals(boxes, labels, ignore_boxes, ioa_threshold=.5):
    """Framework-neutral sampler contract. -1 is ignored, 0 background, >0 positive.

    Apply after positive matching at BOTH RPN and ROI sampling. This function alone
    is not a detector integration; a future trainer must demonstrate both call sites.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels).copy()
    ignore_boxes = np.asarray(ignore_boxes, dtype=np.float64).reshape(-1, 4)
    if labels.shape != (len(boxes),) or not 0 < ioa_threshold <= 1:
        raise ValueError('Invalid proposal labels/threshold')
    for array in [boxes, ignore_boxes]:
        if not np.isfinite(array).all() or ((array[:, 2:] - array[:, :2]) <= 0).any():
            raise ValueError('Invalid xyxy boxes')
    if not len(ignore_boxes) or not len(boxes):
        return labels
    lo = np.maximum(boxes[:, None, :2], ignore_boxes[None, :, :2])
    hi = np.minimum(boxes[:, None, 2:], ignore_boxes[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(2)
    ioa = intersection / (boxes[:, 2:] - boxes[:, :2]).prod(1)[:, None]
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = ((center[:, None, :] >= ignore_boxes[None, :, :2]) & (center[:, None, :] < ignore_boxes[None, :, 2:])).all(2)
    labels[(labels == 0) & ((ioa > ioa_threshold) | inside).any(1)] = -1
    return labels
