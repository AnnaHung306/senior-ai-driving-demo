"""Visible-instance label geometry, independent of simulator and learned inference.

CARLA 0.9.16 Tagger.cpp: R=semantic, G=actor low byte, B=actor high byte.
All pixel boxes use xyxy with exclusive maxima; projected boxes are continuous.
"""
from itertools import combinations, product

import numpy as np


TAGS = {'car': {14}, 'truck': {15}, 'bus': {16}, 'motorcycle': {18, 13},
        'bicycle': {19, 13}, 'pedestrian': {12}, 'traffic_light': {7}}


def decode_instance_rgb(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or (rgb[..., 0] > 28).any():
        raise ValueError('Expected raw instance RGB uint8: R class, G low ID, B high ID')
    return rgb[..., 0], rgb[..., 1].astype(np.uint16) + rgb[..., 2].astype(np.uint16) * 256


def validate_actor_ids(actors):
    ids = [a['actor_id'] for a in actors]
    if any(not isinstance(i, int) or i <= 0 or i > 65535 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Ambiguous/truncated actor IDs; require unique IDs in 1..65535')
    if any(a['kind'] not in TAGS for a in actors):
        raise ValueError('Unverified actor taxonomy')


def mask_bbox(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if x.size else None


def project_box(vertices_world, world_from_camera, k, width, height, near=0.1):
    vertices = np.asarray(vertices_world, dtype=np.float64)
    transform = np.asarray(world_from_camera, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if vertices.shape != (8, 3) or transform.shape != (4, 4) or k.shape != (3, 3) or near <= 0:
        raise ValueError('Invalid box/camera geometry')
    if not all(np.isfinite(a).all() for a in [vertices, transform, k]):
        raise ValueError('Nonfinite box/camera geometry')
    camera = (np.linalg.inv(transform) @ np.c_[vertices, np.ones(8)].T).T[:, :3]
    optical = camera[:, [1, 2, 0]] * [1, -1, 1]
    points = [p for p in optical if p[2] >= near]
    # All vertex chords include every box edge. Additional intersections remain
    # within the convex clipped box and cannot enlarge projected coordinate extrema.
    for a, b in combinations(optical, 2):
        if (a[2] < near) != (b[2] < near):
            points.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    clipped_near = bool((optical[:, 2] < near).any() and (optical[:, 2] >= near).any())
    if not points:
        return {'bbox_xyxy': None, 'unclipped_bbox_xyxy': None, 'near_plane_clipped': False, 'outside_view': True}
    p = np.asarray(points)
    uv = (k @ p.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    lo, hi = uv.min(0), uv.max(0)
    raw = [*lo.tolist(), *hi.tolist()]
    bound = [max(0., raw[0]), max(0., raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
    outside = bound[2] <= bound[0] or bound[3] <= bound[1]
    return {'bbox_xyxy': None if outside else bound, 'unclipped_bbox_xyxy': raw,
            'near_plane_clipped': clipped_near, 'outside_view': outside}


def depth_box_support(mask, depth, k, world_from_camera, world_from_box, extent, tolerance=0.15):
    """Compare axial-depth and Euclidean-range hypotheses against a known OBB.

    This is a geometric consistency diagnostic, not a claimed occlusion fraction.
    """
    y, x = np.nonzero(mask)
    if not x.size:
        return {'pixels': 0, 'axial_in_box_fraction': None, 'range_in_box_fraction': None}
    if depth.shape != mask.shape or not np.isfinite(depth).all():
        raise ValueError('Invalid depth image')
    rays = np.stack((np.ones_like(x), (x + .5 - k[0][2]) / k[0][0],
                     -(y + .5 - k[1][2]) / k[1][1]), 1)
    transform = np.linalg.inv(np.asarray(world_from_box)) @ np.asarray(world_from_camera)
    output = {'pixels': int(x.size)}
    for name, distance in [('axial', depth[y, x]), ('range', depth[y, x] / np.linalg.norm(rays, axis=1))]:
        local = (transform @ np.c_[rays * distance[:, None], np.ones(x.size)].T).T[:, :3]
        inside = (np.abs(local) <= np.asarray(extent) + tolerance).all(1)
        output[name + '_in_box_fraction'] = float(inside.mean())
    return output


def visible_candidate(actor, classes, ids, semantic, depth, camera, near=0.1):
    mask = ids == actor['actor_id']
    tags, counts = np.unique(classes[mask], return_counts=True)
    bbox = mask_bbox(mask)
    allowed = TAGS[actor['kind']]
    invalid_tags = [int(t) for t in tags if int(t) not in allowed]
    projection = project_box(actor['world_vertices_m'], camera['T_world_from_camera'], camera['K'],
                             ids.shape[1], ids.shape[0], near)
    tolerance = 0.15
    local = np.array(list(product(*[[-float(e) - tolerance, float(e) + tolerance] for e in actor['extent_m']])))
    expanded = (np.asarray(actor['T_world_from_box']) @ np.c_[local, np.ones(8)].T).T[:, :3]
    support_envelope = project_box(expanded, camera['T_world_from_camera'], camera['K'], ids.shape[1], ids.shape[0], near)
    result = {'source': 'simulator_ground_truth_candidate_not_training_approved',
        'actor_id': actor['actor_id'], 'kind': actor['kind'], 'visible_pixels': int(mask.sum()),
        'semantic_pixels_by_tag': {str(int(t)): int(n) for t, n in zip(tags, counts)},
        'unexpected_semantic_tags': invalid_tags,
        'instance_semantic_disagreement_pixels': int((classes[mask] != semantic[mask]).sum()),
        'visible_bbox_xyxy': bbox, 'amodal_projection': projection,
        'projection_support_envelope': support_envelope, 'geometry_tolerance_m': tolerance,
        'status': 'not_visible' if bbox is None else 'candidate_requires_policy_and_qa',
        'occlusion': 'unknown; bbox fill is not physical visibility fraction'}
    if bbox is not None:
        result['touches_image_boundary'] = bbox[0] == 0 or bbox[1] == 0 or bbox[2] == ids.shape[1] or bbox[3] == ids.shape[0]
        result['depth_geometry'] = depth_box_support(mask, depth, camera['K'], camera['T_world_from_camera'],
                                                     actor['T_world_from_box'], actor['extent_m'], tolerance=tolerance)
        projected = projection['bbox_xyxy']
        result['visible_pixels_per_projected_bbox_area_not_visibility'] = (float(mask.sum()) /
            ((projected[2] - projected[0]) * (projected[3] - projected[1]))) if projected else None
    return result
