"""RGB crop 3D regression and upright cuboids; no simulator/GT readers."""
from itertools import product
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn

CLASSES = {1: 'car', 2: 'pedestrian', 3: 'bicycle', 4: 'motorcycle', 5: 'truck', 6: 'bus'}
SIGNS = np.array(list(product([-1, 1], repeat=3)), dtype=np.float64)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1]


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def corners(box):
    center = np.asarray(box['center_ego_m'], dtype=np.float64)
    size = np.asarray(box['size_lwh_m'], dtype=np.float64)
    yaw = float(box['yaw_rad'])
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[center, size, yaw]).all() or (size <= 0).any():
        raise ValueError('Finite center, positive full LWH and radian yaw required')
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (SIGNS * size / 2) @ rotation.T + center


def footprint(box):
    points = corners(box)[[0, 4, 6, 2], :2]
    return points.astype(np.float32)


def box_iou(a, b):
    pa, pb = footprint(a), footprint(b)
    area, _ = cv2.intersectConvexConvex(pa, pb)
    area = max(0., float(area))
    sa, sb = np.asarray(a['size_lwh_m']), np.asarray(b['size_lwh_m'])
    ca, cb = np.asarray(a['center_ego_m']), np.asarray(b['center_ego_m'])
    overlap_z = max(0., min(ca[2]+sa[2]/2, cb[2]+sb[2]/2)-max(ca[2]-sa[2]/2, cb[2]-sb[2]/2))
    intersection = min(area, sa[0]*sa[1], sb[0]*sb[1]) * overlap_z
    return float(area/max(1e-9, sa[0]*sa[1]+sb[0]*sb[1]-area)), float(intersection/max(1e-9, np.prod(sa)+np.prod(sb)-intersection))


def crop(rgb, bbox, size=64):
    x0, y0, x1, y1 = bbox
    h, w = rgb.shape[:2]
    bounds = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Empty crop')
    return np.asarray(Image.fromarray(rgb).crop(bounds).resize((size, size), Image.Resampling.BILINEAR)).transpose(2, 0, 1).copy()


def geometry_features(camera, bbox, class_id, max_range=30.):
    b = np.asarray(bbox, dtype=np.float64)
    if class_id not in CLASSES or b.shape != (4,) or not np.isfinite(b).all():
        return None
    x0, y0, x1, y1 = b
    if x0 <= 1 or y0 <= 1 or x1 >= camera.width-1 or y1+2 >= camera.height-1 or min(x1-x0, y1-y0) < 6:
        return None
    anchor, valid = camera.ground([[(x0+x1)/2+.5, y1+2+.5]], 0., max_range)
    if not valid[0]:
        return None
    local = (anchor[0]-camera.T[:3, 3]) @ camera.T[:3, :3]
    normalized_box = [((x0+x1)/2+.5-camera.K[0, 2])/camera.K[0, 0],
                      ((y0+y1)/2+.5-camera.K[1, 2])/camera.K[1, 1],
                      (x1-x0)/camera.K[0, 0], (y1-y0)/camera.K[1, 1]]
    category = np.eye(6)[class_id-1]
    features = np.r_[normalized_box, local/25, camera.T[2, 3]/3, category].astype(np.float32)
    return features, local


class CuboidNet(nn.Module):
    """From scratch, ~100k parameters; metadata contains only RGB-derived geometry."""
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in [16, 32, 64, 96]:
            layers += [nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False), nn.GroupNorm(4, width), nn.SiLU()]
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(96+14, 128), nn.SiLU(), nn.Linear(128, 8))

    def forward(self, rgb, geometry):
        return self.head(torch.cat([self.features(rgb).mean((-2, -1)), geometry], dim=1))


def decode(output, anchor_local, camera, class_id, priors, scale=(5., 5., 2.)):
    value = np.asarray(output, dtype=np.float64)
    if value.shape != (8,) or not np.isfinite(value).all():
        raise ValueError('Nonfinite cuboid network output')
    local = anchor_local + np.clip(value[:3], -2, 2)*scale
    size = np.asarray(priors[str(class_id)]) * np.exp(np.clip(value[3:6], -1, 1))
    center = local @ camera.T[:3, :3].T + camera.T[:3, 3]
    heading = camera.T[:3, :3] @ np.array([value[6], value[7], 0.])
    yaw = float(np.arctan2(heading[1], heading[0]))
    return {'class_id': class_id, 'class_name': CLASSES[class_id], 'center_ego_m': center.tolist(),
            'size_lwh_m': size.tolist(), 'yaw_rad': yaw, 'orientation_reliable': bool(np.linalg.norm(value[6:8]) >= .25),
            'output_clipped': bool((np.abs(value[:3]) > 2).any() or (np.abs(value[3:6]) > 1).any()),
            'geometry': 'learned_RGB_cuboid_with_flat_ground_anchor', 'pitch_roll': 'upright_assumption_not_predicted'}


def prior_box(anchor_local, camera, class_id, priors):
    size = np.asarray(priors[str(class_id)])
    # A ground point is not the actor origin. This baseline assumes center above it.
    base = np.zeros(8);base[2] = size[2]/4;base[6] = 1
    result = decode(base, anchor_local, camera, class_id, priors)
    result['geometry'] = 'train_class_size_and_camera_heading_prior_baseline'
    return result


def fuse_cuboids(observations, distance=1.5):
    groups = []
    for obj in sorted(observations, key=lambda o: (-o['score_uncalibrated'], o['camera'], o['detection_index'])):
        found = None
        for group in groups:
            if obj['class_id'] != group[0]['class_id'] or any(g['camera'] == obj['camera'] for g in group):
                continue
            if all(np.linalg.norm(np.array(obj['center_ego_m'])[:2]-np.array(g['center_ego_m'])[:2]) <= distance for g in group):
                found = group;break
        if found is None:
            groups.append([obj])
        else:
            found.append(obj)
    result = []
    for group in groups:
        weights = np.array([g['score_uncalibrated'] for g in group])
        angles = np.array([g['yaw_rad'] for g in group])
        vector = np.average(np.c_[np.cos(angles), np.sin(angles)], axis=0, weights=weights)
        result.append({'class_id': group[0]['class_id'], 'class_name': group[0]['class_name'],
                       'center_ego_m': np.average([g['center_ego_m'] for g in group], axis=0, weights=weights).tolist(),
                       'size_lwh_m': np.average([g['size_lwh_m'] for g in group], axis=0, weights=weights).tolist(),
                       'yaw_rad': float(np.arctan2(vector[1], vector[0])),
                       'orientation_reliable': bool(np.linalg.norm(vector) >= .5 and all(g['orientation_reliable'] for g in group)),
                       'score_uncalibrated': float(weights.max()), 'sources': group,
                       'geometry': 'fused_learned_RGB_upright_cuboid', 'output_clipped': any(g['output_clipped'] for g in group)})
    return result
