"""Strict read-only WorldFrame adapter. No GT readers, simulator or model calls."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import numpy as np
from .cuboid import CLASSES, corners
from .bev_geometry import Grid, sample
from .semantic_occupancy import VoxelGrid
from .world_schema import validate_world_frame


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def reject(value):raise ValueError('Nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or '%' in value:
        raise ValueError('Plain safe relative path required')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {'..', '.'} for p in value.split('/')) or str(path) != value:
        raise ValueError('Noncanonical/escaping relative path')
    return path


def strict_world(frame, occupancy):
    validate_world_frame(frame, occupancy)
    if type(frame['frame']) is not int or not isinstance(frame['scene_id'], str) or not frame['scene_id']:
        raise ValueError('Invalid scene/frame identity')
    if frame.get('source') != 'learned_RGB_3D_head_plus_geometric_occupancy':
        raise ValueError('Unapproved prediction provenance')
    for obj in frame['objects']:
        if obj.get('class_name') != CLASSES[obj['class_id']] or type(obj.get('track_age_measurements')) is not int or obj['track_age_measurements'] < 1:
            raise ValueError('Object class/track age inconsistent')
        if not -np.pi-1e-12 <= obj['yaw_rad'] <= np.pi+1e-12:raise ValueError('Radian yaw outside wrapped range')
        for source in obj['sources']:
            corners(source)
            box = np.array(source['bbox_xyxy'], float)
            if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 360):
                raise ValueError('Invalid source RGB bbox')
            if type(source['detection_index']) is not int or source['detection_index'] < 0:raise ValueError('Invalid detection identity')
    held = frame['tracking']['held_unobserved_tracks']
    measured = {o['track_id'] for o in frame['objects']}
    if any(h['track_id'] in measured or h['measurement_current'] is not False for h in held):raise ValueError('Held/current track collision')
    path = frame['road_path']
    if path['status'] not in ['candidate', 'unavailable'] or path['control_issued'] is not False:
        raise ValueError('Unknown path provenance')
    points = np.array(path['points_ego_m'], float)
    if points.size and (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()):raise ValueError('Invalid path points')
    if path['status'] == 'unavailable' and points.size:raise ValueError('Unavailable path contains invented points')
    if not all(np.isfinite(v) and v >= 0 for v in frame['timing_ms'].values()):raise ValueError('Invalid computation timing')
    return True


def boundary_segments(labels, grid):
    """Only road/sidewalk shared edges, never unknown/occlusion contours."""
    result = [];r = grid.resolution_m
    for row, col in np.argwhere(((labels[:, :-1] == 1) & (labels[:, 1:] == 2)) | ((labels[:, :-1] == 2) & (labels[:, 1:] == 1))):
        x = grid.x_max-row*r;y = grid.y_min+(col+1)*r
        result.append([[float(x), float(y)], [float(x-r), float(y)]])
    for row, col in np.argwhere(((labels[:-1] == 1) & (labels[1:] == 2)) | ((labels[:-1] == 2) & (labels[1:] == 1))):
        x = grid.x_max-(row+1)*r;y = grid.y_min+col*r
        result.append([[float(x), float(y)], [float(x), float(y+r)]])
    return result


def display_payload(world, occupancy, bev_arrays, bev_config):
    strict_world(world, occupancy)
    voxel = VoxelGrid(**world['occupancy_grid']);plane = voxel.ground
    ground_layer = int(np.floor((0-voxel.z_min)/voxel.resolution_m))
    labels = occupancy[..., ground_layer].copy()
    labels[~np.isin(labels, [1, 2, 9])] = 255
    source_grid = Grid(**bev_config['grid']);paint_grid = Grid(source_grid.x_min, source_grid.x_max, source_grid.y_min, source_grid.y_max, .2)
    uv = source_grid.xy_to_uv(paint_grid.points()[..., :2]).astype(np.float32)
    probabilities = sample(bev_arrays['temporal'], uv);observed = sample(bev_arrays['observed'].astype(np.uint8), uv, True).astype(bool)
    if not np.isfinite(probabilities).all():raise ValueError('Nonfinite lane evidence')
    paint = observed & (probabilities[..., 1] >= .25) & (probabilities[..., 0] >= .4)
    positions = paint_grid.points()[..., :2][paint]
    fields = ['class_id', 'class_name', 'center_ego_m', 'size_lwh_m', 'yaw_rad', 'track_id', 'track_age_measurements',
              'score_uncalibrated', 'orientation_reliable', 'output_clipped', 'measurement_current']
    objects = [{**{k: o[k] for k in fields}, 'camera_support': sorted(s['camera'] for s in o['sources']),
                'orientation_confidence_calibrated': False} for o in world['objects']]
    return {'schema_version': 1, 'source_mode': 'recorded_inference_replay', 'scene_id': world['scene_id'], 'frame': world['frame'],
            'timestamp_s': world['timestamp_s'], 'coordinate_frame': world['coordinate_frame'], 'axes': world['axes'], 'units': world['units'],
            'objects': objects, 'tracking_status': world['tracking']['reset_reason'], 'held_track_count': len(world['tracking']['held_unobserved_tracks']),
            'ground': {'grid': {k: getattr(plane, k) for k in ['x_min', 'x_max', 'y_min', 'y_max', 'resolution_m']},
                       'shape': list(labels.shape), 'labels_row_major': labels.ravel().tolist(), 'plane_z_m': 0., 'height_measured': False},
            'lane_paint': {'cell_size_m': .2, 'centers_ego_xy_m': positions.tolist(), 'source': 'Phase4_predicted_temporal_line_probability'},
            'road_sidewalk_interfaces': {'segments_ego_xy_m': boundary_segments(labels, plane), 'curb_height_measured': False},
            'local_path': world['road_path'], 'occupancy_grid': world['occupancy_grid'],
            'occupancy_unknown_fraction': float((occupancy == 255).mean()), 'timing_ms': world['timing_ms'],
            'quality': {'formal_quality_passed': False, 'score_calibrated': False, 'fine_geometry': False,
                        'route_destination_available': False, 'velocity_available': False},
            'GT_used_in_inference': False, 'control_issued': False}
