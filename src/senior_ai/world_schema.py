"""WorldFrame v1 offline/API validation for Phase6 consumers."""
from pathlib import PurePosixPath
import re
import numpy as np
from .cuboid import corners
from .semantic_occupancy import VoxelGrid


def validate_world_frame(frame, occupancy=None):
    if frame['schema_version'] != 1 or frame['coordinate_frame'] != 'current_ego' or frame['axes'] != 'x_forward_y_right_z_up':
        raise ValueError('Unsupported WorldFrame coordinate schema')
    if frame['units'] != {'position': 'meters', 'size': 'full_LWH_meters', 'angle': 'radians'}:
        raise ValueError('WorldFrame unit mismatch')
    if frame['GT_used_in_inference'] is not False or frame['control_issued'] is not False:
        raise ValueError('RGB-only non-actuating WorldFrame required')
    if not isinstance(frame['frame'], int) or frame['frame'] < 0 or not np.isfinite(frame['timestamp_s']):
        raise ValueError('Invalid frame/time')
    identities = set()
    for obj in frame['objects']:
        corners(obj)
        if obj['class_id'] not in range(1, 7) or not 0 <= obj['score_uncalibrated'] <= 1:
            raise ValueError('Invalid actor class/score')
        identity = obj['track_id']
        if identity in identities or not identity.startswith(frame['scene_id']+':T') or obj['measurement_current'] is not True:
            raise ValueError('Unique scene-local current measured IDs required')
        identities.add(identity)
        if not obj['sources'] or len({s['camera'] for s in obj['sources']}) != len(obj['sources']):
            raise ValueError('Unique per-camera sources required')
        if any(s['class_id'] != obj['class_id'] for s in obj['sources']):
            raise ValueError('Cross-class fused cuboid')
        if any(any(k in s for k in ['actor_id', 'world_pose', 'world_vertices_m', 'depth']) for s in [obj, *obj['sources']]):
            raise ValueError('GT identity/pose/depth fields forbidden in inferred objects')
    name = frame['occupancy']['file'];path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Occupancy file must be a safe relative path')
    if not re.fullmatch('[0-9a-f]{64}', frame['occupancy']['sha256']):
        raise ValueError('Occupancy SHA256 required')
    if frame['tracking']['held_tracks_used_for_occupancy'] is not False:
        raise ValueError('Held tracks cannot be current occupancy evidence')
    grid = VoxelGrid(**frame['occupancy_grid'])
    if occupancy is not None:
        if occupancy.shape != grid.shape or occupancy.dtype != np.uint8 or not np.isin(occupancy, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 255]).all():
            raise ValueError('Occupancy shape/type/labels differ')
        lower = grid.z_min+np.arange(grid.shape[2])*grid.resolution_m
        if np.any(occupancy[..., lower <= 0] == 0):
            raise ValueError('Ground/below-ground slab cannot be ray-carved free')
    return True
