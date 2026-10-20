"""Tri-state semantic grid from visible ground rays and predicted cuboids.

This is a coarse geometric occupancy baseline, not a learned surface network.
255=unknown, 0=ray-supported free, 1=road, 2=sidewalk, 3..8=actor classes,
9=terrain. Predictions never consume GT depth, actors or world poses.
"""
from dataclasses import dataclass
import numpy as np
from .bev_geometry import Grid, projection_maps, sample
from .cuboid import corners

LABELS = {0: 'free', 1: 'road', 2: 'sidewalk', 3: 'car', 4: 'pedestrian', 5: 'bicycle', 6: 'motorcycle', 7: 'truck', 8: 'bus', 9: 'terrain', 255: 'unknown'}


@dataclass(frozen=True)
class VoxelGrid:
    x_min: float = -15.
    x_max: float = 35.
    y_min: float = -20.
    y_max: float = 20.
    z_min: float = -.5
    z_max: float = 4.5
    resolution_m: float = .5

    def __post_init__(self):
        Grid(self.x_min, self.x_max, self.y_min, self.y_max, self.resolution_m)
        if not np.isfinite([self.z_min, self.z_max]).all() or self.z_max <= self.z_min or not np.isclose((self.z_max-self.z_min)/self.resolution_m, round((self.z_max-self.z_min)/self.resolution_m)):
            raise ValueError('Valid whole vertical cells required')

    @property
    def ground(self):
        return Grid(self.x_min, self.x_max, self.y_min, self.y_max, self.resolution_m)

    @property
    def shape(self):
        return (*self.ground.shape, round((self.z_max-self.z_min)/self.resolution_m))

    def index(self, xyz):
        p = np.asarray(xyz)
        indices = np.stack([self.shape[0]-1-np.floor((p[..., 0]-self.x_min)/self.resolution_m),
                            np.floor((p[..., 1]-self.y_min)/self.resolution_m), np.floor((p[..., 2]-self.z_min)/self.resolution_m)], -1)
        finite = np.isfinite(indices).all(-1)
        indices = np.nan_to_num(indices, nan=-1, posinf=-1, neginf=-1).astype(np.int64)
        valid = finite & (indices >= 0).all(-1) & (indices < self.shape).all(-1)
        return indices, valid

    def centers(self):
        xy = self.ground.points()[..., :2]
        z = self.z_min+(np.arange(self.shape[2])+.5)*self.resolution_m
        return np.concatenate([np.broadcast_to(xy[..., None, :], self.shape+(2,)), np.broadcast_to(z, self.shape)[..., None]], -1)


def fill_boxes(values, grid, objects):
    xyz = grid.centers()
    for obj in sorted(objects, key=lambda o: o.get('score_uncalibrated', 1.)):
        corners(obj)  # strict geometry validation
        d = xyz-np.asarray(obj['center_ego_m']);yaw = obj['yaw_rad'];c, s = np.cos(yaw), np.sin(yaw)
        local = np.stack([c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1], d[..., 2]], -1)
        inside = (np.abs(local) <= np.array(obj['size_lwh_m'])/2).all(-1)
        values[inside] = obj['class_id']+2
    return values


def build_occupancy(grid, cameras, camera_ids, objects):
    values = np.full(grid.shape, 255, np.uint8)
    xyz = grid.ground.points();votes = np.zeros(grid.ground.shape+(3,), np.int16)
    ray_count = 0
    for name, camera in cameras.items():
        maps = projection_maps(camera, grid.ground, 0., 35., {'half_length_m': 2.5, 'half_width_m': 1.1})
        ids = sample(camera_ids[name], maps, True)
        ground = maps[1] & np.isin(ids, [1, 2, 10, 16])
        votes[..., 0] += ground & np.isin(ids, [1, 10])
        votes[..., 1] += ground & (ids == 2)
        votes[..., 2] += ground & (ids == 16)
        # Coarse evenly sampled ray endpoints; rays stop before the ground endpoint.
        endpoints = xyz[::2, ::2][ground[::2, ::2]]
        if len(endpoints):
            origin = camera.T[:3, 3];lengths = np.linalg.norm(endpoints-origin, axis=1)
            steps = np.arange(0., 35., grid.resolution_m/2)
            valid_steps = steps[None, :] < (lengths[:, None]-grid.resolution_m/2)
            points = origin+(endpoints[:, None, :]-origin)*np.minimum(steps[None, :]/lengths[:, None], 1.)[..., None]
            indices, valid = grid.index(points)
            endpoint_indices, _ = grid.index(endpoints)
            valid &= np.any(indices != endpoint_indices[:, None, :], axis=-1)
            # A ray through air above a plane does not prove the entire ground
            # voxel empty. Never carve the ground-intersecting/below-ground slab.
            valid &= grid.z_min+indices[..., 2]*grid.resolution_m > 0
            selected = indices[valid & valid_steps]
            values[tuple(selected.T)] = 0;ray_count += len(endpoints)
    observed = votes.sum(-1) > 0
    labels = np.array([1, 2, 9], np.uint8)[votes.argmax(-1)]
    plane_indices, valid = grid.index(xyz)
    chosen = plane_indices[observed & valid]
    values[tuple(chosen.T)] = labels[observed & valid]
    fill_boxes(values, grid, objects)
    self_mask = (abs(xyz[..., 0]) < 2.5) & (abs(xyz[..., 1]) < 1.1)
    values[self_mask] = 255
    return values, {'source': 'predicted_ground_rays_and_learned_cuboid_volume', 'ray_endpoints': ray_count,
                    'unknown_fraction': float((values == 255).mean()), 'free_is_ray_supported_only': True,
                    'free_requires_voxel_lower_face_above_ground': True, 'ray_endpoint_voxel_never_carved': True,
                    'curb_height': 'not_estimated', 'fine_geometry_reconstructed': False,
                    'semantic_labels': LABELS, 'axes': 'row:-x,column:+y,layer:+z', 'resolution_m': grid.resolution_m}
