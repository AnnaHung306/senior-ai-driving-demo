"""Small causal class-gated measured-object tracker. No GT identities/velocities."""
import copy
import numpy as np
from .cuboid import wrap


class WorldTracker:
    def __init__(self, config):
        self.config = config;self.tracks = {};self.previous = None;self.counter = 0

    def update(self, objects, scene, timestamp, calibration_sha, motion):
        if not np.isfinite(timestamp):
            raise ValueError('Finite timestamp required')
        reset = 'cold_start'
        if self.previous is not None:
            old_scene, old_time, old_calibration = self.previous
            dt = timestamp-old_time
            if scene != old_scene or calibration_sha != old_calibration:
                reset = 'stream_identity_changed'
            elif dt <= 0 or dt > self.config['maximum_gap_s']:
                reset = 'timestamp_gap_or_reversal'
            elif motion.get('status') != 'aligned':
                reset = 'untrusted_ego_motion'
            else:
                t = np.asarray(motion['previous_to_current'], dtype=float)
                if t.shape != (3, 3) or not np.isfinite(t).all() or not np.allclose(t[2], [0, 0, 1]) or not np.allclose(t[:2, :2].T@t[:2, :2], np.eye(2), atol=1e-5) or not np.isclose(np.linalg.det(t[:2, :2]), 1):
                    raise ValueError('Rigid previous-to-current SE2 required')
                yaw = np.arctan2(t[1, 0], t[0, 0])
                for track in self.tracks.values():
                    p = track['box']['center_ego_m']
                    track['box']['center_ego_m'][:2] = (t[:2, :2]@p[:2]+t[:2, 2]).tolist()
                    track['box']['yaw_rad'] = float(wrap(track['box']['yaw_rad']+yaw))
                reset = None
        if reset:
            self.tracks.clear()
        pairs = sorted((np.linalg.norm(np.array(obj['center_ego_m'])[:2]-np.array(track['box']['center_ego_m'])[:2]), key, i)
                       for key, track in self.tracks.items() for i, obj in enumerate(objects) if obj['class_id'] == track['box']['class_id'])
        used = set();matched = {}
        for distance, key, i in pairs:
            if distance <= self.config['association_distance_m'] and key not in used and i not in matched:
                matched[i] = key;used.add(key)
        for key in list(self.tracks):
            if key not in used:
                self.tracks[key]['missed'] += 1
                if self.tracks[key]['missed'] > self.config['maximum_missed_frames']:
                    del self.tracks[key]
        measured = []
        for i, obj in enumerate(objects):
            key = matched.get(i)
            if key is None:
                self.counter += 1;key = f'{scene}:T{self.counter:06d}';age = 1
            else:
                age = self.tracks[key]['age']+1
            box = {**copy.deepcopy(obj), 'track_id': key, 'track_age_measurements': age, 'measurement_current': True}
            self.tracks[key] = {'box': copy.deepcopy(box), 'age': age, 'missed': 0}
            measured.append(box)
        held = [{'track_id': key, 'missed_frames': track['missed'], 'measurement_current': False}
                for key, track in self.tracks.items() if track['missed']]
        self.previous = scene, timestamp, calibration_sha
        return measured, {'reset_reason': reset, 'held_unobserved_tracks': held,
                          'association': 'causal_class_gated_greedy_nearest_one_to_one', 'GT_identity_used': False,
                          'held_tracks_used_for_occupancy': False, 'velocity_predicted': False}
