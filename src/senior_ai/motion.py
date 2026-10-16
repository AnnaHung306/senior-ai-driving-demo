"""Audit measurable motion and timing, never classify scripted motion as AI."""
import math


def audit_episode(labels, criteria):
    if len(labels) != criteria['bundles_per_scene']:
        raise ValueError('Incomplete motion episode')
    stride, delta = criteria['capture_stride_ticks'], criteria['fixed_delta_s']
    first = labels[0]
    identities = {o['actor_id'] for o in first['objects_3d']}
    for index, label in enumerate(labels):
        if label['episode_tick'] != index * stride or label['frame'] != first['frame'] + index * stride:
            raise ValueError('Episode frame/tick cadence mismatch')
        error = abs(label['timestamp_s'] - first['timestamp_s'] - index * stride * delta)
        if not math.isfinite(error) or error > criteria['max_timestamp_error_s']:
            raise ValueError('Episode timestamp cadence mismatch')
        if {o['actor_id'] for o in label['objects_3d']} != identities:
            raise ValueError('Participant identity changed within episode')
        points = [label['ego_world_pose']['position_m']] + [o['world_pose']['position_m'] for o in label['objects_3d']]
        if any(len(point) != 3 or not all(math.isfinite(x) for x in point) for point in points):
            raise ValueError('Invalid trajectory position')
    last = labels[-1]
    ego = math.dist(first['ego_world_pose']['position_m'], last['ego_world_pose']['position_m'])
    origins = {o['actor_id']: o['world_pose']['position_m'] for o in first['objects_3d']}
    displacement = {str(o['actor_id']): math.dist(origins[o['actor_id']], o['world_pose']['position_m']) for o in last['objects_3d']}
    duration = last['timestamp_s'] - first['timestamp_s']
    if ego < criteria['min_ego_displacement_m'] or duration < criteria['min_label_duration_s']:
        raise ValueError('Insufficient ego motion or sequence duration')
    if not displacement or any(d < criteria['min_participant_displacement_m'] for d in displacement.values()):
        raise ValueError('A participant did not move enough')
    return {'ego_displacement_m': ego, 'label_duration_s': duration,
            'participant_displacements_m': displacement, 'frame_stride': stride}


def require_class_coverage(pixels, positive_images, criteria):
    for split in ['train', 'validation', 'test']:
        for class_id in criteria['required_semantic_ids_per_split']:
            if pixels[split].get(class_id, 0) < criteria['min_semantic_pixels_per_required_class_per_split']:
                raise ValueError(f'Insufficient semantic pixels: split={split}, class={class_id}')
            if positive_images[split].get(class_id, 0) < criteria['min_positive_images_per_required_class_per_split']:
                raise ValueError(f'Insufficient positive images: split={split}, class={class_id}')
