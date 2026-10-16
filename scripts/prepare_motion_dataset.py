"""Freeze motion scene geometry, participants and splits before any capture results."""
import argparse
import hashlib
import json
from pathlib import Path

from collect_dataset import ROOT, pose, sha, transform
from senior_ai.core import write_json
from senior_ai.dataset import check_group_splits


def forward_path(waypoint, steps, spacing):
    path = [waypoint]
    for _ in range(steps):
        options = path[-1].next(spacing)
        if not options:
            raise ValueError('Frozen path runs off map')
        yaw = path[-1].transform.rotation.yaw
        path.append(min(options, key=lambda w: (abs((w.transform.rotation.yaw - yaw + 180) % 360 - 180), w.road_id, w.lane_id)))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--criteria', type=Path, default=ROOT / 'configs/dataset_motion_criteria_v2.json')
    parser.add_argument('--source-scenes', type=Path, default=ROOT / 'configs/dataset_scenes_v1.json')
    parser.add_argument('--rig', type=Path, default=ROOT / 'configs/camera_rig_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/dataset_motion_scenes_v2.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Frozen plan exists; choose a new version')
    import carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10)
    world, criteria = client.get_world(), json.loads(args.criteria.read_text())
    road_map, library = world.get_map(), world.get_blueprint_library()
    if road_map.name.split('/')[-1] != criteria['map']:
        raise ValueError('Reuse loaded Town01')
    source = json.loads(args.source_scenes.read_text())
    if hashlib.sha256(road_map.to_opendrive().encode()).hexdigest() != source['opendrive_sha256']:
        raise ValueError('Source map geometry changed')
    walkers = sorted(library.filter('walker.pedestrian.*'), key=lambda bp: bp.id)
    driving_points = road_map.generate_waypoints(2.0)
    scenes = []
    for i, original in enumerate(source['scenes']):
        scene = {k: original[k] for k in ['id', 'group_id', 'split', 'spawn_index', 'road_id', 'start', 'weather', 'junction_distance_m', 'seed']}
        start = road_map.get_waypoint(transform(scene['start']).location)
        path = forward_path(start, 30, 2.0)
        scene['ego_route_m'] = [[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in path]
        scene['goal_m'] = scene['ego_route_m'][20]
        scene['route_road_ids'] = sorted({w.road_id for w in path[:21]})
        participants = []
        for blueprint, kind, distance, opposite in [('vehicle.bh.crossbike', 'bicycle', 8, False),
                                                    ('vehicle.kawasaki.ninja', 'motorcycle', 12, True),
                                                    ('vehicle.lincoln.mkz_2020', 'car', 22, True)]:
            library.find(blueprint)
            wp = path[distance // 2]
            if opposite:
                reference = wp.transform
                candidates = [p for p in driving_points if p.lane_type == carla.LaneType.Driving
                              and 2.8 <= p.transform.location.distance(reference.location) <= 8
                              and abs((p.transform.rotation.yaw - reference.rotation.yaw + 180) % 360 - 180) > 120]
                if not candidates:
                    raise ValueError(f'No nearby opposing driving lane: {scene["id"]}, {kind}')
                wp = min(candidates, key=lambda p: (p.transform.location.distance(reference.location), p.road_id, p.s))
            spawn = wp.transform
            spawn.location.z += 0.3
            participants.append({'blueprint': blueprint, 'kind': kind, 'start': pose(spawn),
                                 'controller': 'traffic_manager', 'speed_difference_percent': 70})
        # Two sidewalk paths near the ego; no random navmesh relocation or crossing.
        anchor = path[3].transform
        right = anchor.get_right_vector()
        for side in [-1, 1]:
            query = carla.Location(anchor.location.x + side * 6 * right.x,
                                   anchor.location.y + side * 6 * right.y, anchor.location.z)
            wp = road_map.get_waypoint(query, lane_type=carla.LaneType.Sidewalk)
            if wp is None or wp.transform.location.distance(query) > 5:
                raise ValueError('No nearby sidewalk for frozen pedestrian')
            pedestrian_path = forward_path(wp, 12, 1.0)
            spawn = wp.transform
            spawn.location.z += 1.0
            participants.append({'blueprint': walkers[(i * 2 + (side == 1)) % len(walkers)].id,
                                 'kind': 'pedestrian', 'start': pose(spawn), 'controller': 'walker_path', 'speed_mps': 0.8,
                                 'path_m': [[p.transform.location.x, p.transform.location.y, p.transform.location.z] for p in pedestrian_path]})
        scene['participants'] = participants
        scene['requested_npcs'] = len(participants)
        scenes.append(scene)
    frozen = {'version': 'town01-motion-scenes-v2', 'criteria_sha256': sha(args.criteria), 'rig_sha256': sha(args.rig),
              'opendrive_sha256': source['opendrive_sha256'], 'source_scenes_sha256': sha(args.source_scenes),
              'selection': 'Retain the pre-existing six geometry-selected spatial groups/splits. Freeze route geometry and participant types/poses before capture; no success-based reselection. Shared Town01 is not an unseen-town test.',
              'scenes': scenes}
    check_group_splits(frozen)
    # Also prevent planned ego route road IDs crossing the split boundary.
    road_sets = {split: {road for s in scenes if s['split'] == split for road in s['route_road_ids']}
                 for split in ['train', 'validation', 'test']}
    if any(road_sets[a] & road_sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Planned ego route roads cross split boundaries')
    write_json(args.output, frozen)
    print(json.dumps({'status': 'frozen', 'plan_sha256': sha(args.output), 'route_roads_by_split': {k: sorted(v) for k, v in road_sets.items()},
                      'scenes': [{'id': s['id'], 'participants': [p['kind'] for p in s['participants']]} for s in scenes]}))


if __name__ == '__main__':
    main()
