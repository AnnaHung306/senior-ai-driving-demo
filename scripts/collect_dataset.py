"""Bounded eight-position RGB/semantic/depth pilot; simulator GT is labels only."""
import argparse
from collections import Counter
import faulthandler
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import intrinsics, receive_frame, write_json
from senior_ai.dataset import check_group_splits, decode_depth_bgra, semantic_ids_bgra
from senior_ai.frames import RgbFrame
from rig_capture import validate_rig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyz(v):
    return [v.x, v.y, v.z]


def pose(t):
    return {'position_m': xyz(t.location), 'rotation_deg': [t.rotation.pitch, t.rotation.yaw, t.rotation.roll]}


def transform(p):
    import carla
    return carla.Transform(carla.Location(*p['position_m']), carla.Rotation(*p['rotation_deg']))


def prepare(args, world, criteria):
    import carla
    if args.plan.exists():
        raise FileExistsError('Plan is frozen; use a new version to change it')
    road_map = world.get_map()
    spawns = road_map.get_spawn_points()
    junctions = [w.transform.location for w in road_map.generate_waypoints(10) if w.is_junction]
    candidates = []
    for index, t in enumerate(spawns):
        wp = road_map.get_waypoint(t.location, lane_type=carla.LaneType.Driving)
        if not wp.is_junction:
            candidates.append((index, t, wp, min(t.location.distance(j) for j in junctions)))
    # Geometry-only first junction approach, then farthest-point sampling.
    chosen = [next(c for c in candidates if c[3] <= 25)]
    while len(chosen) < criteria['scene_count']:
        roads = {c[2].road_id for c in chosen}
        remaining = [c for c in candidates if c[2].road_id not in roads]
        nxt = max(remaining, key=lambda c: min(c[1].location.distance(p[1].location) for p in chosen))
        if min(nxt[1].location.distance(c[1].location) for c in chosen) < criteria['min_scene_origin_separation_m']:
            raise ValueError('Cannot satisfy frozen spatial separation')
        chosen.append(nxt)
    settings = [('train', 'ClearNoon', 0), ('train', 'ClearNight', 3), ('train', 'HardRainNoon', 6),
                ('validation', 'WetCloudyNoon', 3), ('test', 'ClearNoon', 6), ('test', 'HardRainNight', 3)]
    scenes = []
    for i, ((index, t, wp, junction_distance), (split, weather, density)) in enumerate(zip(chosen, settings)):
        nearby = sorted(((k, s) for k, s in enumerate(spawns) if 8 < t.location.distance(s.location) <= 100),
                        key=lambda v: t.location.distance(v[1].location))
        if len(nearby) < density:
            raise ValueError('Insufficient NPC slots; do not silently lower requested density')
        scenes.append({'id': f'scene-{i:02d}', 'group_id': f'town01-road-{wp.road_id}', 'split': split,
                       'spawn_index': index, 'road_id': wp.road_id, 'start': pose(t), 'weather': weather,
                       'junction_distance_m': junction_distance, 'requested_npcs': density,
                       'npc_spawns': [pose(s) for _, s in nearby[:density]], 'seed': criteria['seed'] + i})
    plan = {'version': 'town01-pilot-scenes-v1', 'criteria_sha256': sha(args.criteria),
            'rig_sha256': sha(args.rig), 'opendrive_sha256': hashlib.sha256(road_map.to_opendrive().encode()).hexdigest(),
            'selection': 'Map geometry only; first junction approach then farthest distinct roads, >=70m origins; whole scene groups assigned before capture. Town01 is shared across splits, so this is not an unseen-town test.',
            'scenes': scenes}
    check_group_splits(plan)
    write_json(args.plan, plan)
    print(json.dumps({'status': 'frozen', 'plan_sha256': sha(args.plan), 'scenes': scenes}), flush=True)
    return 0


def collect_scene(world, client, scene, rig, criteria, root, count):
    import carla
    folder = root / 'scenes' / scene['id']
    folder.mkdir(parents=True)
    result = {'scene_id': scene['id'], 'split': scene['split'], 'status': 'failed', 'errors': [],
              'bundles': 0, 'images': 0, 'max_timestamp_error_s': 0.0, 'requested_npcs': scene['requested_npcs'],
              'spawned_npcs': 0, 'semantic_ids': [], 'depth_min_m': None, 'depth_max_m': None}
    actors, sensors, queues, indexes, label_actors = [], [], {}, [], []
    autopilot_actors, pedestrian_controls, actor_kinds, collision_events = [], [], {}, []
    moving = 'ego_route_m' in scene
    stride = criteria.get('capture_stride_ticks', 1)
    simulation_ticks = count * stride
    motion_labels, pixel_counts, positive_images = [], Counter(), Counter()
    original, original_weather = world.get_settings(), world.get_weather()
    manager = None
    cleanup = True
    started = time.monotonic()
    classes, depth_min, depth_max = set(), math.inf, -math.inf
    try:
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, criteria['fixed_delta_s'], False
        world.apply_settings(settings)
        world.set_weather(getattr(carla.WeatherParameters, scene['weather']))
        for _ in range(10):
            world.tick(15)
        library = world.get_blueprint_library()
        vehicle = world.spawn_actor(library.find('vehicle.tesla.model3'), transform(scene['start']))
        actors.append(vehicle)
        manager = client.get_trafficmanager(8000)
        manager.set_synchronous_mode(True)
        manager.set_random_device_seed(scene['seed'])
        definitions = scene.get('participants') or [
            {'blueprint': 'vehicle.lincoln.mkz_2020', 'kind': 'car', 'start': p, 'controller': 'traffic_manager'}
            for p in scene.get('npc_spawns', [])]
        for definition in definitions:
            bp = library.find(definition['blueprint'])
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            npc = world.try_spawn_actor(bp, transform(definition['start']))
            if npc is None:
                raise RuntimeError('Frozen NPC spawn failed; no density substitution')
            actors.append(npc)
            label_actors.append(npc)
            actor_kinds[npc.id] = definition['kind']
            if definition['controller'] == 'traffic_manager':
                npc.set_autopilot(True, 8000)
                autopilot_actors.append(npc)
                if 'speed_difference_percent' in definition:
                    manager.vehicle_percentage_speed_difference(npc, definition['speed_difference_percent'])
            else:
                pedestrian_controls.append([npc, definition, 0])
        result['spawned_npcs'] = len(label_actors)
        result['spawned_kinds'] = dict(Counter(actor_kinds.values()))
        agent = None
        if moving:
            sys.path.insert(0, str(ROOT / 'artifacts/carla/PythonAPI/carla'))
            from agents.navigation.basic_agent import BasicAgent
            agent = BasicAgent(vehicle, target_speed=criteria['target_speed_kph'],
                               opt_dict={'dt': criteria['fixed_delta_s'], 'base_min_distance': 1.0,
                                         'distance_ratio': 0.2, 'max_brake': 1.0, 'sampling_resolution': 2.0})
            agent.set_destination(carla.Location(*scene['goal_m']), start_location=vehicle.get_location())
            collision = world.spawn_actor(library.find('sensor.other.collision'), carla.Transform(), attach_to=vehicle)
            actors.append(collision)
            sensors.append(collision)
            collision.listen(lambda event: collision_events.append({'frame': event.frame, 'other_actor': event.other_actor.id}))
        calibration = []
        for definition in rig['cameras']:
            name = definition['name']
            t = carla.Transform(carla.Location(*definition['position_m']), carla.Rotation(yaw=definition['yaw_deg']))
            calibration.append({'name': name, 'K': intrinsics(rig['width'], rig['height'], rig['horizontal_fov_deg']),
                                'T_ego_from_camera_local': t.get_matrix()})
            for modality in criteria['modalities']:
                bp = library.find(f'sensor.camera.{modality}')
                for key, value in {'image_size_x': rig['width'], 'image_size_y': rig['height'],
                                   'fov': rig['horizontal_fov_deg'], 'sensor_tick': rig['sensor_tick_s']}.items():
                    bp.set_attribute(key, str(value))
                sensor = world.spawn_actor(bp, t, attach_to=vehicle)
                actors.append(sensor)
                sensors.append(sensor)
                sink = queue.Queue()
                queues[(name, modality)] = sink
                sensor.listen(lambda image, target=sink: target.put(RgbFrame.from_carla(image)))
        cal_relative = f'calibration/{scene["id"]}.json'
        write_json(root / cal_relative, {'schema_version': 1, 'coordinate_system': rig['coordinate_system'],
                   'optical_from_carla_camera': [[0, 1, 0], [0, 0, -1], [1, 0, 0]], 'cameras': calibration})
        result['stage'] = 'warmup'
        write_json(folder / 'report.json', result)
        for _ in range(criteria.get('sensor_warmup_ticks', 5)):
            vehicle.apply_control(carla.VehicleControl(brake=1))
            world.tick(15)
        # Static road labels are kept exclusively under labels/.
        local_roads = []
        origin = transform(scene['start']).location
        for wp in world.get_map().generate_waypoints(5):
            if wp.transform.location.distance(origin) <= 90:
                local_roads.append({'road_id': wp.road_id, 'lane_id': wp.lane_id, 's_m': wp.s,
                                    'width_m': wp.lane_width, 'is_junction': wp.is_junction, 'pose': pose(wp.transform),
                                    'left_marking': str(wp.left_lane_marking.type), 'right_marking': str(wp.right_lane_marking.type)})
        write_json(root / f'labels/roads/{scene["id"]}.json', {'source': 'simulator_ground_truth', 'waypoints': local_roads})
        pending = None
        result['stage'] = 'capture'
        write_json(folder / 'report.json', result)
        with ThreadPoolExecutor(max_workers=6) as encoder:
            def consume(item, observed):
                nonlocal depth_min, depth_max
                feature, labels = item
                frame, stamp = feature['frame'], feature['timestamp_s']
                if observed - frame != criteria['readback_lag_ticks']:
                    raise ValueError('Unexpected frame readback distance')
                jobs = []
                for (name, modality), source in queues.items():
                    image = receive_frame(source, frame, 15)
                    error = abs(image.timestamp - stamp)
                    if error > criteria['max_timestamp_error_s'] or not math.isfinite(error):
                        raise ValueError('Sensor timestamp mismatch')
                    if (image.width, image.height) != (rig['width'], rig['height']):
                        raise ValueError('Sensor dimensions mismatch')
                    result['max_timestamp_error_s'] = max(result['max_timestamp_error_s'], error)
                    base = 'inputs/rgb' if modality == 'rgb' else f'labels/{modality}'
                    relative = f'{base}/{scene["id"]}/{name}/{frame:08d}.png'
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append(encoder.submit(image.save_to_disk, path))
                    entry = {'image': relative, 'frame': frame, 'timestamp_s': image.timestamp}
                    if modality == 'rgb':
                        feature['cameras'][name] = entry
                    else:
                        labels[modality][name] = entry
                    if modality == 'semantic_segmentation':
                        tags = semantic_ids_bgra(image.raw_data, image.width, image.height)
                        hist = np.bincount(tags.reshape(-1), minlength=29)
                        present = np.flatnonzero(hist)
                        classes.update(int(v) for v in present)
                        pixel_counts.update({int(k): int(hist[k]) for k in present})
                        positive_images.update(int(k) for k in present)
                        if tags.max() > 28:
                            raise ValueError('Unknown semantic class ID')
                    if modality == 'depth':
                        depth = decode_depth_bgra(image.raw_data, image.width, image.height)
                        depth_min, depth_max = min(depth_min, float(depth.min())), max(depth_max, float(depth.max()))
                for job in jobs:
                    job.result()
                input_relative = f'inputs/frames/{scene["id"]}/{frame:08d}.json'
                label_relative = f'labels/frames/{scene["id"]}/{frame:08d}.json'
                write_json(root / input_relative, feature)
                write_json(root / label_relative, labels)
                indexes.append({'scene_id': scene['id'], 'group_id': scene['group_id'], 'split': scene['split'],
                                'frame': frame, 'input': input_relative, 'label': label_relative})
                if moving:
                    motion_labels.append(labels)
                result['bundles'] += 1
                result['images'] += len(jobs)
            for step in range(simulation_ticks):
                if time.monotonic() - started > criteria['max_scene_wall_s']:
                    raise TimeoutError('Scene wall-clock deadline')
                control = agent.run_step() if agent is not None and not agent.done() else carla.VehicleControl(brake=1)
                vehicle.apply_control(control)
                for entry in pedestrian_controls:
                    pedestrian, definition, index = entry
                    location = pedestrian.get_location()
                    path = definition['path_m']
                    while index < len(path) - 1 and math.hypot(path[index][0] - location.x, path[index][1] - location.y) < 0.4:
                        index += 1
                    entry[2] = index
                    dx, dy = path[index][0] - location.x, path[index][1] - location.y
                    length = math.hypot(dx, dy)
                    direction = carla.Vector3D(dx / length, dy / length, 0) if length > 0.1 else carla.Vector3D()
                    pedestrian.apply_control(carla.WalkerControl(direction, definition['speed_mps'] if length > 0.1 else 0, False))
                frame = world.tick(15)
                snapshot = world.get_snapshot()
                if snapshot.frame != frame:
                    raise ValueError('Multiple tick clients detected')
                # Drain selected F after exactly one world tick, even when saving only every N ticks.
                if pending is not None:
                    consume(pending, frame)
                    pending = None
                if step % stride:
                    continue
                # Copy pending GT now, before advancing; live actor state later is forbidden.
                ego = snapshot.find(vehicle.id)
                if ego is None:
                    raise ValueError('Missing ego snapshot')
                stamp = snapshot.timestamp.elapsed_seconds
                feature = {'schema_version': 1, 'source': 'carla_rgb_only', 'scene_id': scene['id'],
                           'frame': frame, 'timestamp_s': stamp, 'cameras': {}, 'calibration': cal_relative}
                labels = {'schema_version': 1, 'source': 'simulator_ground_truth', 'frame': frame, 'timestamp_s': stamp,
                          'ego_world_pose': pose(ego.get_transform()), 'ego_velocity_mps': xyz(ego.get_velocity()),
                          'objects_3d': [], 'semantic_segmentation': {}, 'depth': {},
                          'roads': f'labels/roads/{scene["id"]}.json'}
                if moving:
                    labels['episode_tick'] = step
                    labels['ego_control'] = {'throttle': control.throttle, 'brake': control.brake, 'steer': control.steer,
                                             'source': 'carla_basic_agent_rules_with_simulator_gt'}
                for actor in label_actors:
                    state = snapshot.find(actor.id)
                    if state is None:
                        raise ValueError('Missing actor snapshot')
                    bbox, t = actor.bounding_box, state.get_transform()
                    if step == 0:
                        result['stage'] = 'object_labels'
                        write_json(folder / 'report.json', result)
                    labels['objects_3d'].append({'actor_id': actor.id, 'blueprint': actor.type_id, 'kind': actor_kinds[actor.id],
                         'semantic_tags': list(actor.semantic_tags), 'world_pose': pose(t),
                         'bbox_local_pose': pose(carla.Transform(bbox.location, bbox.rotation)),
                         'extent_m': xyz(bbox.extent), 'world_vertices_m': [xyz(v) for v in bbox.get_world_vertices(t)],
                         'velocity_mps': xyz(state.get_velocity()), 'visibility': 'not_filtered; may be occluded/outside view'})
                pending = (feature, labels)
            if pending is not None:
                consume(pending, world.tick(15))
        result['checks'] = {'complete_bundles': result['bundles'] == count,
                            'complete_images': result['images'] == count * 24,
                            'traffic_density': result['spawned_npcs'] == scene['requested_npcs'],
                            'semantic_signal': 1 in classes and len(classes) >= 3,
                            'valid_depth_range': 0 <= depth_min < depth_max <= 1000}
        if moving:
            first, last = motion_labels[0], motion_labels[-1]
            displacement = math.dist(first['ego_world_pose']['position_m'], last['ego_world_pose']['position_m'])
            origins = {o['actor_id']: o['world_pose']['position_m'] for o in first['objects_3d']}
            deltas = {str(o['actor_id']): math.dist(origins[o['actor_id']], o['world_pose']['position_m']) for o in last['objects_3d']}
            result['motion'] = {'ego_displacement_m': displacement, 'label_duration_s': last['timestamp_s'] - first['timestamp_s'],
                                'participant_displacements_m': deltas, 'simulation_ticks': simulation_ticks,
                                'capture_stride_ticks': stride, 'ego_collisions': len(collision_events)}
            result['checks']['ego_collision_free'] = len(collision_events) <= criteria['max_ego_collisions']
            if count == criteria['bundles_per_scene']:
                result['checks'].update({'ego_motion': displacement >= criteria['min_ego_displacement_m'],
                    'label_duration': result['motion']['label_duration_s'] >= criteria['min_label_duration_s'],
                    'participant_motion': all(v >= criteria['min_participant_displacement_m'] for v in deltas.values())})
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        result['stage'] = 'cleanup'
        write_json(folder / 'report.json', result)
        # Unregister live vehicles before destroying them. Keep TM synchronous
        # through world restoration, as in the packaged traffic example.
        client.set_timeout(15 if not result['errors'] else 2)
        for npc in autopilot_actors:
            try:
                npc.set_autopilot(False, 8000)
            except Exception as exc:
                cleanup = False
                result['errors'].append(f'autopilot unregister: {exc}')
        for sensor in reversed(sensors):
            try:
                sensor.stop()
            except Exception as exc:
                cleanup = False
                result['errors'].append(f'stop: {exc}')
        for actor in reversed(actors):
            try:
                if not actor.destroy():
                    raise RuntimeError('destroy returned false')
            except Exception as exc:
                cleanup = False
                result['errors'].append(f'destroy: {exc}')
        try:
            world.set_weather(original_weather)
            world.apply_settings(original)
            if manager is not None:
                manager.set_synchronous_mode(False)
        except Exception as exc:
            cleanup = False
            result['errors'].append(f'restore: {exc}')
        client.set_timeout(15)
        result.setdefault('checks', {})['cleanup'] = cleanup
        result.update({'wall_s': time.monotonic() - started, 'semantic_ids': sorted(classes),
                       'semantic_pixel_counts': dict(sorted(pixel_counts.items())),
                       'semantic_positive_images': dict(sorted(positive_images.items())),
                       'depth_min_m': depth_min if math.isfinite(depth_min) else None,
                       'depth_max_m': depth_max if math.isfinite(depth_max) else None})
        write_json(folder / 'collisions.json', collision_events)
        if all(result['checks'].values()) and not result['errors'] and result['bundles'] == count:
            result['status'] = 'passed'
        write_json(folder / 'report.json', result)
    return result, indexes


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'probe', 'collect'])
    parser.add_argument('--criteria', type=Path, default=ROOT / 'configs/dataset_pilot_criteria_v1.json')
    parser.add_argument('--rig', type=Path, default=ROOT / 'configs/camera_rig_v1.json')
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/dataset_scenes_v1.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe-scene', type=int, default=0)
    args = parser.parse_args()
    if args.mode != 'prepare' and args.output is None:
        parser.error('--output required')
    criteria, rig = json.loads(args.criteria.read_text()), json.loads(args.rig.read_text())
    validate_rig(rig)
    import carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(15)
    if client.get_server_version() != '0.9.16' or client.get_client_version() != '0.9.16':
        raise ValueError('Matching CARLA 0.9.16 required')
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != criteria['map']:
        raise ValueError('Reuse declared loaded map, no unnecessary reload')
    if args.mode == 'prepare':
        return prepare(args, world, criteria)
    plan = json.loads(args.plan.read_text())
    check_group_splits(plan)
    if plan['criteria_sha256'] != sha(args.criteria) or plan['rig_sha256'] != sha(args.rig):
        raise ValueError('Frozen criteria/rig changed')
    if hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest() != plan['opendrive_sha256']:
        raise ValueError('Frozen map geometry changed')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'plan.json', plan)
    write_json(args.output / 'criteria.json', criteria)
    write_json(args.output / 'rig.json', rig)
    write_json(args.output / 'encoding.json', {'depth': 'Packed RGB PNG, raw CARLA 24-bit channels. meters=(R+256*G+65536*B)*1000/16777215; not grayscale.',
                 'semantic': 'Raw ID in PNG red channel (0..28); not CityScapes visualization colors.',
                 'source': 'https://carla.readthedocs.io/en/0.9.16/ref_sensors/',
                 'labels': 'Simulator GT only. Boxes include all spawned participants regardless of visibility. Participant kinds are recorded in scene plan; no per-object visibility filtering.'})
    report = {'task': f'dataset-{args.mode}', 'status': 'failed', 'scope': criteria['scope'], 'scenes': [], 'ai_model': None,
              'plan_sha256': sha(args.plan), 'criteria_sha256': sha(args.criteria), 'rig_sha256': sha(args.rig),
              'git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
              'server_launch': json.loads((ROOT / 'artifacts/reports/server-launch.json').read_text(encoding='utf-8-sig'))}
    report['controller_source'] = 'carla_basic_agent_rules_with_simulator_gt' if 'ego_route_m' in plan['scenes'][0] else 'stationary_brake'
    if report['controller_source'] != 'stationary_brake':
        report['agent_sha256'] = sha(ROOT / 'artifacts/carla/PythonAPI/carla/agents/navigation/basic_agent.py')
    all_indexes = []
    scenes = [plan['scenes'][args.probe_scene]] if args.mode == 'probe' else plan['scenes']
    count = 5 if args.mode == 'probe' else criteria['bundles_per_scene']
    for scene in scenes:
        result, indexes = collect_scene(world, client, scene, rig, criteria, args.output, count)
        report['scenes'].append(result)
        all_indexes.extend(indexes)
        write_json(args.output / 'report.json', report)
        print(json.dumps(result), flush=True)
        if result['status'] != 'passed':
            break
    for split in ['train', 'validation', 'test']:
        path = args.output / f'indexes/{split}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(item) + '\n' for item in all_indexes if item['split'] == split))
    report['total_bundles'], report['total_images'] = len(all_indexes), sum(s['images'] for s in report['scenes'])
    if len(report['scenes']) == len(scenes) and all(s['status'] == 'passed' for s in report['scenes']):
        report['status'] = 'passed'
    write_json(args.output / 'report.json', report)
    print(json.dumps({'task': report['task'], 'status': report['status'], 'total_bundles': report['total_bundles'], 'total_images': report['total_images']}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
