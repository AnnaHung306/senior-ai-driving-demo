"""Five controlled interventions, 8 cameras x RGB/semantic/depth/instance.

This generates GT engineering evidence, not learned predictions or training data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import intrinsics, receive_frame, write_json
from senior_ai.frames import RgbFrame
from senior_ai.instance_labels import validate_actor_ids
from collect_dataset import pose, transform, xyz
from rig_capture import validate_rig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Capture:
    def __init__(self, image):
        self.image = RgbFrame.from_carla(image)
        self.frame = self.image.frame
        self.world_from_camera = image.transform.get_matrix()


def collect(output, config_path, rig_path, baseline):
    config = json.loads(config_path.read_text())
    rig = json.loads(rig_path.read_text())
    validate_rig(rig)
    prior = json.loads(baseline.read_text())
    if prior['status'] != 'passed':
        raise ValueError('Requires passed Phase 0 regression before 32-sensor experiment')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'task': 'controlled-instance-engineering-probe', 'status': 'failed', 'errors': [],
        'ai_model': None, 'training_approved': False, 'bundles': 0, 'images': 0, 'max_timestamp_error_s': 0.,
        'max_pose_error': 0., 'gpu_samples_mib': [], 'cases': [], 'baseline_sha256': sha(baseline),
        'config_sha256': sha(config_path), 'rig_sha256': sha(rig_path),
        'source_sha256': sha(Path(__file__)),
        'geometry_source_sha256': sha(ROOT / 'src/senior_ai/instance_labels.py'),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    for name, value in [('config.json', config), ('rig.json', rig)]:
        write_json(output / name, value)
    write_json(output / 'report.json', report)
    actors, sensors, queues, dropped = [], [], {}, {}
    world, original, original_weather, client = None, None, None, None
    saved, snapshots = [], []
    cleanup = True
    try:
        import carla
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(config['sensor_timeout_s'])
        if client.get_server_version() != '0.9.16' or client.get_client_version() != '0.9.16':
            raise ValueError('Matching CARLA 0.9.16 required')
        world = client.get_world()
        if world.get_map().name.split('/')[-1] != config['map']:
            raise ValueError('Requires existing Town01 from baseline; no redundant map reload')
        report['server_launch'] = json.loads((ROOT / 'artifacts/reports/server-launch.json').read_text(encoding='utf-8-sig'))
        original, original_weather = world.get_settings(), world.get_weather()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, config['fixed_delta_s'], False
        world.apply_settings(settings)
        world.set_weather(getattr(carla.WeatherParameters, config.get('weather', 'ClearNoon')))
        report['weather'] = config.get('weather', 'ClearNoon')
        for _ in range(10):
            world.tick(config['sensor_timeout_s'])
        scene = next(s for s in json.loads((ROOT / 'configs/dataset_motion_scenes_v2.json').read_text())['scenes']
                     if s['id'] == config['origin_from_motion_scene'])
        origin = transform(scene['start'])
        # Flat static fixture uses known world origin; all actor physics disabled.
        # Controlled overlaps are annotation interventions, not collision-free driving.
        origin.location.z = 0
        library = world.get_blueprint_library()
        ego_start = carla.Transform(carla.Location(origin.location.x, origin.location.y, .3), origin.rotation)
        ego = world.spawn_actor(library.find('vehicle.tesla.model3'), ego_start)
        actors.append(ego)
        ego.set_simulate_physics(False)
        def at(position):
            return carla.Transform(origin.transform(carla.Location(*position)), origin.rotation)
        participants = []
        for definition in config['participants']:
            actor = world.spawn_actor(library.find(definition['blueprint']), at(definition['position_ego_m']))
            actors.append(actor)
            actor.set_simulate_physics(False)
            participants.append((actor, definition))
        validate_actor_ids([{'actor_id': a.id, 'kind': d['kind']} for a, d in participants])
        # Reject all registered aliases, not only aliases among our four targets.
        registered = [a.id for a in world.get_actors()]
        if len({i & 65535 for i in registered}) != len(registered):
            raise ValueError('Registered world actors alias in 16-bit instance IDs')
        camera_definitions = []
        for definition in rig['cameras']:
            name = definition['name']
            local = carla.Transform(carla.Location(*definition['position_m']), carla.Rotation(yaw=definition['yaw_deg']))
            camera_definitions.append({'name': name, 'K': intrinsics(rig['width'], rig['height'], rig['horizontal_fov_deg']),
                                       'T_ego_from_camera_local': local.get_matrix()})
            for modality in config['modalities']:
                bp = library.find('sensor.camera.' + modality)
                for key, value in {'image_size_x': rig['width'], 'image_size_y': rig['height'],
                                   'fov': rig['horizontal_fov_deg'], 'sensor_tick': 0}.items():
                    bp.set_attribute(key, str(value))
                sensor = world.spawn_actor(bp, local, attach_to=ego)
                actors.append(sensor); sensors.append(sensor)
                key = (name, modality)
                sink = queues[key] = queue.Queue(maxsize=4)
                dropped[key] = 0
                def callback(image, target=sink, identity=key):
                    sample = Capture(image)
                    while True:
                        try:
                            target.put_nowait(sample)
                            break
                        except queue.Full:
                            try:
                                target.get_nowait()
                                dropped[identity] += 1
                            except queue.Empty:
                                pass
                sensor.listen(callback)
        write_json(output / 'calibration/probe.json', {'schema_version': 1, 'coordinate_system': rig['coordinate_system'],
            'optical_from_carla_camera': [[0, 1, 0], [0, 0, -1], [1, 0, 0]], 'cameras': camera_definitions})
        report['sensors'] = len(sensors)
        report['stage'] = 'sensor_warmup'
        write_json(output / 'report.json', report)
        for _ in range(config['warmup_ticks']):
            world.tick(config['sensor_timeout_s'])
        occluder = next(a for a, d in participants if d['role'] == 'occluder')
        with ThreadPoolExecutor(max_workers=6) as encoder:
            for case in config['cases']:
                if time.monotonic() - started > config['max_wall_s']:
                    raise TimeoutError('Probe wall deadline exceeded')
                report['stage'] = case['name']
                write_json(output / 'report.json', report)
                occluder.set_transform(at(case['occluder_position_ego_m']))
                for _ in range(config['case_settle_ticks']):
                    world.tick(config['sensor_timeout_s'])
                frame = world.tick(config['sensor_timeout_s'])
                snapshot = world.get_snapshot()
                if frame != snapshot.frame:
                    raise ValueError('Multiple tick clients detected')
                stamp = snapshot.timestamp.elapsed_seconds
                label = {'source': 'simulator_ground_truth', 'frame': frame, 'timestamp_s': stamp,
                         'case': case['name'], 'objects': [], 'cameras': {}}
                for actor, definition in participants:
                    observed = snapshot.find(actor.id)
                    if observed is None:
                        raise ValueError('Actor not observed after warmup')
                    t, box = observed.get_transform(), actor.bounding_box
                    local_box = carla.Transform(box.location, box.rotation)
                    label['objects'].append({'actor_id': actor.id, 'blueprint': actor.type_id,
                        'kind': definition['kind'], 'role': definition['role'], 'semantic_tags': list(actor.semantic_tags),
                        'world_pose': pose(t), 'extent_m': xyz(box.extent),
                        'T_world_from_box': (np.array(t.get_matrix()) @ np.array(local_box.get_matrix())).tolist(),
                        'world_vertices_m': [xyz(v) for v in box.get_world_vertices(t)]})
                # GT is frozen BEFORE advancing to obtain delayed sensor readback.
                observed_frame = world.tick(config['sensor_timeout_s'])
                if observed_frame != frame + 1:
                    raise ValueError('Unexpected readback distance')
                feature = {'schema_version': 1, 'source': 'carla_rgb_only', 'scene_id': 'instance-probe',
                    'frame': frame, 'timestamp_s': stamp, 'calibration': 'calibration/probe.json', 'cameras': {}}
                jobs = []
                for (name, modality), sink in queues.items():
                    captured = receive_frame(sink, frame, config['sensor_timeout_s'])
                    image = captured.image
                    error = abs(image.timestamp - stamp)
                    if not np.isfinite(error) or error > config['max_timestamp_error_s']:
                        raise ValueError('Sensor/snapshot timestamp mismatch')
                    if (image.width, image.height) != (rig['width'], rig['height']):
                        raise ValueError('Capture dimensions mismatch')
                    report['max_timestamp_error_s'] = max(report['max_timestamp_error_s'], error)
                    camera = label['cameras'].setdefault(name, {'K': camera_definitions[[d['name'] for d in camera_definitions].index(name)]['K'],
                        'T_world_from_camera': captured.world_from_camera, 'modalities': {}})
                    difference = float(np.abs(np.array(camera['T_world_from_camera']) - captured.world_from_camera).max())
                    report['max_pose_error'] = max(report['max_pose_error'], difference)
                    if difference > config['max_capture_pose_error']:
                        raise ValueError('Modality camera poses do not match')
                    base = 'inputs/rgb' if modality == 'rgb' else 'labels/' + modality
                    relative = f'{base}/instance-probe/{name}/{frame:08d}.png'
                    path = output / relative; path.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append(encoder.submit(image.save_to_disk, path))
                    entry = {'image': relative, 'frame': frame, 'timestamp_s': image.timestamp}
                    camera['modalities'][modality] = entry
                    if modality == 'rgb':
                        feature['cameras'][name] = entry
                    saved.append(relative)
                for job in jobs:
                    job.result()
                input_relative = f'inputs/frames/instance-probe/{frame:08d}.json'
                label_relative = f'labels/frames/{frame:08d}.json'
                write_json(output / input_relative, feature)
                write_json(output / label_relative, label)
                snapshots.append({'case': case['name'], 'input': input_relative, 'label': label_relative})
                report['bundles'] += 1; report['images'] += len(jobs)
                report['cases'].append({'name': case['name'], 'frame': frame})
                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                report['gpu_samples_mib'].append(int(gpu.strip()))
                write_json(output / 'report.json', report)
        report['checks'] = {'sensors_32': len(sensors) == 32, 'five_bundles': len(snapshots) == 5,
                            'raw_images_160': len(saved) == 160}
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if client is not None:
            client.set_timeout(3 if report['errors'] else config['sensor_timeout_s'])
        for sensor in reversed(sensors):
            try:
                sensor.stop()
            except Exception as exc:
                cleanup = False; report['errors'].append('stop: ' + str(exc))
        for actor in reversed(actors):
            try:
                if not actor.destroy():
                    raise RuntimeError('destroy returned false')
            except Exception as exc:
                cleanup = False; report['errors'].append('destroy: ' + str(exc))
        if world is not None and original is not None:
            try:
                world.set_weather(original_weather)
                world.apply_settings(original)
            except Exception as exc:
                cleanup = False; report['errors'].append('restore: ' + str(exc))
        report['cleanup'] = cleanup
        report['wall_s'] = time.monotonic() - started
        report['warmup_queue_drops'] = {f'{k[0]}/{k[1]}': n for k, n in dropped.items()}
        report['status'] = 'passed' if cleanup and not report['errors'] and all(report.get('checks', {}).values()) and report['bundles'] == 5 else 'failed'
        write_json(output / 'index.json', snapshots)
        inventory = [{'path': name, 'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for name in saved if (output / name).exists()]
        (output / 'image_inventory.jsonl').write_text(''.join(json.dumps(i) + '\n' for i in inventory))
        write_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/instance_probe_v1.json')
    parser.add_argument('--rig', type=Path, default=ROOT / 'configs/camera_rig_v1.json')
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(collect(args.output.resolve(), args.config, args.rig, args.baseline))
