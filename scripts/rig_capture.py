"""Eight-camera synchronized RGB/calibration smoke. Not an AI/training dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import intrinsics, receive_frame, write_json
from senior_ai.frames import RgbFrame


def validate_rig(rig):
    cameras = rig['cameras']
    names = [c['name'] for c in cameras]
    headings = sorted(c['yaw_deg'] % 360 for c in cameras)
    if len(cameras) != 8 or len(set(names)) != 8 or any(not n.replace('_', '').isalnum() for n in names):
        raise ValueError('Rig must contain eight uniquely named cameras')
    if len(set(headings)) != 8:
        raise ValueError('Eight distinct camera directions required')
    gaps = [b - a for a, b in zip(headings, headings[1:] + [headings[0] + 360])]
    if max(gaps) > rig['horizontal_fov_deg']:
        raise ValueError('Camera yaw/FOV configuration leaves a horizontal coverage gap')
    intrinsics(rig['width'], rig['height'], rig['horizontal_fov_deg'])
    for camera in cameras:
        if len(camera['position_m']) != 3 or not all(math.isfinite(v) for v in camera['position_m']):
            raise ValueError('Invalid camera extrinsics')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rig', type=Path, default=ROOT / 'configs/camera_rig_v1.json')
    parser.add_argument('--criteria', type=Path, default=ROOT / 'configs/rig_acceptance_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rig, criteria = json.loads(args.rig.read_text()), json.loads(args.criteria.read_text())
    validate_rig(rig)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'rig.json', rig)
    write_json(args.output / 'criteria.json', criteria)
    report = {'task': 'eight-camera-rig-smoke', 'status': 'failed', 'errors': [], 'checks': {},
              'ai_model': None, 'training_dataset': False, 'labels_collected': False,
              'rig_sha256': hashlib.sha256(args.rig.read_bytes()).hexdigest(),
              'criteria_sha256': hashlib.sha256(args.criteria.read_bytes()).hexdigest(),
              'scope': criteria['scope']}
    report['server_binary_sha256'] = hashlib.sha256((ROOT / 'artifacts/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe').read_bytes()).hexdigest()
    report['git_revision'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    actors, cameras, original, world = [], [], None, None
    queues, bundles, latencies, gpu_samples = {}, [], [], []
    saved, max_error, cleanup_ok = 0, 0.0, True
    capture_started = None
    client = None
    try:
        import carla
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(15)
        if client.get_server_version() != '0.9.16' or client.get_client_version() != '0.9.16':
            raise RuntimeError('Rig requires matching CARLA 0.9.16')
        world = client.get_world()
        report['stage'] = 'world_setup'
        write_json(args.output / 'report.json', report)
        if world.get_map().name.split('/')[-1] != 'Town01':
            raise RuntimeError('Use the already loaded Town01; no redundant map reload permitted')
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, criteria['fixed_delta_s']
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        for _ in range(10):
            world.tick(15)
        library = world.get_blueprint_library()
        vehicle = world.spawn_actor(library.find('vehicle.tesla.model3'), world.get_map().get_spawn_points()[0])
        actors.append(vehicle)
        calibrations = []
        for definition in rig['cameras']:
            name = definition['name']
            folder = args.output / 'sensors' / name
            folder.mkdir(parents=True)
            bp = library.find('sensor.camera.rgb')
            for key, value in {'image_size_x': rig['width'], 'image_size_y': rig['height'],
                               'fov': rig['horizontal_fov_deg'], 'sensor_tick': rig['sensor_tick_s']}.items():
                bp.set_attribute(key, str(value))
            transform = carla.Transform(carla.Location(*definition['position_m']), carla.Rotation(yaw=definition['yaw_deg']))
            camera = world.spawn_actor(bp, transform, attach_to=vehicle)
            actors.append(camera)
            cameras.append(camera)
            target = queue.Queue()
            queues[name] = target
            camera.listen(lambda image, sink=target: sink.put(RgbFrame.from_carla(image)))
            calibrations.append({'name': name, 'intrinsics': intrinsics(rig['width'], rig['height'], rig['horizontal_fov_deg']),
                                 'camera_local_to_ego': transform.get_matrix()})
        write_json(args.output / 'calibration.json', {'schema_version': 1, 'coordinate_system': rig['coordinate_system'],
                    'optical_from_carla_camera': [[0, 1, 0], [0, 0, -1], [1, 0, 0]], 'cameras': calibrations})
        report['stage'] = 'sensor_warmup'
        write_json(args.output / 'report.json', report)
        for _ in range(5):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            world.tick(15)
        capture_started = time.monotonic()
        report['stage'] = 'capture'
        write_json(args.output / 'report.json', report)
        pending = None
        with ThreadPoolExecutor(max_workers=4) as encoder, (args.output / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
            def consume(item, observed_frame):
                nonlocal saved, max_error
                frame, timestamp, issued = item
                if observed_frame - frame != criteria['readback_lag_ticks']:
                    raise RuntimeError('Unexpected GPU readback frame distance')
                images = {name: receive_frame(source, frame, 15) for name, source in queues.items()}
                jobs, entries = [], {}
                for name, image in images.items():
                    error = abs(image.timestamp - timestamp)
                    if not math.isfinite(error):
                        raise RuntimeError('Invalid camera timestamp')
                    max_error = max(max_error, error)
                    if image.width != rig['width'] or image.height != rig['height']:
                        raise RuntimeError('Image resolution does not match calibration')
                    path = args.output / 'sensors' / name / f'{frame:08d}.png'
                    jobs.append(encoder.submit(image.save_to_disk, path))
                    entries[name] = {'image': path.relative_to(args.output).as_posix(), 'frame': image.frame,
                                     'timestamp_s': image.timestamp}
                for job in jobs:
                    job.result()
                for entry in entries.values():
                    path = args.output / entry['image']
                    if path.stat().st_size < 100 or path.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
                        raise RuntimeError('Invalid saved sensor PNG')
                saved += len(entries)
                bundle = {'schema_version': 1, 'frame': frame, 'timestamp_s': timestamp,
                          'source': 'carla_rgb_sensors', 'cameras': entries, 'calibration': 'calibration.json'}
                bundles.append(bundle)
                manifest.write(json.dumps(bundle, allow_nan=False) + '\n')
                latencies.append((time.monotonic() - issued) * 1000)
                if len(bundles) % 5 == 0:
                    try:
                        sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                                capture_output=True, text=True, check=True, timeout=5,
                                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        gpu_samples.append(int(sample.stdout.strip().splitlines()[0]))
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
            for _ in range(criteria['expected_bundles']):
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
                frame = world.tick(15)
                snapshot = world.get_snapshot()
                if snapshot.frame != frame:
                    raise RuntimeError('Multiple tick clients detected')
                if pending is not None:
                    consume(pending, frame)
                pending = (frame, snapshot.timestamp.elapsed_seconds, time.monotonic())
            flush = world.tick(15)
            consume(pending, flush)
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if client is not None:
            client.set_timeout(2)
        write_json(args.output / 'report.json', report)
        capture_wall = time.monotonic() - capture_started if capture_started else None
        for camera in reversed(cameras):
            try:
                camera.stop()
            except Exception as exc:
                cleanup_ok = False
                report['errors'].append(f'sensor cleanup: {exc}')
        for actor in reversed(actors):
            try:
                if not actor.destroy():
                    raise RuntimeError('Actor destroy failed')
            except Exception as exc:
                cleanup_ok = False
                report['errors'].append(f'actor cleanup: {exc}')
        if world is not None and original is not None:
            try:
                world.apply_settings(original)
            except Exception as exc:
                cleanup_ok = False
                report['errors'].append(f'world restore: {exc}')
        report['checks'] = {'eight_cameras': len(cameras) == criteria['expected_cameras'],
                            'complete_bundles': len(bundles) == criteria['expected_bundles'],
                            'saved_images': saved == criteria['expected_saved_images'],
                            'timestamp_alignment': bool(bundles) and max_error <= criteria['max_timestamp_error_s'],
                            'cleanup': cleanup_ok}
        report['metrics'] = {'bundles': len(bundles), 'saved_images': saved,
                             'max_timestamp_error_s': max_error, 'capture_wall_s': capture_wall,
                             'wall_bundle_rate_hz': len(bundles) / capture_wall if capture_wall else None,
                             'mean_bundle_commit_latency_ms': sum(latencies) / len(latencies) if latencies else None,
                             'max_bundle_commit_latency_ms': max(latencies) if latencies else None,
                             'sampled_total_gpu_used_mib_max': max(gpu_samples) if gpu_samples else None}
        if all(report['checks'].values()) and not report['errors']:
            report['status'] = 'passed'
        write_json(args.output / 'report.json', report)
    print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
