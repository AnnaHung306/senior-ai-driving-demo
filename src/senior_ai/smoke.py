"""Phase 0 only: measured vehicle control and a real RGB sensor in CARLA.

This uses simulator ego telemetry for acceptance. It is neither AI perception nor
autonomous routing. Run against a dedicated local server: the map is reloaded.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import time
from .core import assess, intrinsics, receive_frame, write_json


def speed(vector):
    return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def run(args):
    criteria_bytes = args.criteria.read_bytes()
    criteria = json.loads(criteria_bytes)
    if criteria['expected_frames'] != 140:
        raise ValueError('This phase-0 protocol has exactly 140 frames')
    args.output.mkdir(parents=True, exist_ok=False)
    frames_dir = args.output / 'sensors' / 'front'
    frames_dir.mkdir(parents=True)
    evaluation_dir = args.output / 'evaluation'
    evaluation_dir.mkdir()
    write_json(args.output / 'criteria.json', criteria)
    report = {'schema_version': 1, 'task': 'phase0-carla-control-camera',
              'status': 'failed', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'criteria_sha256': hashlib.sha256(criteria_bytes).hexdigest(),
              'provenance': {'images': 'carla_rgb_sensor', 'ego_state': 'simulator_ground_truth',
                             'control': 'open_loop_test_protocol', 'ai_model': None},
              'errors': [], 'checks': {}, 'metrics': {}}
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                  text=True, timeout=5)
        report['git_revision'] = revision.stdout.strip() if revision.returncode == 0 else 'uncommitted'
    except (OSError, subprocess.SubprocessError):
        report['git_revision'] = 'unavailable'
    actors, sensors, saved_settings, world, vehicle = [], [], None, None, None
    observations, samples, saved, max_error = [], [], 0, 0.0
    collisions = queue.Queue()
    started = time.monotonic()
    try:
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        report['client_version'] = client.get_client_version()
        report['server_version'] = client.get_server_version()
        if report['client_version'] != '0.9.16' or report['server_version'] != '0.9.16':
            raise RuntimeError('Client and server must both be CARLA 0.9.16')
        world = client.load_world(args.map)
        saved_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        world = client.reload_world(False)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        library = world.get_blueprint_library()
        bp = library.find('vehicle.tesla.model3')
        bp.set_attribute('role_name', 'phase0_acceptance')
        points = world.get_map().get_spawn_points()
        for index, transform in enumerate(points):
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle:
                report['spawn_index'] = index
                break
        if vehicle is None:
            raise RuntimeError('No free spawn point')
        actors.append(vehicle)
        camera_bp = library.find('sensor.camera.rgb')
        for key, value in {'image_size_x': '640', 'image_size_y': '360',
                           'fov': '90', 'sensor_tick': '0.05'}.items():
            camera_bp.set_attribute(key, value)
        extrinsic = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, extrinsic, attach_to=vehicle)
        actors.append(camera)
        sensors.append(camera)
        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        collision = world.spawn_actor(library.find('sensor.other.collision'),
                                      carla.Transform(), attach_to=vehicle)
        actors.append(collision)
        sensors.append(collision)
        collision.listen(collisions.put)
        write_json(args.output / 'calibration.json', {
            'schema_version': 1, 'camera': 'front', 'width': 640, 'height': 360,
            'horizontal_fov_deg': 90, 'intrinsics': intrinsics(640, 360, 90),
            'camera_local_to_ego': extrinsic.get_matrix(),
            'coordinate_system': 'CARLA left-handed: x forward, y right, z up; meters',
            'optical_from_carla_camera': [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
            'sensor_tick_s': 0.05})
        with (evaluation_dir / 'ego.jsonl').open('w', encoding='utf-8') as log:
            for step in range(140):
                stage = 'settle' if step < 20 else 'accelerate' if step < 60 else 'brake'
                control = carla.VehicleControl(throttle=0.55 if stage == 'accelerate' else 0.0,
                                               brake=0.0 if stage == 'accelerate' else 1.0)
                vehicle.apply_control(control)
                frame = world.tick(args.timeout)
                snapshot = world.get_snapshot()
                if snapshot.frame != frame:
                    raise RuntimeError('Multiple clients are ticking this server')
                image = receive_frame(image_queue, frame, args.timeout)
                if image.width != 640 or image.height != 360 or len(image.raw_data) != 640 * 360 * 4:
                    raise RuntimeError('Malformed RGB sensor image')
                timestamp_error = abs(image.timestamp - snapshot.timestamp.elapsed_seconds)
                max_error = max(max_error, timestamp_error)
                ego = snapshot.find(vehicle.id)
                if ego is None:
                    raise RuntimeError('Ego vehicle missing from snapshot')
                pose = ego.get_transform()
                record = {'frame': frame, 'timestamp_s': snapshot.timestamp.elapsed_seconds,
                          'stage': stage, 'speed_mps': speed(ego.get_velocity()),
                          'position_m': [pose.location.x, pose.location.y, pose.location.z],
                          'yaw_deg': pose.rotation.yaw,
                          'control': {'throttle': control.throttle, 'brake': control.brake,
                                      'steer': control.steer},
                          'source': 'simulator_ground_truth_evaluation_only'}
                observations.append(record)
                log.write(json.dumps(record, allow_nan=False) + '\n')
                if step % 10 == 0:
                    path = frames_dir / f'{frame:08d}.png'
                    image.save_to_disk(str(path))
                    if not path.exists() or path.stat().st_size < 100:
                        raise RuntimeError('Camera image was not saved')
                    samples.append({'frame': frame, 'timestamp_s': image.timestamp,
                                    'image': path.relative_to(args.output).as_posix()})
                    saved += 1
        write_json(args.output / 'sensor_manifest.json', {'samples': samples, 'camera_count': 1,
                    'provenance': 'simulator_rgb', 'purpose': 'phase0_smoke_only'})
    except (Exception, KeyboardInterrupt) as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        cleanup_ok = True
        for sensor in reversed(sensors):
            try:
                sensor.stop()
            except Exception as exc:
                cleanup_ok = False
                report['errors'].append(f'Sensor cleanup: {exc}')
        for actor in reversed(actors):
            try:
                if not actor.destroy():
                    raise RuntimeError(f'Actor {actor.id} could not be destroyed')
            except Exception as exc:
                cleanup_ok = False
                report['errors'].append(f'Actor cleanup: {exc}')
        if world is not None and saved_settings is not None:
            try:
                world.apply_settings(saved_settings)
            except Exception as exc:
                cleanup_ok = False
                report['errors'].append(f'World settings restore: {exc}')
        collision_events = []
        while not collisions.empty():
            event = collisions.get_nowait()
            collision_events.append({'frame': event.frame, 'other_actor': event.other_actor.id})
        write_json(evaluation_dir / 'collisions.json', collision_events)
        if observations:
            begin = observations[min(19, len(observations) - 1)]['position_m']
            end = observations[-1]['position_m']
            metrics = {'distance_m': math.dist(begin, end),
                       'max_speed_mps': max(o['speed_mps'] for o in observations),
                       'final_speed_mps': observations[-1]['speed_mps'],
                       'saved_images': saved, 'max_timestamp_error_s': max_error,
                       'observed_frames': len(observations),
                       'collision_count': len(collision_events), 'cleanup_ok': cleanup_ok}
            report['metrics'] = metrics
            report['checks'] = assess(metrics, criteria)
            if all(report['checks'].values()) and not report['errors']:
                report['status'] = 'passed'
        report['wall_time_s'] = round(time.monotonic() - started, 3)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'report.json', report)
    print(json.dumps({'task': report['task'], 'status': report['status'],
                      'metrics': report['metrics'], 'errors': report['errors'],
                      'report': str(args.output / 'report.json')}))
    return 0 if report['status'] == 'passed' else 1
