"""Bounded, file-logged CARLA experiments motivated by symbolized crash evidence.

Each case gets a fresh owned server, immutable configuration, a separate worker,
finite deadlines and a result. No language-model observation in the tick loop.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import receive_frame, write_json
from senior_ai.frames import RgbFrame


def worker(config, output):
    import carla
    result = {'status': 'failed', 'purpose': 'diagnostic_only', 'config': config,
              'events': [], 'errors': [], 'stage': 'connect', 'rgb_frames': 0}
    actors, world, original = [], None, None
    callback_frames = []
    snapshot_queue, snapshot_callback = queue.Queue(), None
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30)
    try:
        result['server_version'] = client.get_server_version()
        result['client_version'] = client.get_client_version()
        expected = config.get('expected_version', '0.9.16')
        if result['server_version'] != expected or result['client_version'] != expected:
            raise RuntimeError('Client, server and declared experiment versions must match')
        world = client.get_world() if config['setup'] == 'current' else client.load_world('Town01')
        result['map'] = world.get_map().name
        if result['map'].split('/')[-1] != 'Town01':
            raise RuntimeError('Requested Town01 but server loaded another map')
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = not config.get('asynchronous', False)
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        if config['setup'] == 'load-reload':
            world = client.reload_world(False)
        readback = world.get_settings()
        result['settings_readback'] = {'sync': readback.synchronous_mode,
            'delta': readback.fixed_delta_seconds, 'no_rendering': readback.no_rendering_mode}
        def advance():
            return world.wait_for_tick(5).frame if config.get('asynchronous') else world.tick(5)
        client.set_timeout(5)
        result['stage'] = 'empty_world_render'
        write_json(output / 'report.json', result)
        for _ in range(10):
            frame = advance()
            result['events'].append({'stage': result['stage'], 'frame': frame})
        result['stage'] = 'vehicle_physics'
        write_json(output / 'report.json', result)
        vehicle = world.spawn_actor(world.get_blueprint_library().find('vehicle.tesla.model3'),
                                    world.get_map().get_spawn_points()[0])
        actors.append(vehicle)
        for _ in range(20):
            frame = advance()
            result['events'].append({'stage': result['stage'], 'frame': frame})
        result['stage'] = 'rgb_capture'
        write_json(output / 'report.json', result)
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        for key, value in {'image_size_x': '640', 'image_size_y': '360',
                           'fov': '90', 'sensor_tick': config.get('sensor_tick', '0.05')}.items():
            bp.set_attribute(key, value)
        camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=1.7)), attach_to=vehicle)
        actors.append(camera)
        images = queue.Queue()
        if config.get('snapshot_callback'):
            snapshot_callback = world.on_tick(snapshot_queue.put)
        def record_image(sample):
            if config.get('copy_image_data'):
                sample = RgbFrame.from_carla(sample)
            callback_frames.append({'frame': sample.frame, 'timestamp': sample.timestamp})
            if len(callback_frames) <= config.get('save_startup_images', 0):
                sample.save_to_disk(str(output / f'startup-{sample.frame:08d}.png'))
            images.put(sample)
        camera.listen(record_image)
        result['sensor_warmup_ticks'] = config.get('sensor_warmup_ticks', 0)
        for _ in range(result['sensor_warmup_ticks']):
            advance()
            time.sleep(0.05)
        produced = []
        if config.get('collect_before_join'):
            result['stage'] = 'buffered_tick_production'
            result['requested_frames'] = produced
            for _ in range(config['collect_before_join']):
                produced.append(advance())
                time.sleep(0.05)
            time.sleep(0.2)
            result['stage'] = 'buffered_frame_join'
        for index in range(len(produced) if produced else 3):
            frame = produced[index] if produced else advance()
            time.sleep(config.get('post_tick_sleep_s', 0))
            snapshot = receive_frame(snapshot_queue, frame, 5) if config.get('snapshot_callback') else world.get_snapshot()
            for _ in range(config.get('prefetch_ticks', 0)):
                advance()
            sample = receive_frame(images, frame, 5)
            if abs(sample.timestamp - snapshot.timestamp.elapsed_seconds) > 0.0001:
                raise RuntimeError('Image and original snapshot timestamp mismatch')
            sample.save_to_disk(str(output / f'{frame:08d}.png'))
            result['rgb_frames'] += 1
            result['events'].append({'stage': result['stage'], 'frame': frame,
                                      'image_frame': sample.frame,
                                      'timestamp_error_s': abs(sample.timestamp - snapshot.timestamp.elapsed_seconds),
                                      'observed_world_frame': produced[-1] if produced else frame + config.get('prefetch_ticks', 0)})
        result['status'] = 'passed'
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        client.set_timeout(2)
        if snapshot_callback is not None:
            world.remove_on_tick(snapshot_callback)
        for actor in reversed(actors):
            try:
                if not actor.destroy():
                    raise RuntimeError(f'Cannot destroy actor {actor.id}')
            except Exception as exc:
                result['errors'].append(f'cleanup: {exc}')
        if world is not None and original is not None:
            try:
                world.apply_settings(original)
            except Exception as exc:
                result['errors'].append(f'restore: {exc}')
        if result['errors']:
            result['status'] = 'failed'
        result['callback_frames'] = callback_frames
        write_json(output / 'report.json', result)
    return 0 if result['status'] == 'passed' else 1


def batch(args):
    plan = json.loads(args.plan.read_text())
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Reports must remain inside this project')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    server_exe = args.server_root.resolve() / 'CarlaUE4.exe'
    if not server_exe.is_file():
        raise FileNotFoundError(server_exe)
    shipping = server_exe.parent / 'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'
    binary_hash = hashlib.sha256(shipping.read_bytes()).hexdigest()
    summary = {'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [],
               'server_binary_sha256': binary_hash, 'runtime': sys.version,
               'runtime_executable': sys.executable, 'expected_version': args.expected_version,
               'plan_version': plan['version']}
    for config in plan['cases']:
        if args.case and config['id'] != args.case:
            continue
        config = dict(config, expected_version=args.expected_version)
        # Do not disturb a pre-existing CARLA server.
        try:
            with socket.create_connection(('127.0.0.1', 2000), timeout=1):
                raise RuntimeError('Port 2000 already in use; no server was stopped')
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            if isinstance(exc, RuntimeError):
                raise
        case_dir = output / config['id']
        case_dir.mkdir()
        write_json(case_dir / 'config.json', config)
        command = [str(server_exe)]
        if config['direct_map']:
            command.append('/Game/Carla/Maps/Town01')
        command += [f"-{config.get('graphics_api', 'dx11')}", f"-quality-level={config['quality']}",
                    '-carla-rpc-port=2000', '-nosound', '-unattended']
        command += ['-RenderOffScreen'] if config.get('offscreen', True) else ['-windowed', '-ResX=640', '-ResY=360']
        command += config.get('additional_args', [])
        write_json(case_dir / 'launch.json', {'command': command,
                    'server_binary_sha256': binary_hash,
                    'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        case = {'id': config['id'], 'status': 'failed'}
        with (case_dir / 'server.log').open('w') as server_log:
            process = subprocess.Popen(command, cwd=server_exe.parent, startupinfo=startup,
                                       stdout=server_log, stderr=subprocess.STDOUT)
            try:
                import carla
                ready, deadline = False, time.monotonic() + 180
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        client = carla.Client('127.0.0.1', 2000)
                        client.set_timeout(2)
                        client.get_server_version()
                        ready = True
                        break
                    except RuntimeError:
                        time.sleep(2)
                if not ready:
                    raise RuntimeError('Server startup deadline or early exit')
                with (case_dir / 'worker.log').open('w') as worker_log:
                    completed = subprocess.run([sys.executable, __file__, 'worker',
                                                '--config', str(case_dir / 'config.json'),
                                                '--output', str(case_dir)], cwd=ROOT,
                                               stdout=worker_log, stderr=subprocess.STDOUT, timeout=120)
                case['exit_code'] = completed.returncode
                if (case_dir / 'report.json').exists():
                    report = json.loads((case_dir / 'report.json').read_text())
                    case.update(status=report['status'], stage=report['stage'],
                                rgb_frames=report['rgb_frames'], errors=report['errors'],
                                callback_count=len(report.get('callback_frames', [])))
            except Exception as exc:
                case['error'] = f'{type(exc).__name__}: {exc}'
            finally:
                if config.get('dump_hang') and case['status'] != 'passed' and process.poll() is None:
                    with (case_dir / 'dump.log').open('w') as dump_log:
                        subprocess.run([sys.executable, str(ROOT / 'scripts/capture_carla_dump.py'),
                                        '--server-root', str(server_exe.parent), '--output', str(case_dir / 'hang.dmp')],
                                       stdout=dump_log, stderr=subprocess.STDOUT, timeout=45)
                if process.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   capture_output=True, timeout=15)
                    process.wait(timeout=15)
                # Windows can retain the listening socket briefly during GPU teardown.
                # Never let the next case connect to that old endpoint.
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    try:
                        with socket.create_connection(('127.0.0.1', 2000), timeout=0.5):
                            time.sleep(0.25)
                    except OSError:
                        break
                else:
                    case['status'] = 'failed'
                    case['cleanup_error'] = 'Port 2000 did not close; no next case allowed'
                case['launcher_exit_code'] = process.returncode
        summary['cases'].append(case)
        write_json(output / 'summary.json', summary)
        print(json.dumps(case), flush=True)
        if case.get('cleanup_error'):
            break
        if case['status'] == 'passed':
            break
    return 0 if any(c['status'] == 'passed' for c in summary['cases']) else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('batch')
    b.add_argument('--plan', type=Path, default=ROOT / 'configs/landscape_diagnostics_v1.json')
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--case')
    b.add_argument('--server-root', type=Path, default=ROOT / 'artifacts/carla')
    b.add_argument('--expected-version', choices=['0.9.15', '0.9.16'], default='0.9.16')
    w = sub.add_parser('worker')
    w.add_argument('--config', type=Path, required=True)
    w.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'batch':
        return batch(args)
    return worker(json.loads(args.config.read_text()), args.output)


if __name__ == '__main__':
    raise SystemExit(main())
