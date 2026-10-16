"""Bounded diagnostic: separate first vehicle physics tick from RGB capture.

This is not the phase-0 acceptance test. Simulator telemetry is diagnostic GT.
"""
import json
import argparse
from pathlib import Path
import queue
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle', default='vehicle.tesla.model3')
    args = parser.parse_args()
    import carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30)
    actors, settings, world = [], None, None
    result = {'status': 'failed', 'stage': 'connect', 'events': [], 'errors': [],
              'purpose': 'diagnostic_only_not_acceptance', 'vehicle': args.vehicle}
    try:
        world = client.load_world('Town01')
        settings = world.get_settings()
        sync = world.get_settings()
        sync.synchronous_mode, sync.fixed_delta_seconds = True, 0.05
        world.apply_settings(sync)
        world = client.reload_world(False)
        client.set_timeout(5)
        result['stage'] = 'vehicle_physics_without_camera'
        vehicle = world.spawn_actor(world.get_blueprint_library().find(args.vehicle),
                                    world.get_map().get_spawn_points()[0])
        actors.append(vehicle)
        for _ in range(3):
            frame = world.tick(5)
            result['events'].append({'stage': result['stage'], 'frame': frame})
        write_json(ROOT / 'artifacts/reports/isolation-probe.json', result)
        result['stage'] = 'rgb_camera'
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        for key, value in {'image_size_x': '640', 'image_size_y': '360', 'sensor_tick': '0.05'}.items():
            bp.set_attribute(key, value)
        camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=1.7)), attach_to=vehicle)
        actors.append(camera)
        images = queue.Queue()
        camera.listen(images.put)
        frame = world.tick(5)
        image = images.get(timeout=5)
        result['events'].append({'stage': result['stage'], 'tick_frame': frame, 'image_frame': image.frame})
        result['status'] = 'passed'
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        client.set_timeout(2)
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception as exc:
                result['errors'].append(f'cleanup: {exc}')
        if world is not None and settings is not None:
            try:
                world.apply_settings(settings)
            except Exception as exc:
                result['errors'].append(f'restore: {exc}')
        write_json(ROOT / 'artifacts/reports/isolation-probe.json', result)
        print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
