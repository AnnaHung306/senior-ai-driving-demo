"""Fixed-route CARLA BasicAgent baseline; explicitly not learned perception/AI.

Prepare freezes routes using map geometry before any driving results are observed.
Run records ego GT, collision/lane events, arrival and bounded task outcomes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json


def xyz(location):
    return [location.x, location.y, location.z]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args, criteria, client):
    if args.scenarios.exists():
        raise FileExistsError('Frozen scenarios already exist; use a new versioned file')
    world = client.get_world()
    road_map = world.get_map()
    if road_map.name.split('/')[-1] != criteria['map']:
        raise RuntimeError('Prepare requires the declared map already loaded')
    import carla
    routes, used_roads = [], set()
    for index, transform in enumerate(road_map.get_spawn_points()):
        wp = road_map.get_waypoint(transform.location, lane_type=carla.LaneType.Driving)
        if wp.is_junction or wp.road_id in used_roads:
            continue
        path, valid = [wp], True
        # Require 10m beyond the goal to remain away from intersections.
        for _ in range(25):
            candidates = path[-1].next(2.0)
            if len(candidates) != 1:
                valid = False
                break
            following = candidates[0]
            angle = (following.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180
            if following.is_junction or following.road_id != wp.road_id or following.lane_id != wp.lane_id or abs(angle) > 15:
                valid = False
                break
            path.append(following)
        if not valid:
            continue
        goal = path[20].transform.location
        if transform.location.distance(goal) < 35:
            continue
        routes.append({'id': f'town01-road-{wp.road_id}-lane-{wp.lane_id}', 'spawn_index': index,
                       'road_id': wp.road_id, 'lane_id': wp.lane_id,
                       'start': {'position_m': xyz(transform.location),
                                 'rotation_deg': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll]},
                       'goal_m': xyz(goal), 'path_m': [xyz(p.transform.location) for p in path[:21]]})
        used_roads.add(wp.road_id)
        if len(routes) == criteria['scenario_count']:
            break
    if len(routes) != criteria['scenario_count']:
        raise RuntimeError('Not enough qualifying routes; no threshold was changed')
    frozen = {'schema_version': 1, 'map': road_map.name,
              'opendrive_sha256': hashlib.sha256(road_map.to_opendrive().encode()).hexdigest(),
              'criteria_sha256': digest(args.criteria), 'selection': 'First five distinct roads with a straight, non-junction 50m corridor; goal at 40m. Selection used geometry only, no driving results.',
              'routes': routes}
    write_json(args.scenarios, frozen)
    print(json.dumps({'status': 'frozen', 'routes': [r['id'] for r in routes],
                      'scenarios_sha256': digest(args.scenarios)}))
    return 0


def run_case(world, road_map, planner, route, criteria, case_dir):
    import carla
    from agents.navigation.basic_agent import BasicAgent
    case_dir.mkdir()
    original = world.get_settings()
    actors, sensors, collisions, invasions = [], [], [], []
    report = {'id': route['id'], 'status': 'failed', 'provenance': 'carla_basic_agent_rules_with_simulator_gt',
              'errors': [], 'checks': {}, 'metrics': {}, 'seed': 42, 'traffic_compliance': 'not_exercised'}
    final_speed, distance, elapsed, off_lane, max_center_error = math.inf, math.inf, 0.0, 0, 0.0
    arrived, agent_done, cleanup_ok, path_length = False, False, True, 0.0
    previous_position = None
    started = time.monotonic()
    try:
        random.seed(42)
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, criteria['fixed_delta_s']
        settings.no_rendering_mode = True
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        for _ in range(10):
            world.tick(10)
        transform = carla.Transform(carla.Location(*route['start']['position_m']),
                                    carla.Rotation(*route['start']['rotation_deg']))
        bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        bp.set_attribute('role_name', 'rules_baseline')
        vehicle = world.spawn_actor(bp, transform)
        actors.append(vehicle)
        library = world.get_blueprint_library()
        for kind, sink in [('sensor.other.collision', collisions), ('sensor.other.lane_invasion', invasions)]:
            sensor = world.spawn_actor(library.find(kind), carla.Transform(), attach_to=vehicle)
            actors.append(sensor)
            sensors.append(sensor)
            sensor.listen(lambda event, target=sink, label=kind: target.append({'frame': event.frame, 'type': label}))
        for _ in range(20):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            world.tick(10)
        agent = BasicAgent(vehicle, target_speed=criteria['target_speed_kph'],
                           opt_dict=dict(criteria['controller_options']), map_inst=road_map, grp_inst=planner)
        goal = carla.Location(*route['goal_m'])
        agent.set_destination(goal, start_location=vehicle.get_location())
        limit = round(criteria['max_simulation_seconds_per_route'] / criteria['fixed_delta_s'])
        with (case_dir / 'ego.jsonl').open('w', encoding='utf-8') as log:
            for step in range(limit):
                if time.monotonic() - started > 180:
                    raise TimeoutError('Route exceeded 180 second wall-clock limit')
                agent_done = agent.done()
                control = carla.VehicleControl(brake=1.0) if agent_done else agent.run_step()
                vehicle.apply_control(control)
                frame = world.tick(10)
                snapshot = world.get_snapshot()
                if snapshot.frame != frame:
                    raise RuntimeError('Another client advanced this dedicated world')
                state = snapshot.find(vehicle.id)
                pose, velocity = state.get_transform(), state.get_velocity()
                location = pose.location
                final_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                distance = location.distance(goal)
                elapsed = (step + 1) * criteria['fixed_delta_s']
                projected = road_map.get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
                if projected is None:
                    off_lane += 1
                else:
                    center = projected.transform.location
                    max_center_error = max(max_center_error, math.hypot(location.x-center.x, location.y-center.y))
                if previous_position is not None:
                    path_length += location.distance(previous_position)
                previous_position = location
                record = {'frame': frame, 'elapsed_s': elapsed, 'position_m': xyz(location),
                          'speed_mps': final_speed, 'goal_distance_m': distance, 'agent_done': agent_done,
                          'control': {'throttle': control.throttle, 'brake': control.brake, 'steer': control.steer},
                          'source': 'simulator_ground_truth_evaluation_only'}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                if agent_done and distance <= criteria['arrival_radius_m'] and final_speed <= criteria['max_final_speed_mps']:
                    arrived = True
                    break
                if collisions or invasions or off_lane:
                    break
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(20):
            world.tick(10)
        time.sleep(0.05)
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        for sensor in reversed(sensors):
            try:
                sensor.stop()
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
        try:
            world.apply_settings(original)
        except Exception as exc:
            cleanup_ok = False
            report['errors'].append(f'world restore: {exc}')
    report['checks'] = {'arrival': arrived, 'agent_done': agent_done,
                        'collision_free': len(collisions) <= criteria['max_collisions'],
                        'lane_invasion_free': len(invasions) <= criteria['max_lane_invasion_events'],
                        'on_driving_lane': off_lane <= criteria['max_off_driving_lane_ticks'],
                        'within_time': elapsed <= criteria['max_simulation_seconds_per_route'],
                        'cleanup': cleanup_ok}
    report['metrics'] = {'goal_distance_m': distance if math.isfinite(distance) else None,
                         'final_speed_mps': final_speed if math.isfinite(final_speed) else None,
                         'simulation_seconds': elapsed, 'path_length_m': path_length,
                         'collisions': len(collisions), 'lane_invasion_events': len(invasions),
                         'off_driving_lane_ticks': off_lane, 'max_lane_center_error_m': max_center_error}
    report['events'] = {'collisions': collisions, 'lane_invasions': invasions}
    report['wall_seconds'] = time.monotonic() - started
    if all(report['checks'].values()) and not report['errors']:
        report['status'] = 'passed'
    write_json(case_dir / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--criteria', type=Path, default=ROOT / 'configs/phase1_acceptance.json')
    parser.add_argument('--scenarios', type=Path, default=ROOT / 'configs/phase1_scenarios_v1.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import carla
    criteria = json.loads(args.criteria.read_text())
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30)
    if client.get_server_version() != '0.9.16':
        raise RuntimeError('Baseline requires CARLA 0.9.16')
    if args.mode == 'prepare':
        return prepare(args, criteria, client)
    if args.output is None:
        parser.error('--output is required for run')
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.scenarios.read_text())
    if frozen['criteria_sha256'] != digest(args.criteria):
        raise ValueError('Criteria changed after scenarios were frozen')
    world = client.load_world(criteria['map'])
    road_map = world.get_map()
    if hashlib.sha256(road_map.to_opendrive().encode()).hexdigest() != frozen['opendrive_sha256']:
        raise ValueError('Map differs from frozen scenario geometry')
    sys.path.insert(0, str(ROOT / 'artifacts/carla/PythonAPI/carla'))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    planner = GlobalRoutePlanner(road_map, 2.0)
    summary = {'task': 'phase1-rules-baseline', 'status': 'failed', 'ai_model': None,
               'started_utc': datetime.now(timezone.utc).isoformat(), 'routes': [],
               'criteria_sha256': digest(args.criteria), 'scenarios_sha256': digest(args.scenarios),
               'dependencies': {name: importlib.metadata.version(name) for name in ['carla', 'numpy', 'networkx', 'shapely']},
               'agent_sha256': digest(ROOT / 'artifacts/carla/PythonAPI/carla/agents/navigation/basic_agent.py')}
    summary['server_binary_sha256'] = digest(ROOT / 'artifacts/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe')
    summary['git_revision'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    write_json(args.output / 'criteria.json', criteria)
    write_json(args.output / 'scenarios.json', frozen)
    for route in frozen['routes']:
        report = run_case(world, road_map, planner, route, criteria, args.output / route['id'])
        summary['routes'].append(report)
        write_json(args.output / 'report.json', summary)
        print(json.dumps({'route': route['id'], 'status': report['status'], 'metrics': report['metrics'], 'errors': report['errors']}), flush=True)
    completed = sum(r['status'] == 'passed' for r in summary['routes'])
    summary['completed_routes'] = completed
    summary['completion_rate'] = completed / len(frozen['routes'])
    if completed == criteria['required_completed_routes']:
        summary['status'] = 'passed'
    write_json(args.output / 'report.json', summary)
    print(json.dumps({'task': summary['task'], 'status': summary['status'], 'completed': completed, 'total': len(frozen['routes'])}))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
