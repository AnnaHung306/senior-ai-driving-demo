"""Classical map/RGB hybrid planning. Pure numeric; no CARLA/GT actor API."""
import math
import numpy as np


def finite(value, shape=None):
    x = np.asarray(value, dtype=float)
    if not np.isfinite(x).all() or (shape is not None and x.shape != shape):
        raise ValueError('Invalid finite geometry')
    return x


def validate_route(points, max_length=600.):
    p = finite(points)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise ValueError('Route needs at least three XY points')
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if (lengths > 4.).any() or lengths.sum() < 8 or lengths.sum() > max_length:
        raise ValueError('Disconnected, too short or too long route')
    return p

def validate_maneuvers(options):
    if not isinstance(options,list) or not options or any(o not in {'LEFT','RIGHT','STRAIGHT','LANEFOLLOW'} for o in options):
        raise ValueError('此版本僅驗證循線及轉彎；不支援換道路線')
    return True


def ego_coordinates(points, pose):
    x, y, yaw = finite(pose, (3,))
    c, s = math.cos(yaw), math.sin(yaw)
    return (finite(points)-[x, y]) @ np.array([[c, -s], [s, c]])


class RouteCursor:
    def __init__(self, points, max_length=600.):
        self.points = validate_route(points, max_length)
        self.distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        self.progress = 0.

    def update(self, pose):
        xy = finite(pose, (3,))[:2]
        a, b = self.points[:-1], self.points[1:]
        delta = b-a
        t = np.clip(((xy-a)*delta).sum(1)/np.maximum((delta*delta).sum(1), 1e-9), 0, 1)
        projected = a+t[:, None]*delta
        distances = np.linalg.norm(projected-xy, axis=1)
        progress = self.distance[:-1]+t*np.linalg.norm(delta, axis=1)
        valid = (progress >= self.progress-3.) & (progress <= self.progress+20.)
        if not valid.any():
            raise ValueError('Lost route progress')
        k = int(np.argmin(np.where(valid, distances, np.inf)))
        self.progress = max(self.progress, float(progress[k]))
        selected = (self.distance >= self.progress) & (self.distance <= self.progress+16.)
        local = ego_coordinates(self.points[selected], pose)
        if len(local) < 2:
            local = ego_coordinates(self.points[-3:], pose)
        return {'local_xy_m': local.tolist(), 'remaining_m': float(self.distance[-1]-self.progress),
                'cross_track_m': float(distances[k]), 'progress_m': self.progress,
                'total_m': float(self.distance[-1]), 'goal_distance_m': float(np.linalg.norm(xy-self.points[-1]))}


class Planner:
    def __init__(self, config):
        self.c = config
        self.last_time = None
        self.green_count = 0
        self.signal_latched = False

    def plan(self, route, observation, speed, now, dt=.1):
        c = self.c
        speed, now = float(speed), float(now)
        if not math.isfinite(speed+now+dt) or speed < 0 or dt <= 0:
            raise ValueError('Invalid planner time or speed')
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Nonmonotonic planner timestamp')
        self.last_time = now
        path = finite(route['local_xy_m'])
        result = {'state': 'CRUISE', 'target_speed_mps': c['cruise_mps'], 'local_path_xy_m': path.tolist(),
                  'reason': 'map_route_with_RGB_road_and_object_checks', 'map_prior_used': True,
                  'actor_GT_used': False, 'selected_offset_m': 0., 'nearest_obstacle_m': None}
        def stop(state, reason):
            result.update(state=state, target_speed_mps=0., reason=reason)
            return result
        if not observation.get('valid') or now-observation.get('timestamp_s', -1e9) > c['max_image_age_s'] or observation.get('timestamp_s', now+1)>now+1e-6:
            return stop('FAILSAFE', 'missing_stale_or_future_RGB')
        if route['cross_track_m'] > c['maximum_cross_track_m']:
            return stop('FAILSAFE', 'route_deviation')
        if route['goal_distance_m'] <= c['arrival_radius_m']:
            return stop('ARRIVED' if speed <= c['arrival_speed_mps'] else 'GOAL_BRAKE', 'goal_radius')
        result['target_speed_mps'] = min(result['target_speed_mps'], math.sqrt(max(0., 2*c['brake_deceleration_mps2']*(route['remaining_m']-1.8))))
        signals = observation.get('signals', [])
        red = any(s['phase'] in ['red','yellow','unknown','off','unknown_or_unobservable'] and s['score'] >= c['minimum_signal_score'] for s in signals)
        green = any(s['phase']=='green' and s['score'] >= c['minimum_signal_score'] for s in signals)
        if red:
            self.signal_latched = True
            self.green_count = 0
        elif green:
            self.green_count += 1
            if self.green_count >= c['signal_green_frames']:
                self.signal_latched = False
        else:
            self.green_count = 0
        if self.signal_latched:
            return stop('SIGNAL_STOP', 'RGB_signal_red_yellow_or_unknown_until_stable_green')
        candidates = observation.get('candidate_road_fractions', {})
        feasible = []
        for offset in c['lateral_offsets_m']:
            candidate = path.copy()
            # Smooth shift, zero at ego; same-lane only, not an unverified lane change.
            candidate[:,1] += offset*np.clip(candidate[:,0]/6., 0, 1)
            if float(candidates.get(str(offset), 0.)) < c['minimum_road_fraction']:
                continue
            nearest = float('inf')
            # Coasting boxes are precautionary planning constraints, never
            # relabelled as current measurements or inserted into occupancy.
            held=[o for o in observation.get('held_objects',[]) if o.get('age_since_measurement_s',99)<=.65]
            for o in observation.get('objects', [])+held:
                center = finite(o['center_ego_m'], (3,))[:2]
                size = finite(o['size_lwh_m'], (3,))
                if (size<=0).any():raise ValueError('Invalid object dimensions')
                if center[0] < -2. or center[0] > 25.:continue
                lateral = float(np.interp(center[0], candidate[:,0], candidate[:,1])) if len(candidate)>1 else 0.
                if abs(center[1]-lateral) < size[1]/2+c['vehicle_half_width_m']+c['obstacle_margin_m']:
                    nearest = min(nearest, float(center[0]-size[0]/2-c['vehicle_front_m']))
                f = o.get('forecast', {})
                if f.get('status')=='available':
                    future = finite(f['positions_ego_xy_m'])
                    times = finite(f['times_s'])
                    if future.shape != (len(times),2):raise ValueError('Invalid future trajectory')
                    relative = future.copy();relative[:,0]-=speed*times
                    if ((abs(relative[:,1])<size[1]/2+1.2)&(relative[:,0]>-2)&(relative[:,0]<4.)).any():
                        nearest=min(nearest, max(0.,float(center[0]-size[0]/2-c['vehicle_front_m'])))
            clearance = nearest
            # Prefer center. A side option is useful only if it materially clears obstruction.
            feasible.append((min(clearance,30.)-abs(offset)*2, offset, candidate, clearance))
        if not feasible:
            # Occupying objects can make the road segmentation unavailable.
            # Preserve the full-stop action but report the observed obstruction
            # instead of losing its identity behind a generic unknown-road state.
            ahead=[]
            held=[o for o in observation.get('held_objects',[]) if o.get('age_since_measurement_s',99)<=.65]
            for o in observation.get('objects',[])+held:
                center=finite(o['center_ego_m'],(3,));size=finite(o['size_lwh_m'],(3,))
                lateral=float(np.interp(center[0],path[:,0],path[:,1]))
                if 0<center[0]<15 and abs(center[1]-lateral)<size[1]/2+c['vehicle_half_width_m']+c['obstacle_margin_m']:
                    ahead.append(float(center[0]-size[0]/2-c['vehicle_front_m']))
            if ahead:
                result['nearest_obstacle_m']=min(ahead)
                return stop('OBSTACLE_STOP','predicted_object_ahead_and_RGB_corridor_unavailable')
            return stop('FAILSAFE', 'RGB_road_unknown_or_no_same_lane_candidate')
        _, offset, chosen, clearance = max(feasible, key=lambda r:r[0])
        result.update(local_path_xy_m=chosen.tolist(), selected_offset_m=offset,
                      nearest_obstacle_m=None if not math.isfinite(clearance) else clearance)
        if math.isfinite(clearance):
            if clearance <= 2.:
                return stop('OBSTACLE_STOP', 'predicted_object_in_swept_corridor')
            safe = max(0., (clearance-2.)/c['safe_time_headway_s'])
            result['target_speed_mps'] = min(result['target_speed_mps'],safe)
            result.update(state='FOLLOW',reason='predicted_clearance_time_headway')
        if abs(offset)>.01:
            result.update(state='YIELD_OFFSET',reason='RGB_supported_same_lane_lateral_clearance')
        # Reduce speed on tight turns without relying on future actor truth.
        ahead = chosen[chosen[:,0]>1.]
        if len(ahead):
            theta = abs(math.atan2(ahead[-1,1],ahead[-1,0]))
            result['target_speed_mps'] = min(result['target_speed_mps'],3./(1+theta*1.5))
        return result


class Controller:
    def __init__(self):
        self.integral = 0.
        self.steer = 0.

    def step(self, plan, speed, dt):
        speed,dt = float(speed),float(dt)
        if not math.isfinite(speed+dt) or dt<=0 or speed<0:raise ValueError('Invalid control feedback')
        target=float(plan['target_speed_mps'])
        if not math.isfinite(target) or target<0 or target>5.:raise ValueError('Invalid target speed')
        if target<=.01:
            self.integral=0.
            return {'throttle':0.,'brake':1.,'steer':self.steer}
        p=finite(plan['local_path_xy_m']);ahead=p[p[:,0]>.5]
        if not len(ahead):return {'throttle':0.,'brake':1.,'steer':self.steer}
        lookahead=max(3., min(6.,2.+speed))
        goal=ahead[int(np.argmin(abs(np.linalg.norm(ahead,axis=1)-lookahead)))]
        angle=math.atan2(2*2.85*goal[1], max(.1,float(goal@goal)))
        desired=float(np.clip(angle/.60,-.8,.8))
        self.steer+=float(np.clip(desired-self.steer,-dt*1.2,dt*1.2))
        error=target-speed
        self.integral=float(np.clip(self.integral+error*dt,-2,2))
        effort=.65*error+.08*self.integral
        return {'throttle':float(np.clip(effort,0,.55)), 'brake':float(np.clip(-effort,0,1)), 'steer':self.steer}
