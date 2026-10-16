"""GT-assisted acquisition rules. This module is never model inference/control."""
import math
import numpy as np
from shapely.geometry import MultiPoint
from shapely.affinity import translate


def forecast_conflict(vertices_a,velocity_a,vertices_b,velocity_b):
    arrays=[np.asarray(vertices_a),np.asarray(velocity_a),np.asarray(vertices_b),np.asarray(velocity_b)]
    if arrays[0].shape!=(8,3) or arrays[2].shape!=(8,3) or arrays[1].shape!=(3,) or arrays[3].shape!=(3,):
        raise ValueError('Requires finite world OBB vertices and velocity vectors')
    if not all(np.isfinite(v).all() for v in arrays): raise ValueError('Nonfinite acquisition motion')
    a=MultiPoint(arrays[0][:,:2]).convex_hull.buffer(.4);b=MultiPoint(arrays[2][:,:2]).convex_hull.buffer(.4)
    va,vb=arrays[1],arrays[3]
    return any(translate(a,va[0]*dt,va[1]*dt).intersects(translate(b,vb[0]*dt,vb[1]*dt)) for dt in [0.,.5,1.,1.5,2.])


def tm_target_speed(speed_limit_kph,conflict):
    if not math.isfinite(speed_limit_kph) or speed_limit_kph<0: raise ValueError('Invalid speed limit')
    # CARLA 0.9.16 MotionPlanStage.cpp:159 divides by dynamic target velocity.
    # Some assets/positions have not received a client speed-limit trigger and
    # report zero. That is missing telemetry, not a 0.1kph cruising instruction.
    cruise=12. if speed_limit_kph<=.1 else min(12.,speed_limit_kph*.3)
    return .1 if conflict else max(.1,cruise)
