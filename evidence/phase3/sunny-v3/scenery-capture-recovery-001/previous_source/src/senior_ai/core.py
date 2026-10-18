"""Small, dependency-free primitives shared by phase-0 verification."""
import json
import math
from pathlib import Path
from queue import Empty
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def receive_frame(queue, expected, timeout):
    """Discard stale samples; never silently associate a future frame with this tick."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Camera timeout at frame {expected}')
        try:
            sample = queue.get(timeout=remaining)
        except Empty as exc:
            raise TimeoutError(f'Camera timeout at frame {expected}') from exc
        if sample.frame < expected:
            continue
        if sample.frame > expected:
            raise ValueError(f'Expected frame {expected}, received future frame {sample.frame}')
        return sample


def intrinsics(width, height, fov):
    if width <= 0 or height <= 0 or not 0 < fov < 180:
        raise ValueError('Invalid camera geometry')
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    return [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]


def assess(metrics, criteria):
    checks = {
        'movement': metrics['distance_m'] >= criteria['min_distance_m'],
        'acceleration': metrics['max_speed_mps'] >= criteria['min_peak_speed_mps'],
        'stop': metrics['final_speed_mps'] <= criteria['max_final_speed_mps'],
        'images': metrics['saved_images'] >= criteria['min_saved_images'],
        'synchronization': metrics['max_timestamp_error_s'] <= criteria['max_timestamp_error_s'],
        'frame_count': metrics['observed_frames'] == criteria['expected_frames'],
        'collision_free': metrics['collision_count'] == 0,
        'cleanup': metrics['cleanup_ok'],
    }
    if not all(math.isfinite(metrics[k]) for k in
               ['distance_m', 'max_speed_mps', 'final_speed_mps', 'max_timestamp_error_s']):
        checks['finite_metrics'] = False
    return checks
