"""Hash the actual RGB-only bundle before inference/timing; no labels are read."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.dataset import read_sensor_frame,safe_path


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fingerprint_rgb_bundle(root,manifest):
    frame=read_sensor_frame(root,manifest)
    return {'input_manifest_sha256':sha(safe_path(root,manifest)),
            'input_RGB_sha256':{name:sha(safe_path(root,camera['image'])) for name,camera in frame['cameras'].items()},
            'input_fingerprint_source_sha256':sha(Path(__file__))}
