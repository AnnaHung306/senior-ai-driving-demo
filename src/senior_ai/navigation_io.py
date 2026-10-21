"""Lightweight replay I/O; deliberately does not import Torch or CARLA."""
import hashlib,json
from pathlib import Path

def read(path):
    def reject(value):raise ValueError('Nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(encoding='utf8'),parse_constant=reject)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
