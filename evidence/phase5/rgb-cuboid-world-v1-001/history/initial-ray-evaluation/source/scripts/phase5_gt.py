"""Training/evaluation-only targets. NEVER imported by WorldFrame inference."""
import json
import hashlib
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def target_box(actor, reference_from_world):
    vertices = np.asarray(actor['world_vertices_m'], dtype=float)
    if vertices.shape != (8, 3) or not np.isfinite(vertices).all():
        raise ValueError('Eight finite world bbox vertices required')
    points = vertices @ reference_from_world[:3, :3].T+reference_from_world[:3, 3]
    size = 2*np.asarray(actor['extent_m'], dtype=float)
    # CARLA GetLocalVertices uses +/-x in bit2, +/-y in bit1, +/-z in bit0.
    axes = np.stack([points[4]-points[0], points[2]-points[0], points[1]-points[0]], axis=1)/size
    if not np.allclose(axes.T@axes, np.eye(3), atol=3e-3):
        raise ValueError('GT vertex ordering/extent differs from captured CARLA convention')
    yaw = float(np.arctan2(axes[1, 0], axes[0, 0]))
    return {'actor_id': actor['actor_id'], 'center_ego_m': points.mean(0).tolist(),
            'size_lwh_m': size.tolist(), 'yaw_rad': yaw, 'rotation_reference_from_box': axes.tolist()}


def audited_inventory(root, project_root):
    golden = read(project_root/'artifacts/runs/sunny-segmentation-v6-001/provenance.json')['data']
    for name, field in [('audit.json', 'audit_sha256'), ('visual_review.json', 'review_sha256'),
                        ('metadata_inventory.json', 'metadata_inventory_sha256'), ('image_inventory.jsonl', 'image_inventory_sha256')]:
        if sha(root/name) != golden[field]:
            raise ValueError('Frozen approved dataset changed: '+name)
    if read(root/'visual_review.json').get('training_approved') is not True:
        raise ValueError('Dataset has no completed training visual approval')
    inventory = {r['path']: r['sha256'] for r in read(root/'metadata_inventory.json')}
    inventory.update({r['path']: r['sha256'] for r in map(json.loads, (root/'image_inventory.jsonl').read_text(encoding='utf-8').splitlines())})
    return golden, inventory
