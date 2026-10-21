"""Evaluation-only training ego trajectory overlap audit; never a driving input."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read,sha
from senior_ai.core import write_json
base=ROOT/'data/sunny-cross-town-v3-007';points=[];inventory={}
for folder in sorted((base/'scenes').glob('town01-*')):
    index=read(folder/'index.json')
    if not index or index[0]['split']!='train':continue
    for row in index:
        p=base/row['label'];value=read(p);points.append(value['ego_world_pose']['position_m'][:2]);inventory[p.relative_to(base).as_posix()]=sha(p)
points=np.array(points);rows=[]
for r in read(ROOT/'artifacts/runs/phase89-route-catalog-v1-001/catalog.json')['routes']:
    p=np.array(r['points_xy_m']);nearest=np.linalg.norm(p[:,None,:]-points[None,:,:],axis=-1).min(1)
    rows.append({'id':r['id'],'minimum_distance_from_captured_train_ego_m':float(nearest.min()),
                 'fraction_route_vertices_within_10m_training_trajectory':float((nearest<10).mean()),
                 'all_route_vertices_more_than_10m_from_captured_train_trajectory':bool((nearest>=10).all())})
write_json(ROOT/'artifacts/reports/phase89-route-overlap-001.json',{'status':'completed_evaluation_only_train_trajectory_audit','routes':rows,
    'training_ego_positions':len(points),'GT_metadata_inventory':inventory,'source_sha256':sha(Path(__file__)),
    'limits':'Only ego center trajectories from original007 Town01 train. Camera views can extend beyond 10m; other training variants/assets and scene appearance may overlap. Not heldout generalization certification.',
    'development_lockbox_opened':False})
print(rows)
