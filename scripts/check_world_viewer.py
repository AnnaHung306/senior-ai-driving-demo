"""Independently compare every display frame to frozen Phase5/4 outputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from senior_ai.world_display import read, sha, strict_world
from senior_ai.world_viewer_service import ViewerBundle
from senior_ai.cuboid import corners
from senior_ai.bev_geometry import Grid, sample


def main(a):
    bundle = ViewerBundle(a.bundle);state = read(a.run/'state.json');rows = [];objects = 0
    config = read(a.phase4/'config.json');g = Grid(**config['grid']);paint = Grid(g.x_min,g.x_max,g.y_min,g.y_max,.2)
    for source, row in zip(state['records'], bundle.manifest['frames'], strict=True):
        folder = Path(source['folder']);world = read(a.run/folder/'world.json');frame = read(bundle.file(row['file']))
        with np.load(a.run/folder/'occupancy.npz', allow_pickle=False) as n:occ = n['semantic']
        strict_world(world, occ)
        assert frame['world_source_sha256'] == sha(a.run/folder/'world.json')
        assert frame['parent_bev_sha256'] == sha(a.phase4/folder/'bev.npz')
        assert frame['timestamp_s'] == world['timestamp_s'] and frame['frame'] == world['frame'] and frame['scene_id'] == world['scene_id']
        assert bundle.file(frame['occupancy']['file']).read_bytes() == occ.tobytes(order='C')
        z = int(np.floor(-world['occupancy_grid']['z_min']/world['occupancy_grid']['resolution_m']))
        ground = occ[...,z].copy();ground[~np.isin(ground,[1,2,9])] = 255
        np.testing.assert_array_equal(ground.ravel(),frame['ground']['labels_row_major'])
        assert frame['occupancy_unknown_fraction'] == float((occ == 255).mean())
        assert frame['local_path'] == world['road_path']
        # Each semantic interface must touch exactly road and sidewalk in source cells.
        grid = frame['ground']['grid'];r = grid['resolution_m']
        for p,q in frame['road_sidewalk_interfaces']['segments_ego_xy_m']:
            x,y = (np.array(p)+q)/2
            offsets = [(0,-r/4),(0,r/4)] if p[1] == q[1] else [(-r/4,0),(r/4,0)]
            labels = []
            for dx,dy in offsets:
                rr = int(np.floor((grid['x_max']-x-dx)/r));cc = int(np.floor((y+dy-grid['y_min'])/r));labels.append(ground[rr,cc])
            assert set(labels) == {1,2}
        with np.load(a.phase4/folder/'bev.npz', allow_pickle=False) as n:
            uv = g.xy_to_uv(paint.points()[...,:2]).astype(np.float32);p = sample(n['temporal'],uv);seen = sample(n['observed'].astype(np.uint8),uv,True).astype(bool)
            expected = paint.points()[...,:2][seen & (p[...,1]>=.25) & (p[...,0]>=.4)]
        np.testing.assert_array_equal(expected,np.array(frame['lane_paint']['centers_ego_xy_m']))
        expected_objects = []
        for o,w in zip(frame['objects'],world['objects'],strict=True):
            for field in ['class_id','center_ego_m','size_lwh_m','yaw_rad','track_id','score_uncalibrated','track_age_measurements']:assert o[field] == w[field]
            assert not set(o)&{'actor_id','world_vertices_m','sources','prior_baseline_objects'}
            c = corners(w);expected_objects.append({'track_id':w['track_id'],'corners_three':c[:,[1,2,0]]*np.array([1,1,-1])})
            expected_objects[-1]['corners_three'] = expected_objects[-1]['corners_three'].tolist();objects += 1
        rows.append({'key':row['key'],'file':row['file'],'objects':expected_objects})
    report = {'status':'passed_all_source_geometry_and_semantics','frames':len(rows),'objects':objects,
              'bundle_sha256':sha(a.bundle/'bundle.json'),'verified_files':len(bundle.inventory),'geometry_for_node':rows,
              'limits':'Correct transfer and geometry only; model quality remains false.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='geometry_for_node'}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/runs/phase6-viewer-bundle-002')
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/runs/phase5-world-v1-002');p.add_argument('--phase4',type=Path,default=ROOT/'artifacts/runs/phase4-bev-pilot-002')
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
