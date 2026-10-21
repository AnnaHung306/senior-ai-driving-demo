import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.bev_geometry import Grid
from senior_ai.bev_fusion import fuse_road
from senior_ai.road_fusion_cache import CachedRoadFusion
from test_bev_phase4 import camera

class FusionCacheTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/phase4_bev_v1.json').read_text());self.grid=Grid(0,12,-6,6,.5)
        self.cameras={'front':camera(name='front'),'rear':camera(180,name='rear')};rng=np.random.default_rng(10)
        self.observations={name:{'ids':rng.choice(np.array([0,1,2,10,16,3],np.uint8),(360,640)),
            'ground_probabilities':rng.random((360,640,2),dtype=np.float32)} for name in self.cameras}
    def test_all_fusion_arrays_equal_across_different_surface_masks(self):
        cached=CachedRoadFusion(self.cameras,self.grid,self.config)
        for kind in [None,0,1,10]:
            if kind is not None:
                for ob in self.observations.values():ob['ids'].fill(kind)
            before=fuse_road(self.observations,self.cameras,self.grid,self.config);after=cached(self.observations)
            for key in ['probabilities','observed','support','geometric_coverage']:np.testing.assert_array_equal(before[key],after[key])
    def test_camera_set_change_requires_new_cache(self):
        cached=CachedRoadFusion(self.cameras,self.grid,self.config)
        with self.assertRaises(ValueError):cached({'front':self.observations['front']})
    def test_new_rig_projections_do_not_reuse_old_maps(self):
        first=CachedRoadFusion({'front':camera(0,name='front')},self.grid,self.config)
        second=CachedRoadFusion({'front':camera(90,name='front')},self.grid,self.config)
        self.assertFalse(np.array_equal(first.maps['front'][1],second.maps['front'][1]))

if __name__=='__main__':unittest.main()
