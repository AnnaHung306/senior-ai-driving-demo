import sys
from pathlib import Path
import unittest
import numpy as np
from itertools import product

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_sunny_dataset import lamp_targets


class LampGeometryTests(unittest.TestCase):
    def test_semantic_depth_association_not_trigger_volume(self):
        sem=np.zeros((50,50),dtype=np.uint8); sem[22:28,22:28]=7
        depth=np.full((50,50),10.)
        camera={'K':[[50,0,25],[0,50,25],[0,0,1]],'T_world_from_camera':np.eye(4)}
        lamp={'actor_id':123,'lamp_index':0,'extent_m':[.2,1,1],
              'world_pose':{'position_m':[10,0,0],'rotation_deg':[0,0,0]},
              'world_vertices_m':[[10+x,y,z] for x,y,z in product([-.2,.2],[-1,1],[-1,1])]}
        positives,claimed=lamp_targets(sem,depth,camera,[lamp])
        self.assertEqual(len(positives),1); self.assertEqual(int(claimed.sum()),36)
        self.assertEqual(positives[0]['visible_bbox_xyxy'],[22,22,28,28])
        positives,claimed=lamp_targets(sem,depth+5,camera,[lamp])
        self.assertEqual(len(positives),0); self.assertFalse(claimed.any())
        sem[22:28,22:28]=5  # pole/trigger actor tag is not lamp semantic 7
        positives,claimed=lamp_targets(sem,depth,camera,[lamp])
        self.assertEqual(len(positives),0)

if __name__=='__main__': unittest.main()
