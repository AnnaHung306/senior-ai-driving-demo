from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_unused_scenery import select_anchors


class UnusedSceneryPreparationTests(unittest.TestCase):
    def spawns(self,count=40,distance=10):
        rows=[]
        for i in range(count):
            m=np.eye(4);m[0,3]=i*distance
            rows.append({'spawn_index':i,'native_transform_matrix':m.tolist()})
        return rows

    def wall(self,name,index,x):
        m=np.eye(4);m[:2,3]=[x,6]
        return {'native_name':name,'native_id':index,'native_transform_matrix':m.tolist(),'bounding_box_as_returned':{'extent_m':[2,1,1]}}

    def test_hints_interleaved_and_native_poses_preserved(self):
        spawns=self.spawns();walls=[self.wall('wood_example',1,25),self.wall('concrete_example',2,155)]
        anchors=select_anchors(spawns,walls,8)
        self.assertEqual({a['selection']['name_hint_not_material_GT'] for a in anchors[:2]},{'wood','concrete'})
        for a in anchors:self.assertEqual(a['native_transform_matrix'],spawns[a['spawn_index']]['native_transform_matrix'])
        self.assertEqual(len({a['spawn_index'] for a in anchors}),8)
        xy=np.asarray([a['native_transform_matrix'] for a in anchors])[:,:2,3]
        self.assertGreaterEqual(min(np.linalg.norm(xy[i+1:]-xy[i],axis=1).min() for i in range(7)),8)

    def test_impossible_geometry_refused(self):
        with self.assertRaises(ValueError):select_anchors(self.spawns(3),[],4)
        with self.assertRaises(ValueError):select_anchors(self.spawns(40,1),[],20)

    def test_nonfinite_native_poses_refused(self):
        spawns=self.spawns();spawns[0]['native_transform_matrix'][0][3]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):select_anchors(spawns,[],8)


if __name__=='__main__':unittest.main()
