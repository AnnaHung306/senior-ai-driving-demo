"""Supplement guards: no train access, no invented groups, real pose round trips."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from train_light_state import LightCrops
from collect_light_supplement import camera_transform


class SupplementTests(unittest.TestCase):
    def test_heldout_only_and_original_groups_count_once(self):
        config={'minimum_samples_per_class':{'lockbox':2},'minimum_groups_per_class':2}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs/test-temp') as folder:
            roots=[Path(folder)/name for name in ['main','extra']]
            for root in roots:
                (root/'training').mkdir(parents=True)
                Image.fromarray(np.zeros((360,640,3),dtype=np.uint8)).save(root/'rgb.png')
                labels={'positive_objects':[{'category_id':7,'light_state_class':c,'visible_bbox_xyxy':[0,0,8,16],'world_phase':'Off'} for c in range(4)]}
                (root/'labels.json').write_text(json.dumps(labels),encoding='utf-8')
                (root/'training/lockbox.jsonl').write_text(json.dumps({'split':'lockbox','group_id':'same-original-route','rgb':'rgb.png','detection':'labels.json'})+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'heldout only'):LightCrops(roots[0],'train',config,supplement_root=roots[1])
            with self.assertRaisesRegex(ValueError,'Lockbox forbidden'):LightCrops(roots[0],'lockbox',config,supplement_root=roots[1])
            data=LightCrops(roots[0],'lockbox',config,allow_lockbox=True,supplement_root=roots[1])
            self.assertEqual(data.counts,[2,2,2,2]);self.assertFalse(data.approved)
            self.assertTrue(all(v['groups']==1 for v in data.coverage.values()))

    def test_actual_frozen_camera_matrix_roundtrips(self):
        import carla
        plan=json.loads((ROOT/'configs/light_lockbox_supplement_v1.json').read_text(encoding='utf-8'))
        matrices=[p['T_world_from_camera'] for p in plan['poses']]
        matrices.append(carla.Transform(carla.Location(3,4,5),carla.Rotation(13,147,-7)).get_matrix())
        for matrix in matrices:
            self.assertLess(np.abs(np.asarray(camera_transform(matrix).get_matrix())-matrix).max(),.0001)


if __name__=='__main__':unittest.main()
