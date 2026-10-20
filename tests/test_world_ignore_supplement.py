import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_phase5_errors import ignore_objects


def object_box(x=0):
    return {'class_id': 1, 'center_ego_m': [x, 0, 1], 'size_lwh_m': [4, 2, 2], 'yaw_rad': 0.,
            'sources': [{'camera': 'front', 'bbox_xyxy': [2, 2, 6, 6]}]}


class WorldIgnoreSupplementTests(unittest.TestCase):
    def labels(self):return {'front': {'ignore_regions': [{'bbox_xyxy': [0, 0, 20, 20]}], 'positive_objects': []}}

    def test_requires_all_camera_sources_ignored(self):
        o = object_box();v = self.labels();retained, ignored = ignore_objects([o], [], v)
        self.assertEqual(retained, []);self.assertEqual(ignored, [0])
        o['sources'].append({'camera': 'left', 'bbox_xyxy': [2, 2, 6, 6]});v['left'] = {'ignore_regions': [], 'positive_objects': []}
        self.assertEqual(ignore_objects([o], [], v)[1], [])

    def test_known_2D_or_3D_positive_overrides_ignore(self):
        o = object_box();v = self.labels();v['front']['positive_objects'] = [{'visible_bbox_xyxy': [2, 2, 6, 6]}]
        self.assertEqual(ignore_objects([o], [], v)[1], [])
        self.assertEqual(ignore_objects([o], [object_box()], self.labels())[1], [])

    def test_supplement_does_not_mutate_predictions_labels_or_truth(self):
        objects = [object_box()];truth = [object_box(30)];visible = self.labels();before = copy.deepcopy((objects, truth, visible))
        ignore_objects(objects, truth, visible)
        self.assertEqual((objects, truth, visible), before)


if __name__ == '__main__':unittest.main()
