from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_phase5_world import occupancy_counts, occupancy_scores, match_objects, metrics_objects, regression_error, merge_audit


def box(x=0., category=1, yaw=0.):
    return {'center_ego_m': [x, 0., 1.], 'size_lwh_m': [4., 2., 2.], 'yaw_rad': yaw, 'class_id': category}


class WorldEvaluationTests(unittest.TestCase):
    def test_unknown_counts_FN_and_GT_domain_does_not_use_prediction(self):
        gt = np.array([1, 0, 255], np.uint8);p = np.array([255, 255, 1], np.uint8)
        c = occupancy_counts(p, gt);r = occupancy_scores(c)
        self.assertEqual(int(c.sum()), 2);self.assertEqual(r['by_class']['1']['FN'], 1)
        self.assertEqual(r['by_class']['1']['FP'], 0);self.assertEqual(r['coverage_on_GT_domain'], 0.)

    def test_false_free_counts_and_absent_class_not_assessed(self):
        c = occupancy_counts(np.array([0, 0, 3]), np.array([1, 3, 0]));r = occupancy_scores(c)
        self.assertEqual(r['GT_occupied_predicted_free'], 2);self.assertEqual(r['false_free_rate_on_GT_occupied'], 1.)
        self.assertIsNone(r['by_class']['8']['IoU'])

    def test_one_to_one_and_class_gated_matching(self):
        counts, _, pairs = metrics_objects([box(), box(.1), box(category=2)], [box()])
        self.assertEqual(counts['TP'], 1);self.assertEqual(counts['FP'], 2);self.assertEqual(len(pairs), 1)
        self.assertEqual(match_objects([box()], [box(10)]), [])

    def test_3D_IoU_gate_and_yaw_symmetry(self):
        self.assertEqual(len(match_objects([box()], [box(yaw=np.pi)], 'IoU')), 1)
        error = regression_error(box(yaw=np.pi), box());self.assertAlmostEqual(error['yaw_absolute_deg'], 180)
        self.assertAlmostEqual(error['yaw_pi_symmetric_deg'], 0)

    def test_merge_support_can_detect_different_actors(self):
        sources = [{'camera': n, 'bbox_xyxy': [0, 0, 10, 10]} for n in ['front', 'left']]
        objects = [{'class_id': 1, 'sources': sources}]
        targets = {n: [{'category_id': 1, 'visible_bbox_xyxy': [0, 0, 10, 10], 'actor_id': actor}] for n, actor in [('front', 1), ('left', 2)]}
        audit = merge_audit(objects, targets);self.assertEqual(audit['mixed_GT_actor_merges'], 1)
        targets['left'][0]['actor_id'] = 1;self.assertEqual(merge_audit(objects, targets)['GT_supported_merged_clusters'], 1)


if __name__ == '__main__':unittest.main()
