import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.phase10_metrics import semantic_confusion,visible_targets,summary,binary_iou


class AuditMetricsTests(unittest.TestCase):
    def test_ignore_and_class_support_are_counted(self):
        m,ignored=semantic_confusion(np.array([[1,24,0,21]],np.uint8),np.array([[1,1,2,3]],np.uint8))
        self.assertEqual(ignored,2);self.assertEqual(m.sum(),2);self.assertEqual(m[10,1],1)
    def test_tight_visible_box_exclusive_and_not_amodal(self):
        a=np.zeros((20,20,3),np.uint8);a[2:8,3:10]=[14,1,0]
        targets,ignored,_=visible_targets(a,{1});self.assertEqual(targets[0]['box'],[3,2,10,8]);self.assertEqual(ignored,[])
    def test_unbound_instance_is_unknown_not_negative(self):
        a=np.zeros((20,20,3),np.uint8);a[2:8,3:10]=[15,9,0]
        t,i,d=visible_targets(a,{1});self.assertEqual(t,[]);self.assertEqual(len(i),1);self.assertIn('snapshot',d[0]['reason'])
    def test_empty_class_not_perfect_score(self):
        self.assertIsNone(binary_iou(np.zeros(2,bool),np.zeros(2,bool)));self.assertIsNone(summary([])['mean'])
    def test_tiny_object_ignore_keeps_pixels(self):
        a=np.zeros((20,20,3),np.uint8);a[2:4,3:5]=[12,1,0]
        t,i,d=visible_targets(a,{1});self.assertEqual(t,[]);self.assertEqual(d[0]['pixels'],4)
    def test_signal_render_instance_is_separate_from_actor_3D(self):
        a=np.zeros((20,20,3),np.uint8);a[2:8,3:10]=[7,6,0]
        t,_,_=visible_targets(a,set());self.assertIsNone(t[0]['actor_id']);self.assertEqual(t[0]['label'],7)

if __name__=='__main__':unittest.main()
