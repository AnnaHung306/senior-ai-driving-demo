import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.detection_metrics import evaluate_detection


class DetectionMetricsTests(unittest.TestCase):
    def image(self):
        return {'targets':[{'label':1,'box':[10,10,20,20]}], 'predictions':[{'label':1,'box':[10,10,20,20],'score':.9}], 'ignore_boxes':[]}

    def test_perfect_and_missing_class(self):
        r=evaluate_detection([self.image()],{'car':1,'bus':2})
        self.assertEqual(r['mAP50_95'],1.)
        self.assertIsNone(r['per_class']['bus']['AP50'])
        self.assertFalse(r['all_classes_present'])

    def test_duplicate_false_positive_and_ignore(self):
        i=self.image(); i['predictions']=[{'label':1,'box':[30,30,40,40],'score':.99}]+i['predictions']*2
        r=evaluate_detection([i],{'car':1})['per_class']['car']
        self.assertEqual(r['false_positives'],2); self.assertEqual(r['AP50'],.5)
        i['ignore_boxes']=[[30,30,40,40]]
        r=evaluate_detection([i],{'car':1})['per_class']['car']
        self.assertEqual(r['false_positives'],1); self.assertEqual(r['AP50'],1.)
        i['ignore_boxes'].append([10,10,20,20])
        r=evaluate_detection([i],{'car':1})['per_class']['car']
        self.assertEqual(r['false_positives'],1)  # duplicate positive cannot hide in ignore

    def test_wrong_class_and_iou_threshold(self):
        i=self.image(); i['predictions'][0]['label']=2
        r=evaluate_detection([i],{'car':1})['per_class']['car']
        self.assertEqual(r['AP50'],0.)
        i=self.image(); i['predictions'][0]['box']=[10,10,16,20]
        r=evaluate_detection([i],{'car':1})['per_class']['car']
        self.assertEqual(r['AP50'],1.); self.assertLess(r['AP50_95'],1.)

    def test_size_and_truncation_recall_use_fixed_score(self):
        i=self.image();i['targets'][0]['truncated']=True;i['predictions'][0]['score']=.4
        r=evaluate_detection([i],{'car':1})['per_class']['car']
        self.assertEqual(r['AP50'],1.)
        self.assertEqual(r['recall_strata_IoU50']['small_lt_32_squared']['recall'],0.)
        self.assertEqual(r['recall_strata_IoU50']['truncated']['ground_truth'],1)

if __name__=='__main__': unittest.main()
