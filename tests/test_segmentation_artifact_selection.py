"""Artifact checker must respect the model's frozen selection policy."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_sunny_segmentation_artifacts import selected_epoch


class ArtifactSelectionTests(unittest.TestCase):
    def test_qualified_lower_mean_beats_failing_higher_mean(self):
        rows=[{'epoch':0,'validation_score':.8,'validation_quality':{'passed':False}},
              {'epoch':1,'validation_score':.6,'validation_quality':{'passed':True}}]
        self.assertEqual(selected_epoch(rows,'qualified_first_then_score')['epoch'],1)
        self.assertEqual(selected_epoch(rows,'score_only')['epoch'],0)

    def test_no_qualified_candidate_uses_mean_with_earliest_tie(self):
        rows=[{'epoch':i,'validation_score':score,'validation_quality':{'passed':False}} for i,score in enumerate([.4,.6,.6])]
        self.assertEqual(selected_epoch(rows,'qualified_first_then_score')['epoch'],1)

    def test_no_silent_policy_or_missing_qualification_fallback(self):
        rows=[{'epoch':0,'validation_score':.6}]
        with self.assertRaises(ValueError):selected_epoch(rows,'unknown')
        with self.assertRaises(ValueError):selected_epoch(rows,'qualified_first_then_score')
        self.assertEqual(selected_epoch(rows,'score_only')['epoch'],0)


if __name__=='__main__':unittest.main()
