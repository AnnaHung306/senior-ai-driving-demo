"""Protect original data boundaries before native expansion inventory work."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scout_unused_training_town import town_family,validate_unused


class UnusedTownPolicyTests(unittest.TestCase):
    def setUp(self):
        self.plan={'scenes':[{'town':town,'split':split} for town,split in [('Town01','train'),('Town03','train'),('Town04','validation'),('Town05','development'),('Town02','lockbox')]],
            'maps':{town:{'opendrive_sha256':'hash-'+town} for town in ['Town01','Town02','Town03','Town04','Town05']}}
        self.policy={'task':'segmentation_training_expansion_scout_only','training_approved':False,'candidate_training_towns':['Town06','Town07']}

    def test_explicit_unused_full_towns_allowed(self):
        self.assertEqual(validate_unused('Town06',self.plan,self.policy,'new-map-hash'),'Town06')
        self.assertEqual(validate_unused('Town07',self.plan,self.policy),'Town07')

    def test_original_splits_variants_and_unreserved_towns_rejected(self):
        for name in ['Town01','Town02','Town03','Town04','Town05','Town04_Opt','Town06_Opt','Town10HD','/Game/Carla/Maps/Town04','town06']:
            with self.subTest(name=name),self.assertRaises(ValueError):validate_unused(name,self.plan,self.policy)

    def test_native_opendrive_alias_rejected_for_every_original_map(self):
        for entry in self.plan['maps'].values():
            with self.assertRaises(ValueError):validate_unused('Town06',self.plan,self.policy,entry['opendrive_sha256'])

    def test_policy_cannot_override_original_split_or_grant_training_approval(self):
        self.policy['candidate_training_towns'].append('Town04')
        with self.assertRaises(ValueError):validate_unused('Town04',self.plan,self.policy)
        self.policy['training_approved']=True
        with self.assertRaises(ValueError):validate_unused('Town06',self.plan,self.policy)

    def test_added_original_split_blocks_candidate_and_family_normalizes_alias(self):
        self.plan['scenes'].append({'town':'/Game/Carla/Maps/Town06_Opt','split':'validation'})
        with self.assertRaises(ValueError):validate_unused('Town06',self.plan,self.policy)
        self.assertEqual(town_family('/Game/Carla/Maps/Town04_Opt'),'Town04')


if __name__=='__main__':unittest.main()
