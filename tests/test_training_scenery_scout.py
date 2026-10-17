"""Train-only boundary and native scalar serialization without CARLA RPC."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scout_training_scenery import training_town,object_record


class TrainingSceneryScoutTests(unittest.TestCase):
    def test_development_or_validation_town_cannot_be_used_as_training_scout(self):
        plan={'scenes':[{'town':'Town01','split':'train'},{'town':'Town03','split':'train'},
                        {'town':'Town04','split':'validation'},{'town':'Town05','split':'development'},{'town':'Town02','split':'lockbox'}]}
        training_town('Town01',plan);training_town('Town03',plan)
        for town in ['Town02','Town04','Town05']:
            with self.assertRaises(ValueError):training_town(town,plan)

    def test_native_geometry_serializes_without_inventing_world_bbox_conversion(self):
        import carla
        transform=carla.Transform(carla.Location(1,2,3),carla.Rotation(4,5,6))
        box=carla.BoundingBox(carla.Location(7,8,9),carla.Vector3D(2,3,4))
        obj=SimpleNamespace(id=123,name='SM_WoodFence',type=carla.CityObjectLabel.Fences,transform=transform,bounding_box=box)
        record=object_record(obj)
        self.assertEqual(json.loads(json.dumps(record)),record)
        self.assertEqual(record['bounding_box_as_returned']['location'],[7.,8.,9.])
        self.assertIn('wood',record['name_hints_not_material_GT'])
        self.assertNotIn('world_camera_pose',record)


if __name__=='__main__':unittest.main()
