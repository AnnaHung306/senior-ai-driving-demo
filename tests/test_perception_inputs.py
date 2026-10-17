"""Bundle provenance must distinguish RGB changes with the same frame number."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from perception_inputs import fingerprint_rgb_bundle


class PerceptionInputTests(unittest.TestCase):
    def setUp(self):
        self.parent=Path(__file__).resolve().parents[1]/'artifacts/runs/test-temp';self.parent.mkdir(parents=True,exist_ok=True)

    def fixture(self,root):
        (root/'inputs/rgb').mkdir(parents=True);(root/'inputs/frames').mkdir(parents=True)
        cameras={}
        for i in range(8):
            path=f'inputs/rgb/{i}.png';Image.new('RGB',(2,2),(i,0,0)).save(root/path)
            cameras[str(i)]={'image':path,'frame':123,'timestamp_s':1.}
        frame={'schema_version':1,'source':'carla_rgb_only','scene_id':'fixture','frame':123,'timestamp_s':1.,'cameras':cameras,'calibration':'calibration/not-opened.json'}
        path=root/'inputs/frames/frame.json';path.write_text(json.dumps(frame),encoding='utf-8');return frame,path

    def test_RGB_change_is_detected_without_changing_frame_or_manifest(self):
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            root=Path(folder);self.fixture(root)
            before=fingerprint_rgb_bundle(root,'inputs/frames/frame.json')
            Image.new('RGB',(2,2),(255,0,0)).save(root/'inputs/rgb/0.png')
            after=fingerprint_rgb_bundle(root,'inputs/frames/frame.json')
            self.assertEqual(before['input_manifest_sha256'],after['input_manifest_sha256'])
            self.assertNotEqual(before['input_RGB_sha256']['0'],after['input_RGB_sha256']['0'])
            self.assertEqual(len(after['input_RGB_sha256']),8)
            self.assertFalse((root/'calibration/not-opened.json').exists())

    def test_GT_fields_cannot_enter_fingerprinted_inference_manifest(self):
        with tempfile.TemporaryDirectory(dir=self.parent) as folder:
            root=Path(folder);frame,path=self.fixture(root);frame['labels']='labels/GT.json'
            path.write_text(json.dumps(frame),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'GT is forbidden'):fingerprint_rgb_bundle(root,'inputs/frames/frame.json')


if __name__=='__main__':unittest.main()
