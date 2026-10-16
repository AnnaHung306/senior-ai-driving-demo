import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.segmentation_model import TinyUNet
from senior_ai.sunny_data import SEGMENTATION_NAMES


class SunnyInferenceTests(unittest.TestCase):
    def test_rgb_only_without_gt_or_training_indexes(self):
        temp_root=ROOT/'artifacts/runs/test-temp';temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root=Path(directory);(root/'inputs/rgb').mkdir(parents=True);(root/'inputs/frames').mkdir()
            Image.fromarray(np.full((360,640,3),128,dtype=np.uint8)).save(root/'inputs/rgb/view.png')
            names=['front','front_right','right','rear_right','rear','rear_left','left','front_left']
            frame={'schema_version':1,'source':'carla_rgb_only','scene_id':'fixture','frame':1,'timestamp_s':.05,
                'calibration':'calibration/rig.json','cameras':{name:{'image':'inputs/rgb/view.png','frame':1,'timestamp_s':.05} for name in names}}
            (root/'inputs/frames/frame.json').write_text(json.dumps(frame),encoding='utf-8')
            checkpoint=root/'test-untrained.pt'
            torch.save({'task':'segmentation','config':{'segmentation':{'width':4}},'epoch':0,
                        'model':TinyUNet(width=4,classes=len(SEGMENTATION_NAMES)).state_dict()},checkpoint)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_sunny.py'),'--checkpoint',str(checkpoint),
                '--root',str(root),'--manifest','inputs/frames/frame.json','--output',str(root/'prediction'),'--device','cpu'],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads((root/'prediction/prediction.json').read_text(encoding='utf-8'))
            self.assertTrue(receipt['RGB_only']);self.assertEqual(receipt['source'],'learned_prediction')
            self.assertFalse((root/'labels').exists());self.assertFalse((root/'training').exists())
            with Image.open(root/'prediction/class_ids.png') as image:
                self.assertEqual(image.size,(640,360))

if __name__=='__main__':unittest.main()
