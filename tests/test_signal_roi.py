import sys,unittest
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.live_perception import track_signal_rois
class SignalROITests(unittest.TestCase):
    def test_flow_follows_previously_detected_signal(self):
        image=np.zeros((360,640),np.uint8);cv2.rectangle(image,(100,100),(114,140),180,-1);cv2.circle(image,(107,110),3,255,-1)
        shifted=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,2]]),(640,360))
        old={'bbox_xyxy':[100,100,114,140],'last_detected_s':1.,'phase':'red','score':.9}
        result=track_signal_rois(image,shifted,[old],1.2);self.assertEqual(len(result),1)
        np.testing.assert_allclose(result[0]['bbox_xyxy'],[104,102,118,142],atol=.15)
        self.assertEqual(old['bbox_xyxy'],[100,100,114,140])
    def test_expired_or_missing_images_never_make_ROI(self):
        image=np.zeros((360,640),np.uint8);old={'bbox_xyxy':[100,100,114,140],'last_detected_s':1.}
        self.assertEqual(track_signal_rois(None,image,[old],2),[]);self.assertEqual(track_signal_rois(image,image,[old],32),[])
    def test_textureless_tracking_not_trusted(self):
        image=np.zeros((360,640),np.uint8);old={'bbox_xyxy':[100,100,114,140],'last_detected_s':1.}
        self.assertEqual(track_signal_rois(image,image,[old],2),[])
