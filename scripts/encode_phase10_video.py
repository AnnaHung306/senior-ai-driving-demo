"""Encode actual browser screenshots as an explicitly recorded VP8/WebM replay."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    folder=ROOT/'artifacts/runs/phase10-demo-video-v1-001';capture=json.loads((folder/'capture.json').read_text(encoding='utf8'))
    target=folder/'phase10-recorded-replay.webm'
    if target.exists():raise FileExistsError(target)
    paths=[folder/x['screenshot'] for x in capture['frames']];sizes=[Image.open(p).size for p in paths]
    width=max(s[0] for s in sizes);height=max(s[1] for s in sizes);width+=width%2;height+=height%2
    writer=cv2.VideoWriter(str(target),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*'VP80'),5.,(width,height))
    if not writer.isOpened():raise RuntimeError('Existing VP8 encoder unavailable; no fake movie produced')
    try:
        for path in paths:
            rgb=np.array(Image.open(path).convert('RGB'));canvas=np.zeros((height,width,3),np.uint8);canvas[:rgb.shape[0],:rgb.shape[1]]=rgb
            writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
    finally:writer.release()
    reader=cv2.VideoCapture(str(target));count=0
    fps=reader.get(cv2.CAP_PROP_FPS)
    while True:
        ok,frame=reader.read()
        if not ok:break
        assert frame.shape==(height,width,3) and frame.std()>1;count+=1
    reader.release();assert count==len(paths)==67 and abs(fps-5)<1e-9
    report={'status':'encoded_and_decoded_actual_browser_replay_video','file':target.name,'sha256':sha(target),'bytes':target.stat().st_size,
        'codec':'VP8 in WebM via existing OpenCV/FFmpeg','frames':count,'fps':fps,'duration_s':count/fps,'width':width,'height':height,
        'source_run_id':capture['run_id'],'not_live_inference':True,'GT_geometry_in_main_view':False,
        'screenshots':[{'file':p.relative_to(folder).as_posix(),'sha256':sha(p)} for p in paths],
        'limits':'Nominal 5Hz simulation-time recorded playback, not wall-clock live inference; JPEG screenshot and lossy video are for presentation, not accuracy scoring.',
        'H264_probe_unavailable_no_package_installed':True,'source_sha256':sha(Path(__file__))}
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    (ROOT/'artifacts/reports/phase10-demo-video-final.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print({k:v for k,v in report.items() if k!='screenshots'})

if __name__=='__main__':main()
