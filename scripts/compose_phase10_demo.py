"""Readable 16:9 composition of actual captured UI panels; no synthetic scene or GT."""
import hashlib,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    folder=ROOT/'artifacts/runs/phase10-demo-video-v1-001';capture=read(folder/'capture.json');crops=read(folder/'crops.json')
    target=folder/'phase10-focus-replay.webm'
    if target.exists():raise FileExistsError(target)
    writer=cv2.VideoWriter(str(target),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*'VP80'),5.,(1280,720))
    if not writer.isOpened():raise RuntimeError('VP8 encoder unavailable')
    title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    report=read(ROOT/'artifacts/runs'/capture['run_id']/'report.json');records=[read(ROOT/'artifacts/runs'/capture['run_id']/x['file']) for x in report['records']]
    def panel(source,selector,width):
        r=crops[selector];a=source.crop((math.floor(r['x']),math.floor(r['y']),math.ceil(r['x']+r['width']),math.ceil(r['y']+r['height'])))
        return a.resize((width,round(a.height*width/a.width)),Image.Resampling.LANCZOS)
    try:
        for item,r in zip(capture['frames'],records):
            assert item['frame']==r['frame'];source=Image.open(folder/item['screenshot']).convert('RGB')
            page=Image.new('RGB',(1280,720),'#0b1118');draw=ImageDraw.Draw(page)
            draw.text((16,12),'RECORDED CARLA REPLAY | 8 RGB > AI > PLANNING > CONTROL',font=title,fill='#dcece8')
            draw.text((16,45),'5 Hz simulation-time playback. Not live inference. Sunny controlled simulator scene.',font=small,fill='#e5bb83')
            draw.text((12,76),'AI road, objects and planned path',font=font,fill='#91d8c6');draw.text((650,76),'Original front RGB - same captured frame',font=font,fill='#91d8c6')
            page.paste(panel(source,'#viewport',640),(0,100));page.paste(panel(source,'#rgb',640),(640,100))
            page.paste(panel(source,'.stats',640),(0,484));page.paste(panel(source,'#objectSummary',640),(0,582))
            page.paste(panel(source,'#map',250),(654,480))
            lines=['Map / ego pose: explicit priors','Neural geometry only in AI view','Class and box errors remain','GT used only for separate audit','No vehicle controlled by this video']
            for index,text in enumerate(lines):draw.text((930,496+index*29),text,font=small,fill='#aebdca')
            draw.text((16,628),f'Frame {r["frame"]} | simulation t={r["simulation_s"]:.1f}s | pedestrian crossing test',font=font,fill='#dcece8')
            draw.text((16,659),'Actual browser panels cropped for presentation. Evaluation uses original PNG/JSON, not this lossy video.',font=small,fill='#aebdca')
            writer.write(cv2.cvtColor(np.array(page),cv2.COLOR_RGB2BGR))
    finally:writer.release()
    reader=cv2.VideoCapture(str(target));count=0
    while True:
        ok,frame=reader.read()
        if not ok:break
        assert frame.shape==(720,1280,3);count+=1
    reader.release();assert count==67
    result={'status':'encoded_and_decoded_readable_16_9_actual_panel_replay','file':target.name,'sha256':sha(target),'bytes':target.stat().st_size,
        'frames':count,'fps':5,'duration_s':13.4,'width':1280,'height':720,'not_live_inference':True,'GT_geometry_in_main_view':False,
        'capture_manifest_sha256':sha(folder/'capture.json'),'crop_rects_sha256':sha(folder/'crops.json'),
        'upstream_full_page_video_report_sha256':sha(folder/'report.json'),'source_sha256':sha(Path(__file__)),
        'composition':'Only real captured viewport, RGB, stats, class-summary and map panels plus explicit playback annotations; no invented model outputs.'}
    (folder/'focus-report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    (ROOT/'artifacts/reports/phase10-demo-video-focus-final.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(result)

if __name__=='__main__':main()
