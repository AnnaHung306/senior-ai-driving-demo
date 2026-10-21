"""Make the 16:9 presentation from actual CUA browser screenshots and DOM receipts.

This is video composition, not new model rendering: RGB and 3D crops are from
the same verified browser frame. No interpolation, relabeling, skipped source
frames or GT overlays. Capture manifest is generated through browser UI only.
Requires OpenCV, Pillow, NumPy; only maintaining the video needs these packages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msjh.ttc'))
    args = parser.parse_args()
    rows = json.loads(args.captures.read_text(encoding='utf8'))
    assert len(rows) == 296
    assert [(r['chapter'], r['index']) for r in rows] == (
        [('perception', i) for i in range(60)]+[('signal-turn', i) for i in range(236)])
    target = ROOT/'docs/media/showcase.webm'
    output = cv2.VideoWriter(str(target), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'VP80'), 5, (1280, 720))
    assert output.isOpened()
    fonts = {n: ImageFont.truetype(str(args.font), n) for n in [15, 17, 19, 22, 30]}
    signal_bundle = json.loads((ROOT/'docs/data/signal-turn.json').read_text(encoding='utf8'))
    receipts = []
    for number, row in enumerate(rows):
        assert row['verified'] == 'true' and row['frame'] == row['world'] == row['RGB']
        image_path = args.captures.parent/row['file']
        source = Image.open(image_path).convert('RGB')
        canvas = Image.new('RGB', (1280, 720), '#0a1119')
        draw = ImageDraw.Draw(canvas)
        def text(x, y, value, size=19, color='#edf3f2'):
            draw.text((x, y), value, font=fonts[size], fill=color)
        def wrap(value, x, y, width, size=17, max_lines=3):
            lines = [];line = ''
            for character in value:
                if draw.textlength(line+character, font=fonts[size]) > width:
                    lines.append(line);line = character
                else:line += character
            if line:lines.append(line)
            for j, line in enumerate(lines[:max_lines]):text(x, y+j*(size+8), line, size, '#e3bf8b')
            assert len(lines) <= max_lines, ('Video text would be truncated', value)
        perception = row['chapter'] == 'perception'
        text(20, 12, '01  多物件辨識與短期記憶' if perception else '02  紅燈停等、綠燈起步、左轉抵達', 30)
        text(20, 56, '已知訓練場景的感知重播 · 不含駕駛控制' if perception else
            '實際RGB閉環錄製回放 · 測試環境控制燈色，駕駛只讀RGB', 17, '#a2e6cd')
        text(20, 91, 'AI 三維感知 · 八鏡頭融合', 19)
        text(660, 91, '同步原始 RGB · '+('左前鏡頭' if perception else '前方鏡頭'), 19)
        for key, position, size in [('worldRect', (20, 123), (620, 374)), ('rgbRect', (660, 123), (600, 338))]:
            rect = row[key];box = tuple(round(v) for v in [rect['x'], rect['y'], rect['x']+rect['width'], rect['y']+rect['height']])
            crop = source.crop(box).resize(size, Image.Resampling.LANCZOS)
            canvas.paste(crop, position)
        # A same-frame JPEG inset makes the small, original signal legible.
        # Its crop comes from the RGB model ROI, never the test director's GT.
        roi_receipt = None
        if not perception:
            record = signal_bundle['records'][row['index']]
            assert record['frame'] == int(row['frame'])
            signals = [s for s in record['relevant_RGB_signals'] if s['camera'] == row['camera']]
            if signals:
                asset = signal_bundle['frame_assets'][row['index']]['images'][row['camera']]
                original = ROOT/'docs'/asset['url'];assert sha(original) == asset['web_JPEG_sha256']
                x1,y1,x2,y2 = signals[0]['bbox_xyxy']
                box = (max(0,int(x1)-5),max(0,int(y1)-5),min(640,int(x2)+6),min(360,int(y2)+6))
                zoom = Image.open(original).convert('RGB').crop(box);zoom.thumbnail((100,130))
                # thumbnail does not upscale: explicitly keep the ROI aspect ratio.
                factor = min(100/zoom.width,130/zoom.height)
                zoom = zoom.resize((round(zoom.width*factor),round(zoom.height*factor)),Image.Resampling.NEAREST)
                draw.rectangle((1130,123,1259,291),fill='#101a25',outline='#e3bf8b',width=1)
                canvas.paste(zoom,(1145,149));text(1140,126,'號誌局部放大',15,'#e3bf8b')
                roi_receipt = {'source_JPEG_sha256':asset['web_JPEG_sha256'],'box_xyxy':box,'source':'current_RGB_model_ROI'}
        text(660, 470, row['objects'], 19)
        draw.line((20, 510, 1260, 510), fill='#273642', width=1)
        text(20, 523, '短期記憶 · 金色線框 / TRACK ID', 19, '#a2e6cd')
        wrap(row['memory'], 20, 554, 610, 17, 3)
        text(660, 523, '決策：'+row['behavior']+'    車速：'+row['speed']+' km/h', 22, '#a2e6cd')
        wrap(row['signal'], 660, 561, 600, 19, 2)
        text(660, 620, '感知片段沒有目的地導航' if perception else '剩餘路程：'+row['remaining']+' m', 17, '#9eb0b8')
        text(20, 657, '記憶最長0.65秒，非當前觀測；錯類、漏檢及身份關聯錯誤仍可能存在。', 15, '#9eb0b8')
        text(20, 687, f'F{row["frame"]}  |  模擬時間 {row["time"]}  |  完整時序 1× · 非即時推論 · 無聲', 17, '#9eb0b8')
        output.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        if number in [0, 2, 4, 5, 120, 157, 210, 270, 295]:
            canvas.save(args.captures.parent/f'video-preview-{number:03}.jpg', quality=92)
        if number == 0:canvas.save(ROOT/'docs/media/poster.jpg', quality=90)
        receipts.append({'chapter': row['chapter'], 'index': row['index'], 'frame': int(row['frame']),
            'camera': row['camera'], 'screenshot_sha256': sha(image_path), 'clip': row['clip'],
            'world_crop': row['worldRect'], 'RGB_crop': row['rgbRect'], 'signal_ROI':roi_receipt})
    output.release()
    decoder = cv2.VideoCapture(str(target));count = 0
    while True:
        ok, frame = decoder.read()
        if not ok:break
        assert frame.shape == (720, 1280, 3)
        count += 1
    decoder.release();assert count == 296
    result = {'schema_version': 1, 'video': 'docs/media/showcase.webm', 'sha256': sha(target),
        'codec': 'VP8/WebM', 'fps': 5, 'size': [1280, 720], 'frames': count, 'duration_s': count/5,
        'complete_contiguous_chapters': True, 'interpolated_frames': False, 'audio': False,
        'source': 'Verified CUA screenshots, contemporaneous DOM captions and same-frame RGB model-ROI signal inset',
        'captures_manifest_sha256': sha(args.captures), 'source_crops': receipts}
    (args.captures.parent/'video-receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_crops'}))


if __name__ == '__main__':
    main()
