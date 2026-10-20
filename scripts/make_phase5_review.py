"""Contact pages for all saved WorldFrame cases, fixed representatives and errors."""
import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    out = args.output
    if out.exists():raise FileExistsError('New QA gallery required')
    out.mkdir(parents=True);report = read(args.evaluation/'report.json');records = report['records'];pages = []
    for kind, source, filename, size in [('prediction', args.run, 'prediction.png', (640, 475)), ('evaluation', args.evaluation, 'evaluation.png', (640, 310))]:
        for start in range(0, len(records), 4):
            image = Image.new('RGB', (1280, size[1]*2));bindings = []
            for i, row in enumerate(records[start:start+4]):
                p = source/row['folder']/filename
                with Image.open(p) as im:image.paste(im.resize(size), ((i%2)*640, (i//2)*size[1]))
                d = ImageDraw.Draw(image);pos = ((i%2)*640+2, (i//2)*size[1]+2)
                d.rectangle((*pos, pos[0]+265, pos[1]+15), fill='black');d.text(pos, f"{row['scene_id']} F{row['frame']} {kind}", fill='white')
                bindings.append({'source': str(p), 'sha256': sha(p)})
            name = f'{kind}-{start//4:02}.png';image.save(out/name);pages.append({'kind': kind, 'path': name, 'sha256': sha(out/name), 'sources': bindings})
    scenes = defaultdict(list)
    for row in records:scenes[row['scene_id']].append(row)
    chosen = {(rows[4]['scene_id'], rows[4]['frame']): ['fixed_middle'] for rows in scenes.values()}
    for row in sorted(records, key=lambda r: r['learned']['counts']['FP'], reverse=True)[:2]:chosen.setdefault((row['scene_id'], row['frame']), []).append('largest_object_FP')
    for row in sorted(records, key=lambda r: max([x['yaw_absolute_deg'] for x in r['learned']['matched_errors']] or [-1]), reverse=True)[:2]:chosen.setdefault((row['scene_id'], row['frame']), []).append('largest_matched_yaw_error')
    cases = []
    for (scene, frame), reasons in chosen.items():
        row = next(r for r in records if r['scene_id'] == scene and r['frame'] == frame)
        a = args.run/row['folder']/'prediction.png';b = args.evaluation/row['folder']/'evaluation.png'
        cases.append({'scene_id': scene, 'frame': frame, 'reasons': reasons, 'prediction': str(a), 'prediction_sha256': sha(a), 'evaluation': str(b), 'evaluation_sha256': sha(b)})
    (out/'catalog.json').write_text(json.dumps({'status': 'pending_actual_visual_review', 'run_contract_sha256': sha(args.run/'contract.json'),
           'evaluation_report_sha256': sha(args.evaluation/'report.json'), 'pages': pages, 'full_cases': cases}, indent=2)+'\n', encoding='utf-8')
    print({'pages': len(pages), 'full_cases': len(cases)})


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--run', type=Path, required=True);p.add_argument('--evaluation', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
