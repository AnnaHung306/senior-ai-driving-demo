"""Archive actually inspected, diagnostic-only Phase3 wall QA evidence."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination=ROOT/'evidence/phase3/sunny-v3/wall-diagnosis-001'
    if destination.exists():raise FileExistsError('Immutable evidence destination already exists')
    destination.mkdir(parents=True)
    specs=[('sunny-segmentation-v5-first-wall-review-001','catalog.json',[f'page-{i}.png' for i in range(3)]),
        ('wall-training-shape-survey-001','report.json',[f'{kind}-{i}.png' for kind in ['low_wall_hint','neutral_nonwall_hint'] for i in range(4)]),
        ('wall-training-appearance-survey-001','report.json',[f'page-{i}.png' for i in range(6)])]
    copied=[]
    for run,report,pages in specs:
        source=ROOT/'artifacts/runs'/run;target=destination/run;target.mkdir()
        for name in [report]+pages+(['records.jsonl.gz'] if run=='wall-training-shape-survey-001' else []):
            shutil.copyfile(source/name,target/name)
            copied.append({'path':(target/name).relative_to(destination).as_posix(),'sha256':sha(target/name)})
    source=ROOT/'artifacts/runs/wall-low-geometry-inventory-001.json'
    shutil.copyfile(source,destination/source.name)
    receipt={'status':'reviewed_diagnostic_only','training_approved':False,'reviewed_by':'Codex actual view_image inspection in active task',
        'reviewed_images':[r for r in copied if r['path'].endswith('.png')],'files':copied,
        'observations':[
            'First-epoch validation QA: some missed Wall GT is solid wooden perimeter fencing; highway concrete barriers also missed. Rocks/cliffs create Wall false positives. Earlier low-concrete-only diagnosis was incomplete.',
            'Train shape survey: 24 inspected examples include low concrete walls, red-brick walls and building/vegetation/rock nonwall context. Neutral colour is not a material label.',
            'Train appearance survey: all 16 selected contexts on 6 pages inspected. They show concrete block walls, brick compound walls, tunnel walls, masonry and low retaining walls. Inspected pages do not establish coverage of the solid wooden Wall GT appearance.',
            'Current static supplement does contain visually wooden Fence GT. Similar material alone does not justify merging or relabelling Wall and Fence.',
            'Geometry inventory is offline train-town metadata hints only; no new captures, physical-height proof, or material labels.'],
        'limits':'Selected QA is not exhaustive, nor a new acceptance set. CPU FP32 checkpoint diagnostics do not replace completed GPU BF16 validation. No heldout opened, gates changed, or training samples approved.'}
    (destination/'visual_review.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'destination':str(destination),'actually_viewed_pages':len(receipt['reviewed_images'])}))


if __name__=='__main__':main()
