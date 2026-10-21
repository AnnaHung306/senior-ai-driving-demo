"""Export one successful real driving run to a portable, read-only static site.

Original PNG/record hashes remain in provenance. JPEG is only a web preview;
model outputs are copied unchanged and no GT geometry enters the site bundle.
Run from any directory. Existing audited artifacts/evidence are never changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'phase10-drive-pedestrian-probe-v1-001'


def sha(path):
    # hashlib.file_digest is unavailable on Python 3.10, which is sufficient
    # for maintaining the static demonstration without the training runtime.
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-only', action='store_true', help='Refresh checksums after editing site code; original run is unnecessary.')
    args = parser.parse_args()
    site = ROOT/'docs'
    if not args.manifest_only:
        from PIL import Image
        run = ROOT/'artifacts/runs'/RUN_ID
        report = read(run/'report.json')
        assert report['engineering_route_passed'] and all(report['checks'].values())
        assert report['status']=='arrived' and report['frames']==67
        records = []
        entries = []
        for entry in report['records']:
            source = run/entry['file']
            assert sha(source)==entry['sha256']
            record = read(source)
            assert record['other_actor_GT_in_planning'] is False
            assert record['perception']['GT_other_actors_used'] is False
            images = {}
            for camera, rgb in record['RGB'].items():
                original = run/rgb['file']
                assert sha(original)==rgb['sha256'] and rgb['frame']==record['frame']
                target = site/f'images/{record["frame"]}/{camera}.jpg'
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(original) as image:
                    assert image.size==(640,360)
                    image.convert('RGB').save(target,quality=86,optimize=True,subsampling=0)
                images[camera]={'url':target.relative_to(site).as_posix(),
                    'web_JPEG_sha256':sha(target),'source_PNG_sha256':rgb['sha256'],
                    'width':640,'height':360}
            # Preserve numeric model/control output exactly; transport formatting differs.
            records.append(record)
            entries.append({'frame':record['frame'],'record_source_sha256':entry['sha256'],'images':images})
        catalog = read(ROOT/'configs/phase89_routes_v1.json')
        bundle = {'schema_version':1,'mode':'recorded_replay','run_id':RUN_ID,
            'summary':{key:report[key] for key in ['status','engineering_route_passed','frames','simulation_s',
                'wall_s','checks','cleanup_final_speed_mps','formal_quality_passed']},
            'route':report['route'],'map_roads_xy_m':catalog['roads_xy_m'],
            'camera_names':list(records[0]['RGB']), 'records':records,'frame_assets':entries,
            'provenance':{'source_report_sha256':sha(run/'report.json'),
                'source_contract_sha256':sha(run/'contract.json'),
                'models':read(run/'contract.json')['models'],
                'model_outputs_unchanged':True,'JPEG_is_lossy_preview_only':True,
                'other_actor_GT_in_main_view':False,'map_and_ego_pose_are_explicit_priors':True,
                'not_live_inference':True,'only_this_named_route_verified':True}}
        write(site/'data/route.json',bundle)
        vendor = site/'vendor'
        vendor.mkdir(parents=True,exist_ok=True)
        for name in ['three.module.js','three.core.js','OrbitControls.js','LICENSE.three.txt','provenance.json']:
            shutil.copyfile(ROOT/'web/world-viewer/vendor'/name,vendor/name)
        shutil.copyfile(ROOT/'web/world-viewer/navigation-contract.js',site/'record-contract.js')
        media=site/'media';media.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'evidence/phase10/system-audit-v1-001/video/phase10-focus-replay.webm',media/'driving-replay.webm')
        first=site/f'images/{records[0]["frame"]}/front.jpg'
        shutil.copyfile(first,media/'poster.jpg')
        (site/'.nojekyll').write_text('',encoding='utf8')
    payload=read(site/'data/route.json')
    catalog_path=site/'data/showcases.json'
    bundles=[read(site/c['file']) for c in read(catalog_path)['chapters']] if catalog_path.exists() else [payload]
    inventory=[{'path':p.relative_to(site).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(site.rglob('*')) if p.is_file() and p.name!='manifest.json']
    write(site/'manifest.json',{'schema_version':1,'files':inventory,'manifest_self_excluded':True,
        'bundle_sha256':sha(site/'data/route.json'),'frames':sum(len(b['records']) for b in bundles),
        'RGB_previews':sum(len(x['images']) for b in bundles for x in b['frame_assets']),
        'purpose':'Static recorded demo; no control endpoint or model weights.'})
    print(json.dumps({'status':'static_site_exported','files':len(inventory),
        'bytes':sum(x['bytes'] for x in inventory),'frames':sum(len(b['records']) for b in bundles)}))


if __name__=='__main__':
    main()
