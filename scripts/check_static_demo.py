"""Verify the committed static site using only the Python standard library.

This runs after cloning without CARLA, Torch, Pillow, original recordings or
network access. It checks all web bytes and the eight-camera/frame provenance.
"""
import hashlib
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'docs'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.local=[];self.ids=set()
    def handle_starttag(self,tag,attributes):
        attrs=dict(attributes)
        if attrs.get('id'):self.ids.add(attrs['id'])
        for key in ['href','src','poster']:
            value=attrs.get(key,'')
            if value.startswith('./') or value.startswith('#'):self.local.append(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-index',action='store_true',help='Also require every deployed file to be included in the Git index.')
    args=parser.parse_args()
    manifest=read(SITE/'manifest.json')
    for entry in manifest['files']:
        path=SITE/entry['path']
        assert path.resolve().is_relative_to(SITE.resolve()),entry['path']
        data=path.read_bytes()
        assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],path
    expected={x['path'] for x in manifest['files']}|{'manifest.json'}
    actual={x.relative_to(SITE).as_posix() for x in SITE.rglob('*') if x.is_file()}
    assert expected==actual,actual^expected
    if args.git_index:
        tracked=set(subprocess.check_output(['git','ls-files','-z','--','docs'],cwd=ROOT).decode().rstrip('\0').split('\0'))
        missing={'docs/'+path for path in actual}-tracked
        assert not missing,('Deployed files ignored or not staged',sorted(missing))
    catalog=read(SITE/'data/showcases.json');frames=previews=0
    assert len(catalog['chapters'])==3
    for chapter in catalog['chapters']:
        path=SITE/chapter['file'];data=path.read_bytes()
        assert path.resolve().is_relative_to((SITE/'data').resolve())
        assert hashlib.sha256(data).hexdigest()==chapter['sha256']
        bundle=json.loads(data);perception=chapter['kind']=='perception'
        assert bundle['provenance']['other_actor_GT_in_main_view'] is False
        assert bundle['summary']['formal_quality_passed'] is False
        assert len(bundle['records'])==len(bundle['frame_assets'])==bundle['summary']['frames']
        if perception:
            assert bundle['mode']=='recorded_perception' and bundle['summary']['status']=='perception_only'
            assert bundle['provenance']['control_issued'] is False and bundle['memory_max_coast_s']==.65
        else:
            assert bundle['summary']['status']=='arrived' and bundle['summary']['engineering_route_passed']
            assert all(bundle['summary']['checks'].values())
        previous=-1
        for record,assets in zip(bundle['records'],bundle['frame_assets']):
            assert record['frame']==assets['frame'] and record['frame']>previous
            previous=record['frame'];frames+=1
            if perception:
                assert record['GT_used_in_inference'] is False and record['control_issued'] is False
                assert record['future_observations_used'] is False
                objects=record['objects'];cameras=record['cameras']
                held=record['dynamics']['held_unobserved_tracks']
                assert record['dynamics']['held_tracks_used_for_occupancy'] is False
                for h in held:
                    assert h['measurement_current'] is False and h['used_for_occupancy'] is False
                    assert 0<h['age_since_measurement_s']<=bundle['memory_max_coast_s']
                    assert h['track_id'] not in {o['track_id'] for o in objects}
            else:
                assert record['other_actor_GT_in_planning'] is False and record['plan']['actor_GT_used'] is False
                objects=record['perception']['objects'];cameras=record['RGB']
            assert set(assets['images'])==set(cameras) and len(assets['images'])==8
            for camera,image in assets['images'].items():
                assert image['url']==f'images/{record["frame"]}/{camera}.jpg'
                if not perception:assert cameras[camera]['frame']==record['frame']
                assert cameras[camera]['sha256']==image['source_PNG_sha256']
                assert hashlib.sha256((SITE/image['url']).read_bytes()).hexdigest()==image['web_JPEG_sha256']
                previews+=1
            for object in objects:assert 'actor_id' not in object and 'world_pose' not in object
    assert (frames,previews)==(363,2904)
    assert hashlib.sha256((SITE/'data/route.json').read_bytes()).hexdigest()==manifest['bundle_sha256']
    html=(SITE/'index.html').read_text(encoding='utf8');links=Links();links.feed(html)
    for value in links.local:
        if value.startswith('#'):assert value[1:] in links.ids,value
        else:assert (SITE/value[2:]).is_file(),value
    assert '"three":"./vendor/three.module.js"' in html
    assert not re.search(r'https?://[^\s"\']+\.(?:js|css)["\']',html)
    assert (SITE/'vendor/LICENSE.three.txt').is_file()
    assert not list(SITE.rglob('*.pt'))
    print(json.dumps({'status':'passed_portable_static_site','manifest_files':len(manifest['files']),
        'frames':frames,'RGB_previews':previews,'chapters':3,'total_bytes':sum(x['bytes'] for x in manifest['files']),
        'all_HTML_links_resolve':True,'CARLA_Torch_or_original_run_required':False}))


if __name__=='__main__':
    main()
