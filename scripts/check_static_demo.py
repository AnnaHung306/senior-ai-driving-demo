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
    data=(SITE/'data/route.json').read_bytes()
    assert hashlib.sha256(data).hexdigest()==manifest['bundle_sha256']
    bundle=json.loads(data)
    assert len(bundle['records'])==67 and manifest['RGB_previews']==536
    assert bundle['summary']['status']=='arrived' and bundle['summary']['engineering_route_passed']
    assert all(bundle['summary']['checks'].values())
    previous=-1
    for record,assets in zip(bundle['records'],bundle['frame_assets']):
        assert record['frame']==assets['frame'] and record['frame']>previous
        previous=record['frame']
        assert record['other_actor_GT_in_planning'] is False and record['plan']['actor_GT_used'] is False
        assert set(assets['images'])==set(record['RGB']) and len(assets['images'])==8
        for camera,image in assets['images'].items():
            assert image['url']==f'images/{record["frame"]}/{camera}.jpg'
            assert record['RGB'][camera]['frame']==record['frame']
            assert record['RGB'][camera]['sha256']==image['source_PNG_sha256']
            assert hashlib.sha256((SITE/image['url']).read_bytes()).hexdigest()==image['web_JPEG_sha256']
        for object in record['perception']['objects']:
            assert 'actor_id' not in object and 'world_pose' not in object
    html=(SITE/'index.html').read_text(encoding='utf8');links=Links();links.feed(html)
    for value in links.local:
        if value.startswith('#'):assert value[1:] in links.ids,value
        else:assert (SITE/value[2:]).is_file(),value
    assert '"three":"./vendor/three.module.js"' in html
    assert not re.search(r'https?://[^\s"\']+\.(?:js|css)["\']',html)
    assert (SITE/'vendor/LICENSE.three.txt').is_file()
    assert not list(SITE.rglob('*.pt'))
    print(json.dumps({'status':'passed_portable_static_site','manifest_files':len(manifest['files']),
        'frames':67,'RGB_previews':536,'total_bytes':sum(x['bytes'] for x in manifest['files']),
        'all_HTML_links_resolve':True,'CARLA_Torch_or_original_run_required':False}))


if __name__=='__main__':
    main()
