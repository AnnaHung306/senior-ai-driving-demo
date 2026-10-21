"""Verify an actual static deployment against the local manifest, read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='https://annahung306.github.io/senior-ai-driving-demo/')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'artifacts/reports'):
        raise ValueError('Use a new receipt inside artifacts/reports')
    manifest=json.loads((ROOT/'docs/manifest.json').read_text(encoding='utf8'))
    entries={r['path']:r for r in manifest['files']}
    paths=['index.html','app.js','replay.js','record-contract.js','style.css',
        'vendor/three.module.js','vendor/three.core.js','vendor/OrbitControls.js',
        'data/route.json','data/showcases.json','data/perception.json','data/signal-turn.json',
        'manifest.json','media/showcase.webm']
    bundle=json.loads((ROOT/'docs/data/route.json').read_text(encoding='utf8'))
    paths += [a['url'] for a in bundle['frame_assets'][18]['images'].values()]
    for name,index in [('perception',4),('signal-turn',97),('signal-turn',235)]:
        extra=json.loads((ROOT/f'docs/data/{name}.json').read_text(encoding='utf8'))
        paths += [a['url'] for a in extra['frame_assets'][index]['images'].values()]
    rows=[]
    for relative in paths:
        with urllib.request.urlopen(urllib.request.Request(args.url.rstrip('/')+'/'+relative,
                headers={'User-Agent':'senior-ai-static-deployment-check','Accept-Encoding':'identity'}),timeout=45) as response:
            body=response.read();mime=response.headers.get_content_type();status=response.status
        expected=entries.get(relative,{}).get('sha256') or hashlib.sha256((ROOT/'docs'/relative).read_bytes()).hexdigest()
        digest=hashlib.sha256(body).hexdigest()
        assert digest==expected,(relative,'deployment bytes differ')
        if relative.endswith('.js'):assert mime in ['text/javascript','application/javascript'],(relative,mime)
        if relative.endswith('.webm'):assert mime=='video/webm',mime
        rows.append({'path':relative,'status':status,'bytes':len(body),'content_type':mime,'sha256':digest,'exact':True})
    result={'status':'passed_actual_static_deployment','url':args.url,'checks':rows,
        'site_manifest_sha256':hashlib.sha256((ROOT/'docs/manifest.json').read_bytes()).hexdigest(),
        'limits':'HTTP bytes/MIME check; browser rendering and interaction have separate receipts.'}
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'url':args.url,'HTTP_checks':len(rows)}))


if __name__=='__main__':
    main()
