"""Manage only the user's explicitly requested demo repository and Pages.

Uses an existing Git Credential Manager login, in process memory only. Never
prints credentials, writes tokens, opens interactive login, overwrites another
repository, force-pushes, or changes account-wide settings.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

OWNER='AnnaHung306'
NAME='senior-ai-driving-demo'
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['inspect','create','pages','status'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'artifacts/reports') or output.exists():
        raise ValueError('Use a new output receipt inside artifacts/reports')
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    lookup=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=env,timeout=30)
    credential=dict(line.split('=',1) for line in lookup.stdout.splitlines() if '=' in line)
    token=credential.get('password')
    if not token:
        raise RuntimeError('No existing noninteractive GitHub credential; sign in with Git Credential Manager first.')

    def api(path,method='GET',body=None):
        request=urllib.request.Request('https://api.github.com'+path,
            data=None if body is None else json.dumps(body).encode(),method=method,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                'X-GitHub-Api-Version':'2022-11-28','User-Agent':'senior-ai-project-deployment',
                'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=40) as response:
                raw=response.read()
                return response.status,json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            value=json.loads(error.read())
            return error.code,{'message':value.get('message'),'errors':value.get('errors')}

    code,user=api('/user')
    if code!=200 or user.get('login')!=OWNER:
        raise RuntimeError('Existing login does not match requested repository owner.')
    path=f'/repos/{OWNER}/{NAME}'
    code,repository=api(path)
    created=False
    if args.operation=='create' and code==404:
        code,repository=api('/user/repos','POST',{'name':NAME,'private':False,'auto_init':False,
            'description':'Eight-camera CARLA autonomous driving research with a verified interactive replay and GitHub Pages demo.',
            'homepage':f'https://{OWNER.lower()}.github.io/{NAME}/'})
        created=code==201
    if code not in [200,201]:
        raise RuntimeError(f'GitHub repository operation failed: HTTP {code}, {repository.get("message")}')
    if repository['full_name']!=f'{OWNER}/{NAME}':raise RuntimeError('Wrong repository')
    if args.operation=='create' and not created and repository.get('size',0)>0:
        raise RuntimeError('Repository already has content; inspect history before pushing.')
    result={'sampled_utc':datetime.now(timezone.utc).isoformat(),'operation':args.operation,
        'repository':repository['html_url'],'created':created,'private':repository['private'],
        'default_branch':repository['default_branch'],'credential_disclosed':False}
    if args.operation in ['pages','status']:
        code,pages=api(path+'/pages')
        if args.operation=='pages':
            source={'branch':'main','path':'/docs'}
            if code==404:code,pages=api(path+'/pages','POST',{'source':source,'build_type':'legacy'})
            elif code==200 and pages.get('source')!=source:
                raise RuntimeError('Existing Pages source differs; do not replace an unrelated deployment.')
        result['pages_http_status']=code
        result['pages']={k:pages.get(k) for k in ['html_url','status','source','build_type','https_enforced']}
        if code not in [200,201]:result['error']=pages
        latest_code,latest=api(path+'/pages/builds/latest')
        result['latest_build_http_status']=latest_code
        result['latest_build']={k:latest.get(k) for k in ['status','commit','duration','created_at','updated_at','error']}
        # Compact read-only workflow status lets a release verify the exact SHA,
        # rather than mistaking an earlier successful Pages build for this one.
        actions_code,actions=api(path+'/actions/runs?per_page=10')
        result['actions_http_status']=actions_code
        result['actions']=[{k:run.get(k) for k in ['name','status','conclusion','head_sha','html_url']}
            for run in actions.get('workflow_runs',[])] if actions_code==200 else []
        if code not in [200,201]:
            output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            raise RuntimeError(f'Pages unavailable, HTTP {code}: {pages.get("message")}')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
