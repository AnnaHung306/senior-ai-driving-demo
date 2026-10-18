"""Read-only proof that tracked evidence survives Git byte for byte."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    algorithm=subprocess.check_output(['git','rev-parse','--show-object-format'],cwd=ROOT,text=True).strip()
    command=['git','ls-files','--stage','-z','--','evidence'] if args.index else ['git','ls-tree','-r','-z',args.revision,'--','evidence']
    entries=subprocess.check_output(command,cwd=ROOT).split(b'\0');rows=[]
    for entry in entries:
        if not entry:continue
        meta,name=entry.split(b'\t',1);fields=meta.decode().split();path=name.decode('utf-8')
        if args.index:
            mode,expected,stage=fields
            if stage!='0':raise ValueError('Unresolved evidence merge')
        else:mode,kind,expected=fields
        if mode not in {'100644','100755'}:raise ValueError('Only ordinary evidence files supported')
        data=(ROOT/path).read_bytes()
        actual=hashlib.new(algorithm,b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        rows.append({'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'stored_git_object':expected,'raw_git_object':actual,'exact':actual==expected})
    failures=[r['path'] for r in rows if not r['exact']]
    report={'status':'passed' if rows and not failures else 'failed','sampled_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Raw working evidence bytes versus Git index' if args.index else 'Raw working evidence bytes versus committed revision',
        'revision':None if args.index else subprocess.check_output(['git','rev-parse',args.revision],cwd=ROOT,text=True).strip(),
        'git_object_algorithm':algorithm,'checked_files':len(rows),'mismatches':failures,'files':rows,
        'limits':'Byte-preservation proof only. Does not establish model quality, data validity, or semantic receipt correctness.'}
    if args.output:
        output=args.output.resolve()
        if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Write diagnostic under artifacts, not self-referential evidence')
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}));return 0 if report['status']=='passed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',action='store_true');p.add_argument('--revision',default='HEAD')
    p.add_argument('--output',type=Path);raise SystemExit(main(p.parse_args()))
