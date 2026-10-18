"""Inspect or add official CARLA content while preserving all existing bytes.

No server launch, executable extraction, overwrite, or training-data mutation.
The official release supplies no independently verified publisher checksum here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json
from senior_ai.sunny_data import sha

URL = 'https://downloads.carlasim.com/Windows/AdditionalMaps_0.9.16.zip'
EXPECTED_SIZE = 7250055782


def content_path(name):
    """Fail closed before using an archive name on Windows."""
    if '\\' in name or ':' in name or name.startswith('/'):
        raise ValueError('Unsafe archive path')
    parts = name.rstrip('/').split('/')
    if any(p in {'', '.', '..'} or p.endswith((' ', '.')) for p in parts):
        raise ValueError('Unsafe archive component')
    if len(parts) < 3 or parts[:2] != ['CarlaUE4', 'Content']:
        raise ValueError('Only CarlaUE4/Content assets may be installed')
    if any(p.split('.')[0].upper() in {'CON','PRN','AUX','NUL',
           *('COM'+str(i) for i in range(1,10)), *('LPT'+str(i) for i in range(1,10))} for p in parts):
        raise ValueError('Reserved Windows archive component')
    if PurePosixPath(name).suffix.lower() in {'.exe','.dll','.bat','.cmd','.ps1','.py'}:
        raise ValueError('Executable content is not a map asset')
    return Path(*parts)


def same_streams(left, right):
    while True:
        a, b = left.read(4*1024*1024), right.read(4*1024*1024)
        if a != b: return False
        if not a: return True


def main(args):
    archive, output, destination = args.archive.resolve(), args.output.resolve(), args.destination.resolve()
    if not output.is_relative_to(ROOT/'artifacts/runs') or output.exists():
        raise ValueError('Use a new installation receipt directory under artifacts/runs')
    if destination != ROOT/'artifacts/carla' or not (destination/'CarlaUE4.exe').is_file():
        raise ValueError('Existing validated project CARLA installation required')
    if archive.stat().st_size != EXPECTED_SIZE: raise ValueError('Official HEAD byte count differs')
    output.mkdir(parents=True)
    report = {'status':'failed', 'source':URL, 'release_page':'https://github.com/carla-simulator/carla/releases/tag/0.9.16',
        'archive':str(archive), 'bytes':archive.stat().st_size, 'publisher_checksum_verified':False,
        'tool_sha256':sha(Path(__file__)), 'started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'CARLA content assets only; no map/data approval or model-quality claim', 'installed_new_files':[]}
    binary=destination/'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'
    report['shipping_binary_before_sha256']=sha(binary)
    write_json(output/'report.json',report)
    try:
        report['archive_sha256'] = sha(archive)
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            write_json(output/'archive_inventory.json',[{'name':i.filename,'bytes':i.file_size,'CRC32':i.CRC} for i in entries])
            report.update(archive_entries=len(entries), uncompressed_bytes=sum(i.file_size for i in entries))
            if not args.install:
                report['status']='inspected_archive_only_not_installed'
            else:
                files, seen = [], set()
                for item in entries:
                    if item.is_dir(): continue
                    relative = content_path(item.filename)
                    key = relative.as_posix().casefold()
                    if key in seen or stat.S_ISLNK(item.external_attr>>16) or item.flag_bits&1:
                        raise ValueError('Duplicate/symlink/encrypted archive entry')
                    seen.add(key)
                    target = (destination/relative).resolve()
                    if not target.is_relative_to(destination): raise ValueError('Asset escapes destination')
                    files.append((item,relative,target))
                for town in ['Town06','Town07']:
                    if f'CarlaUE4/Content/Carla/Maps/{town}.umap'.casefold() not in seen:
                        raise ValueError('Expected full map absent: '+town)
                # All old paths must match exactly before any installation writes.
                for item,relative,target in files:
                    if target.exists():
                        if not target.is_file() or target.stat().st_size != item.file_size:
                            raise ValueError('Existing asset differs: '+relative.as_posix())
                        with target.open('rb') as old, bundle.open(item) as incoming:
                            if not same_streams(old,incoming): raise ValueError('Existing asset bytes differ: '+relative.as_posix())
                # Stage every new asset. ZipExtFile verifies CRC on complete read.
                staged=[]
                for item,relative,target in files:
                    if target.exists(): continue
                    stage=output/'staged'/relative;stage.parent.mkdir(parents=True,exist_ok=True)
                    with bundle.open(item) as src, stage.open('xb') as dst: shutil.copyfileobj(src,dst,4*1024*1024)
                    staged.append((relative,target,stage,sha(stage)))
                write_json(output/'staged_inventory.json',[{'path':r.as_posix(),'sha256':h,'bytes':p.stat().st_size} for r,t,p,h in staged])
                with (output/'installed_files.jsonl').open('x',encoding='utf-8') as log:
                    for relative,target,stage,digest in staged:
                        target.parent.mkdir(parents=True,exist_ok=True)
                        # A same-directory temporary file cannot be mistaken
                        # for a complete engine asset after interruption.
                        pending=target.with_name(target.name+'.'+output.name+'.partial')
                        with stage.open('rb') as src,pending.open('xb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
                        if sha(pending)!=digest:raise ValueError('Pending asset digest differs')
                        if os.name=='nt':os.rename(pending,target)  # Windows rename refuses an existing target.
                        else:
                            os.link(pending,target)  # Exclusive atomic publication on POSIX too.
                            pending.unlink()
                        if sha(target)!=digest: raise ValueError('Installed asset digest differs')
                        record={'path':relative.as_posix(),'sha256':digest,'bytes':target.stat().st_size}
                        report['installed_new_files'].append(record)
                        log.write(json.dumps(record)+'\n');log.flush()
                        if len(report['installed_new_files'])%100==0:write_json(output/'report.json',report)
                report.update(status='installed_content_only_existing_bytes_preserved',zip_crc_verified=True,
                    existing_exact_files=len(files)-len(staged),new_files=len(staged),
                    publication='Verified sibling .partial copy then exclusive atomic publication; no partially copied asset target')
                report['shipping_binary_after_sha256']=sha(binary)
                if report['shipping_binary_after_sha256']!=report['shipping_binary_before_sha256']:
                    raise ValueError('Existing server binary changed')
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__,message=str(exc));raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='installed_new_files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'artifacts/downloads/AdditionalMaps_0.9.16.zip.partial')
    p.add_argument('--destination',type=Path,default=ROOT/'artifacts/carla')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--install',action='store_true')
    main(p.parse_args())
