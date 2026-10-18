"""Clone a separate CARLA runtime before overlaying differing addon assets.

Never writes the original installation. File-level exact-contract recovery;
no server launch or map/data approval. Saved runtime outputs are not cloned.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import stat
import zipfile
from install_additional_maps import ROOT,EXPECTED_SIZE,URL,content_path
from senior_ai.core import write_json
from senior_ai.sunny_data import sha


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def inventory(base):
    result=[]
    for p in sorted(base.rglob('*')):
        relative=p.relative_to(base)
        if relative.parts[:2]==('CarlaUE4','Saved'):continue
        if p.is_symlink() or p.is_junction():raise ValueError('No links in baseline distribution')
        if p.is_file():result.append({'path':relative.as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    return result


def atomic_copy(source,target,receipt_name,destination):
    pending=target.with_name(target.name+'.'+receipt_name+'.partial')
    if not target.resolve().is_relative_to(destination) or not pending.resolve().is_relative_to(destination) or pending.is_symlink() or target.is_symlink():
        source.close();raise ValueError('Asset escapes isolated runtime or is a symlink')
    target.parent.mkdir(parents=True,exist_ok=True)
    with source as src,pending.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
    return pending


def main(args):
    base=(ROOT/'artifacts/carla').resolve();destination=args.destination.resolve();out=args.output.resolve();archive=args.archive.resolve()
    if destination.parent!=ROOT/'artifacts' or not destination.name.startswith('carla-phase3-expansion-') or destination==base:
        raise ValueError('Only a distinct project carla-phase3-expansion-* destination')
    if not out.is_relative_to(ROOT/'artifacts/runs') or out==destination or out.is_relative_to(destination):raise ValueError('Separate project run receipt required')
    if not (base/'CarlaUE4.exe').is_file() or archive.stat().st_size!=EXPECTED_SIZE:raise ValueError('Validated base and complete official-sized addon required')
    if (out.exists() or destination.exists()) and not args.resume:raise FileExistsError('Use exact-contract --resume or fresh runtime/run versions')
    if args.resume and not (out/'contract.json').is_file():raise ValueError('No frozen contract to resume')
    out.mkdir(parents=True,exist_ok=True)
    state={'status':'preparing_not_native_validated','started_utc':datetime.now(timezone.utc).isoformat(),'baseline':str(base),'runtime':str(destination),
        'scope':'Isolated runtime content only. No native startup/availability, new data, or model-quality claim.'}
    write_json(out/'state.json',state)
    try:
        print(json.dumps({'stage':'hashing_baseline_and_archive'}),flush=True)
        original=inventory(base);original_by={r['path']:r for r in original}
        with zipfile.ZipFile(archive) as bundle:
            files=[];seen=set()
            for item in bundle.infolist():
                if item.is_dir():continue
                relative=content_path(item.filename);key=relative.as_posix().casefold()
                if key in seen or stat.S_ISLNK(item.external_attr>>16) or item.flag_bits&1:raise ValueError('Unsafe/duplicate addon member')
                seen.add(key);files.append((item,relative))
            for town in ['Town06','Town07']:
                if f'carlaue4/content/carla/maps/{town.lower()}.umap' not in seen:raise ValueError('Expected full map absent')
            contract={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'source':URL,
                'tool_sha256':sha(Path(__file__)),'asset_path_guard_sha256':sha(ROOT/'scripts/install_additional_maps.py'),
                'io_source_sha256':{p:sha(ROOT/p) for p in ['src/senior_ai/core.py','src/senior_ai/sunny_data.py']},
                'baseline':str(base),'runtime':str(destination),'baseline_files':original,'excluded_baseline_prefix':'CarlaUE4/Saved',
                'publisher_checksum_verified':False}
            if (out/'contract.json').exists():
                if read(out/'contract.json')!=contract:raise ValueError('Source/archive/baseline/runtime contract changed')
            else:write_json(out/'contract.json',contract)
            destination.mkdir(exist_ok=True)
            # Overlay cannot start until the baseline clone has completed.
            if not (out/'base_clone_complete.json').exists():
                for index,entry in enumerate(original):
                    source=base/entry['path'];target=destination/entry['path']
                    if target.exists():
                        if sha(target)!=entry['sha256']:raise ValueError('Incomplete clone contains changed bytes: '+entry['path'])
                        continue
                    pending=atomic_copy(source.open('rb'),target,out.name,destination)
                    if sha(pending)!=entry['sha256']:raise ValueError('Base copy differs')
                    if os.name=='nt':os.rename(pending,target)
                    else:os.link(pending,target);pending.unlink()
                    if index%1000==0:print(json.dumps({'stage':'cloning_baseline','files':index+1,'total':len(original)}),flush=True)
                write_json(out/'base_clone_complete.json',{'contract_sha256':sha(out/'contract.json'),'files':len(original)})
            elif read(out/'base_clone_complete.json')['contract_sha256']!=sha(out/'contract.json'):raise ValueError('Clone receipt differs')
            progress=out/'overlay_progress.json';ledger=read(progress) if progress.exists() else {}
            for index,(item,relative) in enumerate(files):
                name=relative.as_posix();target=destination/relative
                if name in ledger:
                    if not target.is_file() or sha(target)!=ledger[name]['sha256'] or ledger[name]['CRC32']!=item.CRC:
                        raise ValueError('Completed overlay asset changed: '+name)
                    continue
                pending=atomic_copy(bundle.open(item),target,out.name,destination)
                digest=sha(pending);old=sha(target) if target.exists() else None
                baseline_entry=original_by.get(name)
                if old==digest:action='already_identical';pending.unlink()
                else:
                    if old is not None and (baseline_entry is None or old!=baseline_entry['sha256']):raise ValueError('Unrecognized isolated asset edit: '+name)
                    # Replacement is restricted to this new isolated clone;
                    # original installation is never a mutation target.
                    os.replace(pending,target);action='replace_in_isolated_clone' if old else 'new'
                ledger[name]={'sha256':digest,'bytes':item.file_size,'CRC32':item.CRC,'action':action}
                if (index+1)%128==0:
                    write_json(progress,ledger)
                    print(json.dumps({'stage':'overlaying_addon','files':index+1,'total':len(files)}),flush=True)
            write_json(progress,ledger)
        expected={r['path']:{'sha256':r['sha256'],'bytes':r['bytes']} for r in original}
        expected.update({name:{'sha256':v['sha256'],'bytes':v['bytes']} for name,v in ledger.items()})
        print(json.dumps({'stage':'verifying_runtime_and_preserved_baseline','files':len(expected)}),flush=True)
        for name,entry in expected.items():
            p=destination/name
            if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('Final isolated asset mismatch: '+name)
        if inventory(base)!=original:raise ValueError('Original baseline changed during preparation')
        binary='CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'
        if expected[binary]['sha256']!=original_by[binary]['sha256']:raise ValueError('Server executable must remain exact')
        write_json(out/'runtime_inventory.json',expected)
        state.update(status='completed_content_not_native_validated',contract_sha256=sha(out/'contract.json'),
            runtime_inventory_sha256=sha(out/'runtime_inventory.json'),baseline_unchanged=True,
            shipping_binary_sha256=expected[binary]['sha256'],baseline_files=len(original),addon_files=len(ledger),
            replaced_only_in_new_clone=sum(v['action']=='replace_in_isolated_clone' for v in ledger.values()),
            new_assets=sum(v['action']=='new' for v in ledger.values()),zip_crc_verified=True,
            finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        state.update(status='failed_preserve_for_exact_contract_resume',error=type(exc).__name__,message=str(exc));raise
    finally:write_json(out/'state.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=ROOT/'artifacts/downloads/AdditionalMaps_0.9.16.zip.partial')
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    main(p.parse_args())
