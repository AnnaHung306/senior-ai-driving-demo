"""Resumable per-episode cross-town acquisition; no scene reselection on failure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import shutil

from collect_dataset import ROOT, collect_scene, sha
from senior_ai.core import write_json


def main(args):
    import carla
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    rig_path=ROOT/'configs/camera_rig_v1.json'; rig=json.loads(rig_path.read_text(encoding='utf-8'))
    if sha(rig_path)!=plan['rig_sha256']: raise ValueError('Frozen rig changed')
    root=args.output.resolve()
    criteria={**plan['criteria'],'record_traffic_lights':True,'record_weather':True}
    signature={'plan':sha(args.plan),'collector':sha(ROOT/'scripts/collect_dataset.py'),'safety':sha(ROOT/'src/senior_ai/acquisition_safety.py'),
               'runner':sha(Path(__file__)),'frame':sha(ROOT/'src/senior_ai/frames.py')}
    if root.exists():
        report=json.loads((root/'report.json').read_text(encoding='utf-8'))
        if report['signature']!=signature: raise ValueError('Cannot resume changed acquisition')
        if report['status']=='passed': raise ValueError('Already completed')
    else:
        root.mkdir(parents=True)
        for name,value in [('plan',plan),('rig',rig),('criteria',criteria)]: write_json(root/(name+'.json'),value)
        report={'task':'sunny-cross-town-collect-v3','status':'in_progress','signature':signature,
                'weather':'ClearNoon','training_approved':False,'scenes':[],'reuse_lineage':[],
                'server_launch':json.loads((ROOT/'artifacts/reports/server-launch.json').read_text(encoding='utf-8-sig'))}
        write_json(root/'report.json',report)
    client=carla.Client('127.0.0.1',2000); client.set_timeout(120)
    if client.get_server_version()!='0.9.16' or client.get_client_version()!='0.9.16':
        raise ValueError('Matching CARLA 0.9.16 required')
    report['server_version']=client.get_server_version();report['client_version']=client.get_client_version()
    all_indexes=[]
    for scene in plan['scenes']:
        folder=root/'scenes'/scene['id']
        if not folder.exists() and args.reuse_root:
            source=args.reuse_root.resolve();old_plan=json.loads((source/'plan.json').read_text(encoding='utf-8'))
            old_scene=next((s for s in old_plan['scenes'] if s['id']==scene['id']),None)
            previous=source/'scenes'/scene['id']/'report.json'
            # Reuse only an exactly unchanged frozen scene with all original
            # acceptance gates passed. Never copy a failed/incomplete episode.
            if old_scene==scene and previous.exists():
                result=json.loads(previous.read_text(encoding='utf-8'))
                old_report=json.loads((source/'report.json').read_text(encoding='utf-8'))
                if result['status']=='passed' and all(result['checks'].values()):
                    if old_plan['criteria']!=plan['criteria'] or old_plan['rig_sha256']!=plan['rig_sha256']:
                        raise ValueError('Reuse criteria or optics changed')
                    def copy_file(src,dst):
                        # Raw PNG hardlinks are immutable acquisition bytes;
                        # metadata is copied so reports cannot mutate the source.
                        if str(src).endswith('.png'): __import__('os').link(src,dst);return str(dst)
                        return shutil.copy2(src,dst)
                    for relative in [f'scenes/{scene["id"]}',f'inputs/frames/{scene["id"]}',f'inputs/rgb/{scene["id"]}',
                                     f'labels/frames/{scene["id"]}']+[f'labels/{m}/{scene["id"]}' for m in criteria['modalities'] if m!='rgb']:
                        shutil.copytree(source/relative,root/relative,copy_function=copy_file)
                    for relative in [f'calibration/{scene["id"]}.json',f'labels/roads/{scene["id"]}.json',f'labels/traffic_lights/{scene["id"]}.json']:
                        if (source/relative).exists():
                            (root/relative).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,root/relative)
                    report['scenes'].append(result)
                    report['reuse_lineage'].append({'scene_id':scene['id'],'source_root':str(source),'source_report_sha256':sha(source/'report.json'),
                        'source_plan_sha256':sha(source/'plan.json'),'source_episode_report_sha256':sha(previous),
                        'source_signature':old_report['signature'],'note':'Same geometry/seed/split/criteria/rig; no model scoring used. Full independent raw audit still required.'})
                    write_json(root/'report.json',report)
        if folder.exists():
            prior=json.loads((folder/'report.json').read_text(encoding='utf-8'))
            if prior['status']!='passed': raise ValueError(f'Failed/incomplete scene retained: {scene["id"]}; use a new capture version')
            indexes=json.loads((folder/'index.json').read_text(encoding='utf-8')); all_indexes.extend(indexes); continue
        world=client.get_world()
        if world.get_map().name.split('/')[-1]!=scene['town']: world=client.load_world(scene['town'])
        if hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()!=plan['maps'][scene['town']]['opendrive_sha256']:
            raise ValueError('Map geometry changed')
        result,indexes=collect_scene(world,client,scene,rig,criteria,root,criteria['bundles_per_scene'])
        write_json(folder/'index.json',indexes); report['scenes'].append(result); all_indexes.extend(indexes)
        write_json(root/'report.json',report)
        print(json.dumps({'scene':scene['id'],'status':result['status'],'images':result['images'],
            'wall_s':result['wall_s'],'errors':result['errors'],'checks':result['checks'],'motion':result.get('motion')}),flush=True)
        if result['status']!='passed':
            report['status']='failed'; write_json(root/'report.json',report); return 1
    for split in ['train','validation','development','lockbox']:
        path=root/f'indexes/{split}.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(e)+'\n' for e in all_indexes if e['split']==split),encoding='utf-8')
    report['scenes']=[json.loads((root/'scenes'/s['id']/'report.json').read_text(encoding='utf-8')) for s in plan['scenes']]
    if len(all_indexes)!=len(plan['scenes'])*criteria['bundles_per_scene'] or any(s['status']!='passed' for s in report['scenes']):
        raise ValueError('Incomplete episode collection')
    report.update({'status':'passed','total_bundles':len(all_indexes),'total_images':sum(s['images'] for s in report['scenes'])})
    write_json(root/'report.json',report); return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--plan',type=Path,required=True); p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-root',type=Path)
    sys.exit(main(p.parse_args()))
