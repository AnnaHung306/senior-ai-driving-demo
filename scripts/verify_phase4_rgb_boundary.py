"""Exercise actual CLI rejection before model loading; CPU/no GT inference."""
import argparse,json,hashlib,subprocess,sys,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(args):
    out=args.output.resolve()
    if out.exists():raise ValueError('New boundary verification output required')
    out.mkdir(parents=True);spec=json.loads((args.selection/'sequence_inputs.json').read_text());original=args.selection/'rgb_only';seq=spec['sequences'][0];manifest=seq['manifests'][0];frame=json.loads((original/manifest).read_text());cases=[]
    for kind in ['extra_GT_file','GT_pose_in_sequence']:
        folder=out/kind;stage=folder/'root';stage.mkdir(parents=True);inventory={}
        for rel in [manifest,frame['calibration']]+[x['image'] for x in frame['cameras'].values()]:
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/rel,target);inventory[rel]=sha(target)
        inv=folder/'inventory.json';inv.write_text(json.dumps(inventory),encoding='utf-8');value={**spec,'sequences':[{'scene_id':seq['scene_id'],'manifests':[manifest]}],'inventory_sha256':sha(inv)}
        if kind=='extra_GT_file':
            (stage/'labels').mkdir();(stage/'labels/fake.json').write_text('{"source":"simulator_ground_truth"}',encoding='utf-8');expected='Staging must contain only frozen RGB inputs and calibration'
        else:value['sequences'][0]['ego_world_pose']=[0,0,0];expected='GT fields forbidden in sequence input'
        inp=folder/'inputs.json';inp.write_text(json.dumps(value),encoding='utf-8');prediction=folder/'must_not_exist'
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'}
        p=subprocess.run([sys.executable,str(ROOT/'scripts/run_phase4_pilot.py'),'--root',str(stage),'--input-list',str(inp),'--inventory',str(inv),'--config',str(ROOT/'configs/phase4_bev_v1.json'),'--output',str(prediction),'--device','cpu'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=60)
        (folder/'stderr.log').write_text(p.stderr,encoding='utf-8');assert p.returncode!=0 and expected in p.stderr and not prediction.exists(),kind
        cases.append({'case':kind,'return_code':p.returncode,'expected_rejection':expected,'output_directory_not_created':True,'stderr_sha256':sha(folder/'stderr.log')})
    report={'status':'passed_expected_rejections','tool_sha256':sha(Path(__file__)),'inference_tool_sha256':sha(ROOT/'scripts/run_phase4_pilot.py'),'scope':'Actual CPU CLI rejected GT contamination before model construction/forward; no simulator or GPU work','cases':cases}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'cases':len(cases)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
