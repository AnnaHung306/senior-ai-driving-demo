import argparse,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read
from senior_ai.navigation_runtime import drive
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--route',default='straight-1');p.add_argument('--output',type=Path,required=True);p.add_argument('--fault',choices=['sensor_loss','camera_timeout']);a=p.parse_args()
    preferred=ROOT/'artifacts/runs/phase89-route-catalog-v1-002/catalog.json'
    routes=read(preferred if preferred.is_file() else ROOT/'configs/phase89_routes_v1.json')['routes']
    r=next(r for r in routes if r['id']==a.route)
    result=drive(r,a.output,threading.Event(),fault=a.fault)
    print({k:result[k] for k in ['status','engineering_route_passed','checks','frames','wall_s']})
