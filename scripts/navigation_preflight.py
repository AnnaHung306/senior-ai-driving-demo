"""Bounded startup and explicit map identity; no vehicle spawning or world ticks."""
import sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.world_display import read
def main():
    import carla
    client=carla.Client('127.0.0.1',2000);client.set_timeout(2)
    deadline=time.monotonic()+90
    while True:
        try:version=client.get_server_version();break
        except RuntimeError:
            if time.monotonic()>deadline:raise TimeoutError('CARLA not ready; start simulator or use replay-only')
            time.sleep(.5)
    if version!='0.9.16':raise ValueError('CARLA0.9.16 required')
    client.set_timeout(60);w=client.get_world()
    if w.get_settings().synchronous_mode:raise RuntimeError('Another synchronous controller owns world')
    if w.get_map().name.split('/')[-1]!='Town01':w=client.load_world('Town01')
    preferred=ROOT/'artifacts/runs/phase89-route-catalog-v1-002/catalog.json'
    catalog=read(preferred if preferred.exists() else ROOT/'configs/phase89_routes_v1.json')
    if hashlib.sha256(w.get_map().to_opendrive().encode()).hexdigest()!=catalog['opendrive_sha256']:raise ValueError('Map identity changed')
    print({'status':'ready','CARLA':version,'map':w.get_map().name,'weather_scope':'ClearNoon enforced on every job','tick_owner_started':False})
if __name__=='__main__':main()
