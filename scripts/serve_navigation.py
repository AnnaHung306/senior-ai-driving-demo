import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_service import NavigationService,create_server
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8768);p.add_argument('--replay-only',action='store_true');a=p.parse_args()
    service=NavigationService(replay_only=a.replay_only);server=create_server(service,a.port)
    print(f'Navigation: http://127.0.0.1:{a.port}/ (map/RGB hybrid, not certified autonomous driving)',flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:
        service.cancel.set()
        if service.worker:service.worker.join(timeout=90)
        server.server_close()
