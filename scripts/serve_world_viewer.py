"""Serve only the verified local 3D viewer bundle; binds loopback, no writes/API."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from senior_ai.world_viewer_service import ViewerBundle, create_server


def main(args):
    bundle = ViewerBundle(args.bundle)
    server = create_server(ROOT/'web/world-viewer', bundle, args.port, args.feed)
    print(f'World viewer ready: http://127.0.0.1:{server.server_address[1]} | {len(bundle.frames)} verified frames', flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__ == '__main__':
    preferred=ROOT/'artifacts/runs/phase7-viewer-bundle-v1-001'
    default_bundle=preferred if (preferred/'bundle.json').is_file() else ROOT/'artifacts/runs/phase6-viewer-bundle-002'
    p = argparse.ArgumentParser();p.add_argument('--bundle', type=Path, default=default_bundle)
    p.add_argument('--port', type=int, default=8766);p.add_argument('--feed', type=Path);main(p.parse_args())
