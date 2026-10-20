"""Loopback read-only viewer service with a hash-bound incremental feed."""
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from threading import Lock
import time
import uuid
from urllib.parse import unquote, urlsplit
from .world_display import read, relative_path, sha

WEB_FILES = {'index.html', 'styles.css', 'app.js', 'math.js', 'data-client.js',
             'vendor/three.module.js', 'vendor/three.core.js', 'vendor/OrbitControls.js', 'vendor/LICENSE.three.txt'}


class ViewerBundle:
    def __init__(self, root):
        self.root = Path(root).resolve();self.manifest = read(self.root/'bundle.json')
        m = self.manifest
        if m['schema_version'] != 1 or m['source_mode'] != 'recorded_inference_replay' or m['control_issued'] is not False or m.get('GT_geometry_exported') is not False or m.get('formal_model_quality_passed') is not False:
            raise ValueError('Unsupported viewer bundle provenance')
        self.inventory = m['file_inventory'];self.frames = {r['key']: r for r in m['frames']}
        if len(self.frames) != len(m['frames']):raise ValueError('Duplicate frame identity')
        for relative, digest in self.inventory.items():
            p = relative_path(relative)
            if p.parts[0] not in {'frames', 'rgb'} or p.suffix not in {'.json', '.bin', '.png'}:raise ValueError('Non-display content in bundle')
            if sha(self.root/relative) != digest:raise ValueError('Changed display bundle file: '+relative)
        for frame in self.frames.values():
            if self.inventory.get(frame['file']) != frame['sha256']:raise ValueError('Unbound display frame')
        previous = {}
        for frame in m['frames']:
            if frame['key'] != frame['scene_id']+'/'+str(frame['frame']) or frame['timestamp_s'] <= previous.get(frame['scene_id'], -float('inf')):
                raise ValueError('Invalid/reversed frame identity')
            previous[frame['scene_id']] = frame['timestamp_s']
        if {key for s in m['scenes'] for key in s['frames']} != set(self.frames):raise ValueError('Incomplete scene index')

    def file(self, relative):
        relative_path(relative)
        if relative not in self.inventory:raise FileNotFoundError(relative)
        p = (self.root/relative).resolve()
        if not p.is_relative_to(self.root) or sha(p) != self.inventory[relative]:raise ValueError('Changed display data')
        return p


class FeedInbox:
    def __init__(self, path, bundle):
        self.path = Path(path).resolve() if path else None;self.bundle = bundle;self.last = None;self.lock = Lock()

    def latest(self):
        with self.lock:return self._latest()

    def _latest(self):
        if self.path is None or not (self.path/'latest.json').exists():return {'status': 'waiting', 'source_mode': 'recorded_inference_replay'}
        value = read(self.path/'latest.json')
        if set(value) != {'schema_version', 'source_contract_sha256', 'key', 'published_utc'} or value['schema_version'] != 1:
            raise ValueError('Strict prediction feed schema required')
        if value['source_contract_sha256'] != self.bundle.manifest['run_contract_sha256'] or value['key'] not in self.bundle.frames:
            raise ValueError('Unknown/unbound prediction feed source')
        published = datetime.fromisoformat(value['published_utc'])
        if published.tzinfo is None:raise ValueError('UTC-aware publication timestamp required')
        age = (datetime.now(timezone.utc)-published).total_seconds()
        if age < -2:raise ValueError('Future publication timestamp rejected')
        frame = self.bundle.frames[value['key']]
        if self.last and self.last['scene_id'] == frame['scene_id'] and frame['timestamp_s'] < self.last['timestamp_s']:
            raise ValueError('Reversed prediction feed timestamp')
        self.last = frame
        return {'status': 'stale' if age > 2 else 'ready', 'source_mode': 'recorded_inference_replay',
                'frame': frame, 'published_utc': value['published_utc'], 'age_ms': max(0., age*1000)}


def publish_existing_frame(bundle, feed, key):
    if key not in bundle.frames:raise ValueError('Only verified existing predicted frames may be published')
    feed = Path(feed);feed.mkdir(parents=True, exist_ok=True)
    value = {'schema_version': 1, 'source_contract_sha256': bundle.manifest['run_contract_sha256'],
             'key': key, 'published_utc': datetime.now(timezone.utc).isoformat()}
    temporary = feed/('.latest.'+uuid.uuid4().hex+'.pending.json');temporary.write_text(json.dumps(value)+'\n', encoding='utf-8')
    try:
        # Windows readers briefly hold a handle without FILE_SHARE_DELETE.
        # Keep the old complete pointer available and retry this atomic replace.
        deadline = time.monotonic()+2
        while True:
            try:temporary.replace(feed/'latest.json');break
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.01)
    finally:temporary.unlink(missing_ok=True)
    return value


def create_server(web, bundle, port=8766, feed=None):
    web = Path(web).resolve();inbox = FeedInbox(feed, bundle)
    html = (web/'index.html').read_text(encoding='utf-8')
    imports = re.search(r'<script type="importmap">(.*?)</script>', html, flags=re.S)
    if not imports:raise ValueError('Pinned local import map required')
    import base64
    map_hash = base64.b64encode(hashlib.sha256(imports.group(1).encode('utf-8')).digest()).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):pass
        def respond(self, status, content, mime='application/json; charset=utf-8'):
            self.send_response(status);self.send_header('Content-Type', mime);self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store');self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', f"default-src 'self'; script-src 'self' 'sha256-{map_hash}'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'")
            self.end_headers()
            if self.command != 'HEAD':self.wfile.write(content)
        def do_HEAD(self):self.do_GET()
        def do_GET(self):
            try:
                path = unquote(urlsplit(self.path).path)
                if path == '/api/health':return self.respond(200, json.dumps({'status': 'ready', 'frames': len(bundle.frames), 'read_only': True}).encode())
                if path == '/api/manifest':return self.respond(200, json.dumps(bundle.manifest).encode())
                if path == '/api/feed':return self.respond(200, json.dumps(inbox.latest()).encode())
                if path.startswith('/data/'):
                    p = bundle.file(path[6:])
                else:
                    relative = 'index.html' if path == '/' else path.lstrip('/')
                    relative_path(relative)
                    if relative not in WEB_FILES:raise FileNotFoundError(relative)
                    p = (web/relative).resolve()
                    if not p.is_relative_to(web):raise ValueError('Unsafe asset path')
                mime = {'js': 'text/javascript; charset=utf-8', 'html': 'text/html; charset=utf-8', 'css': 'text/css; charset=utf-8', 'json': 'application/json'}.get(p.suffix[1:], mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
                return self.respond(200, p.read_bytes(), mime)
            except (ValueError, KeyError, TypeError) as e:self.respond(422, json.dumps({'status': 'invalid', 'message': str(e)}).encode())
            except (FileNotFoundError, IsADirectoryError):self.respond(404, b'{"status":"not_found"}')
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)
