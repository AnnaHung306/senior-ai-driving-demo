"""Collect reproducible environment evidence; unknown values never become passes."""
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from .core import write_json


def inspect(root, host='127.0.0.1', port=2000):
    info = {'python': sys.version, 'executable': sys.executable,
            'os': platform.platform(), 'logical_cpus': os.cpu_count(),
            'disk_free_gib': round(shutil.disk_usage(root).free / 2**30, 2)}
    errors = []
    if sys.platform == 'win32':
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
                (n, ctypes.c_ulonglong) for n in
                ['total', 'avail', 'page', 'availpage', 'virt', 'availvirt', 'ext']]
        m = Memory()
        m.length = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            info['ram_gib'] = round(m.total / 2**30, 2)
            info['free_ram_gib'] = round(m.avail / 2**30, 2)
        else:
            errors.append('GlobalMemoryStatusEx failed')
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                info['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError as exc:
            errors.append(str(exc))
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                            '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, timeout=15, check=True)
        info['gpus'] = []
        for line in p.stdout.strip().splitlines():
            name, memory, driver = [v.strip() for v in line.split(',')]
            info['gpus'].append({'name': name, 'vram_mib': int(memory), 'driver': driver})
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        errors.append(f'GPU probe: {exc}')
    try:
        info['carla_client'] = importlib.metadata.version('carla')
        import carla
        info['carla_import'] = True
    except (ImportError, importlib.metadata.PackageNotFoundError, OSError) as exc:
        info['carla_import'] = False
        errors.append(f'CARLA: {exc}')
    info['server_tcp_open'] = False
    try:
        with socket.create_connection((host, port), timeout=2):
            info['server_tcp_open'] = True
    except OSError:
        pass
    if info['carla_import'] and info['server_tcp_open']:
        try:
            client = carla.Client(host, port)
            client.set_timeout(10)
            info['server_version'] = client.get_server_version()
        except RuntimeError as exc:
            errors.append(f'CARLA RPC: {exc}')
    checks = {'python_compatible': (3, 10) <= sys.version_info[:2] <= (3, 12),
              'carla_client': info.get('carla_client') == '0.9.16' and info['carla_import'],
              'server_rpc': info.get('server_version') == '0.9.16',
              # 8GB laptop GPUs reserve a few MiB, so nominal 8GB is >=8000 MiB.
              'gpu_budget': any(g['vram_mib'] >= 8000 for g in info.get('gpus', [])),
              'ram_budget': info.get('ram_gib', 0) >= 16,
              'disk_budget': info['disk_free_gib'] >= 50}
    return {'schema_version': 1, 'task': 'phase0-environment',
            'status': 'passed' if all(checks.values()) else 'blocked',
            'environment': info, 'checks': checks, 'errors': errors,
            'note': 'Environment readiness does not establish simulator control or AI capability.'}


def run(args):
    report = inspect(Path.cwd(), args.host, args.port)
    write_json(args.output, report)
    print(json.dumps({'task': report['task'], 'status': report['status'],
                      'checks': report['checks'], 'report': str(args.output)}))
    return 0 if report['status'] == 'passed' else 2
