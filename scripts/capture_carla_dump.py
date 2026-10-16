"""Capture a small native diagnostic dump of this project's own CARLA server.

No heap/full-memory dump. The exact executable path must match the owned install.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
import msvcrt
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Dump must be a new path within this project')
    executable = (args.server_root / 'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe').resolve()
    literal = str(executable).replace("'", "''")
    query = "Get-Process -Name CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq '" + literal + "' } | Select-Object Id,Path | ConvertTo-Json -Compress"
    probe = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', query],
                           capture_output=True, text=True, check=True, timeout=15)
    matches = json.loads(probe.stdout) if probe.stdout.strip() else []
    matches = [matches] if isinstance(matches, dict) else matches
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one server matching this install; no dump captured')
    pid = matches[0]['Id']
    kernel = C.WinDLL('kernel32', use_last_error=True)
    dbg = C.WinDLL('dbghelp', use_last_error=True)
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [W.DWORD, W.BOOL, W.DWORD], W.HANDLE
    kernel.CloseHandle.argtypes = [W.HANDLE]
    dbg.MiniDumpWriteDump.argtypes = [W.HANDLE, W.DWORD, W.HANDLE, W.DWORD, C.c_void_p, C.c_void_p, C.c_void_p]
    dbg.MiniDumpWriteDump.restype = W.BOOL
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        raise C.WinError(C.get_last_error())
    try:
        with output.open('xb') as stream:
            if not dbg.MiniDumpWriteDump(handle, pid, msvcrt.get_osfhandle(stream.fileno()),
                                        0x1000, None, None, None):
                raise C.WinError(C.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    print(json.dumps({'pid': pid, 'executable': str(executable), 'dump_bytes': output.stat().st_size}))


if __name__ == '__main__':
    main()
