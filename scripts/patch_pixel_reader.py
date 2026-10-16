"""Reversible, build-specific CARLA scheduling fix, backed by source/PDB evidence.

This is NOT a general executable patcher. Only the exact archived 0.9.16 build and
the verified instruction site are accepted. The original executable is preserved.
For maintained builds, apply patches/pixel-reader-task-thread.patch and rebuild.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'artifacts/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'
BACKUP = BINARY.with_suffix('.exe.stock')
MANIFEST = ROOT / 'artifacts/reports/pixel-reader-patch.json'
INSTRUCTION_RVA = 0x8945B8
ORIGINAL = bytes.fromhex('48 8d 54 24 70 b9 00 02 00 00 e8 2e c1 7c 00')
PATCHED = bytes.fromhex('48 8d 54 24 70 b9 ff 02 00 00 e8 2e c1 7c 00')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_offset(data, rva):
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe + 4] != b'PE\0\0':
        raise ValueError('Not a PE executable')
    count = struct.unpack_from('<H', data, pe + 6)[0]
    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
    for i in range(count):
        section = pe + 24 + optional_size + i * 40
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from('<IIII', data, section + 8)
        if virtual_address <= rva < virtual_address + min(virtual_size, raw_size):
            return raw_pointer + rva - virtual_address
    raise ValueError('Instruction RVA is outside file-backed sections')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    assert BINARY.resolve().is_relative_to((ROOT / 'artifacts/carla').resolve())
    current = BINARY.read_bytes()
    if args.restore:
        manifest = json.loads(MANIFEST.read_text())
        if sha(current) != manifest['patched_sha256'] or sha(BACKUP.read_bytes()) != manifest['stock_sha256']:
            raise ValueError('Restore refused: binary or backup was changed')
        shutil.copy2(BACKUP, BINARY)
        manifest['active'] = False
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(json.dumps({'status': 'restored', 'sha256': sha(BINARY.read_bytes())}))
        return
    with zipfile.ZipFile(ROOT / 'artifacts/downloads/CARLA_0.9.16.zip') as archive:
        member = 'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'
        archived = archive.read(member)
    if sha(current) != sha(archived):
        raise ValueError('Patch refused: executable is not the original archived build')
    offset = file_offset(current, INSTRUCTION_RVA - 5)
    if current[offset:offset + len(ORIGINAL)] != ORIGINAL:
        raise ValueError('Patch refused: disassembled instruction context does not match')
    if BACKUP.exists() and sha(BACKUP.read_bytes()) != sha(current):
        raise ValueError('Existing original backup has a different hash; preserving it')
    if not BACKUP.exists():
        shutil.copy2(BINARY, BACKUP)
    modified = bytearray(current)
    modified[offset:offset + len(PATCHED)] = PATCHED
    if sum(a != b for a, b in zip(current, modified)) != 1:
        raise ValueError('Expected exactly one changed byte')
    BINARY.write_bytes(modified)
    manifest = {'active': True, 'status': 'experimental_pending_acceptance',
                'stock_sha256': sha(current), 'patched_sha256': sha(modified),
                'changed_bytes': 1, 'instruction_rva': hex(INSTRUCTION_RVA),
                'file_offset': hex(offset + 6), 'from_task_enum': 512, 'to_task_enum': 767,
                'from_name': 'HighTaskPriority (thread index 0 = StatsThread)',
                'to_name': 'AnyNormalThreadHiPriTask (AnyThread | HighTaskPriority)',
                'backup': str(BACKUP), 'source_patch': 'patches/pixel-reader-task-thread.patch'}
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
