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
EXPECTED_STOCK_SHA256 = 'ba05ec36f7f0b548b266865ea2787a1dc5a028e298ebefcf0c49196ac629c4d3'
ORIGINAL = bytes.fromhex('48 8d 54 24 70 b9 00 02 00 00 e8 2e c1 7c 00')
VARIANTS = {'normal': (767, bytes.fromhex('48 8d 54 24 70 b9 ff 02 00 00 e8 2e c1 7c 00')),
            'background': (2815, bytes.fromhex('48 8d 54 24 70 b9 ff 0a 00 00 e8 2e c1 7c 00'))}


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
    parser.add_argument('--pool', choices=['normal', 'background'], default='background')
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
    if sha(archived) != EXPECTED_STOCK_SHA256:
        raise ValueError('Patch refused: archive executable is not the verified 0.9.16 build')
    offset = file_offset(archived, INSTRUCTION_RVA - 5)
    if archived[offset:offset + len(ORIGINAL)] != ORIGINAL:
        raise ValueError('Patch refused: disassembled instruction context does not match')
    supported_hashes = {sha(archived)}
    for _, variant in VARIANTS.values():
        supported = bytearray(archived)
        supported[offset:offset + len(ORIGINAL)] = variant
        supported_hashes.add(sha(supported))
    if sha(current) not in supported_hashes:
        raise ValueError('Patch refused: not the archived build or an exactly known scheduling variant')
    if BACKUP.exists() and sha(BACKUP.read_bytes()) != sha(archived):
        raise ValueError('Existing original backup has a different hash; preserving it')
    if not BACKUP.exists():
        if sha(current) != sha(archived):
            raise ValueError('Original backup is required before changing a known variant')
        shutil.copy2(BINARY, BACKUP)
    desired_enum, desired_bytes = VARIANTS[args.pool]
    modified = bytearray(archived)
    modified[offset:offset + len(desired_bytes)] = desired_bytes
    changed = sum(a != b for a, b in zip(archived, modified))
    if changed != (1 if args.pool == 'normal' else 2):
        raise ValueError('Unexpected byte change count')
    BINARY.write_bytes(modified)
    manifest = {'active': True, 'status': 'experimental_pending_acceptance',
                'stock_sha256': sha(archived), 'patched_sha256': sha(modified),
                'previous_sha256': sha(current), 'worker_pool': args.pool,
                'changed_bytes': changed, 'instruction_rva': hex(INSTRUCTION_RVA),
                'file_offset': hex(offset + 6), 'from_task_enum': 512, 'to_task_enum': desired_enum,
                'from_name': 'HighTaskPriority (thread index 0 = StatsThread)',
                'to_name': 'AnyNormalThreadHiPriTask' if args.pool == 'normal' else 'AnyBackgroundHiPriTask',
                'backup': str(BACKUP), 'source_patch': 'patches/pixel-reader-task-thread-normal.patch' if args.pool == 'normal' else 'patches/pixel-reader-task-thread.patch'}
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
