"""Extract x64 thread instruction pointers / stack candidates from a local dump.

Stack candidates are scanned return-address candidates, not a fully unwound call
stack. Exact instruction pointers and matching local PDB symbols are distinguished.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump', type=Path)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--symbolizer', type=Path, required=True)
    args = parser.parse_args()
    raw = args.dump.read_bytes()
    if raw[:4] != b'MDMP':
        raise ValueError('Invalid minidump')
    u32 = lambda offset: struct.unpack_from('<I', raw, offset)[0]
    u64 = lambda offset: struct.unpack_from('<Q', raw, offset)[0]
    streams = {}
    for i in range(u32(8)):
        kind, size, rva = struct.unpack_from('<III', raw, u32(12) + i * 12)
        streams[kind] = (rva, size)
    def string(rva):
        return raw[rva + 4:rva + 4 + u32(rva)].decode('utf-16-le', errors='replace')
    modules = []
    module_rva, _ = streams[4]
    for i in range(u32(module_rva)):
        pos = module_rva + 4 + i * 108
        modules.append({'base': u64(pos), 'size': u32(pos + 8), 'name': string(u32(pos + 20))})
    image = next(m for m in modules if Path(m['name']).name.lower() == args.binary.name.lower())
    names = {}
    if 24 in streams:
        pos, _ = streams[24]
        for i in range(u32(pos)):
            names[u32(pos + 4 + i * 12)] = string(u64(pos + 8 + i * 12))
    def image_offset(address):
        return address - image['base'] if image['base'] <= address < image['base'] + image['size'] else None
    threads, offsets = [], set()
    thread_rva, _ = streams[3]
    for i in range(u32(thread_rva)):
        pos = thread_rva + 4 + i * 48
        tid, stack_size, stack_rva, context_size, context_rva = u32(pos), u32(pos + 32), u32(pos + 36), u32(pos + 40), u32(pos + 44)
        if context_size < 256:
            continue
        rip = u64(context_rva + 248)
        module = next((m for m in modules if m['base'] <= rip < m['base'] + m['size']), None)
        candidates = []
        for cursor in range(stack_rva, stack_rva + min(stack_size, 4096), 8):
            offset = image_offset(u64(cursor))
            if offset is not None and offset not in candidates:
                candidates.append(offset)
                offsets.add(offset)
        ip_offset = image_offset(rip)
        if ip_offset is not None:
            offsets.add(ip_offset)
        threads.append({'id': tid, 'name': names.get(tid), 'ip_module': Path(module['name']).name if module else None,
                        'ip_offset': ip_offset, 'stack_candidates': candidates[:20]})
    ordered = sorted(offsets)
    symbolized = subprocess.run([str(args.symbolizer), f'--obj={args.binary.resolve()}', '--relative-address',
                                 '--no-inlines', '--output-style=JSON'],
                                input=''.join(f'{offset:#x}\n' for offset in ordered),
                                capture_output=True, text=True, check=True, timeout=60)
    symbols = {}
    for line in symbolized.stdout.splitlines():
        item = json.loads(line)
        address = int(item['Address'], 16)
        symbols[address] = [s.get('FunctionName') for s in item.get('Symbol', [])]
    for thread in threads:
        thread['instruction_symbol'] = symbols.get(thread['ip_offset'])
        thread['stack_candidates'] = [{'offset': hex(o), 'symbol': symbols.get(o)} for o in thread['stack_candidates']]
    report = {'note': 'Stack entries are scanned candidates, not an unwound stack.', 'threads': threads}
    args.dump.with_suffix('.threads.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    selected = [t for t in threads if t['name'] in ['GameThread', 'RHIThread', 'RenderThread 0', 'RenderingThread']]
    print(json.dumps(selected or threads[:5]))


if __name__ == '__main__':
    main()
