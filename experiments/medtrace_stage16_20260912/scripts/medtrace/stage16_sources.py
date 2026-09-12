#!/usr/bin/env python3
"""Exact PMC-VQA coverage recovery; metadata/range reads, no model-based selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import urllib.parse
import urllib.request
import zipfile
import zlib

ROOT = Path(os.environ.get('MEDTRACE_BASELINE_CODE', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from scripts.medtrace.stage15_sources import read, write

REVISION = 'b56ae594f794867893143b337b4118a835794647'
URL = 'https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/'+REVISION+'/'
ARCHIVES = {'images_2.zip': 2206255503, 'images.zip': 18945102275}


class RemoteZip(io.RawIOBase):
    """Seekable range reader for stdlib ZIP metadata, with strict response bounds."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.bytes_read = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('negative offset')
        return self.pos

    def read(self, size=-1):
        size = self.size-self.pos if size < 0 else min(size, self.size-self.pos)
        if size <= 0: return b''
        if size > 64*1024*1024: raise ValueError('Unexpected large range; no whole archive fallback')
        start = self.pos; end = start+size-1
        req = urllib.request.Request(self.url+'?download=true&range='+str(start)+'-'+str(end),
                                     headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server did not return the exact requested byte range')
            data = response.read(size+1)
        if len(data) != size: raise ValueError('Truncated or oversized range')
        self.pos += size; self.bytes_read += size
        return data


def manifest(old, run):
    tasks = read(old/'private/QUEUE.json'); missing = []; controls = []
    for task in tasks:
        for p in task['probes']:
            if p['status'] == 'UNSUPPORTED_IMAGE_MISSING':
                missing.append(dict(edit=task['order'], probe=p['probe_id'], metric=p['metric'], source_ref=p['source_image']))
            elif p['metric'] == 'I_Locality' and len(controls) < 1:
                controls.append(dict(source_ref=p['source_image'], sha256=p['image_sha256']))
    write(run/'private/RECOVERY_MANIFEST.json', dict(missing=missing, existing_identity_controls=controls,
        author_revision=REVISION, matching='Exact full original basename inside author archives; preserve original image/crop'))
    print('SOURCE_MANIFEST', len(missing), flush=True)


def member_bytes(url, archive_size, info):
    remote = RemoteZip(url, archive_size); remote.seek(info.header_offset)
    header = remote.read(30)
    fields = struct.unpack('<4s5H3I2H', header)
    if fields[0] != b'PK\x03\x04' or fields[2]&1: raise ValueError('Invalid or encrypted member')
    name_length, extra_length = fields[-2:]
    remote.seek(info.header_offset+30+name_length+extra_length)
    compressed = remote.read(info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED: data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED: data = zlib.decompress(compressed, -15)
    else: raise ValueError('Unsupported ZIP compression; no archive mutation')
    # ZIP's native CRC check, not an additional transfer hash pass.
    if len(data) != info.file_size or zlib.crc32(data)&0xffffffff != info.CRC:
        raise ValueError('ZIP member integrity failure')
    return data, remote.bytes_read


def recover(manifest_path, destination):
    m = read(manifest_path); destination.mkdir(parents=True, exist_ok=True)
    required = {Path(r['source_ref']).name:r['source_ref'] for r in m['missing'] if r['metric'] == 'I_Locality'}
    controls = {Path(r['source_ref']).name:r for r in m['existing_identity_controls']}
    wanted = set(required)|set(controls); bindings = {}; audit = []; bytes_read = 0
    for archive, size in ARCHIVES.items():
        if not wanted: break
        remote = RemoteZip(URL+archive, size)
        with zipfile.ZipFile(remote) as z:
            matched = {}
            for info in z.infolist():
                name = Path(info.filename).name
                if name in wanted: matched.setdefault(name, []).append(info)
            if any(len(v) != 1 for v in matched.values()): raise ValueError('Ambiguous exact author basename')
        bytes_read += remote.bytes_read
        def extract(pair):
            name, infos = pair; info = infos[0]
            data, n = member_bytes(URL+archive, size, info)
            identity = hashlib.sha256(data).hexdigest()  # Required source/input identity, computed on received bytes once.
            if name in controls:
                if identity != controls[name]['sha256']: raise ValueError('Known original identity control disagrees')
            if name in required:
                path = destination/'images'/name; path.parent.mkdir(exist_ok=True)
                if path.exists(): raise FileExistsError('Recovered image already present; inspect resume ledger')
                path.write_bytes(data)
            return name, dict(archive=archive, member=info.filename, revision=REVISION, sha256=identity,
                              source_ref=required.get(name, controls.get(name, {}).get('source_ref')), bytes=len(data)), n
        with ThreadPoolExecutor(max_workers=4) as pool:
            for name, binding, n in pool.map(extract, sorted(matched.items())):
                wanted.remove(name); bytes_read += n; audit.append(binding)
                if name in required: bindings[required[name]] = binding
                print('SOURCE_IMAGE', len(bindings), len(required), name, flush=True)
        write(destination/'RECOVERED.json', dict(bindings=bindings, identity_controls=audit,
            remaining=sorted(wanted), download_bytes=bytes_read, archive=archive))
    assert all(any(r['sha256'] == c['sha256'] and r['source_ref'] == c['source_ref'] for r in audit) for c in controls.values()), 'No original identity control verified'
    write(destination/'RECOVERED.json', dict(bindings=bindings, identity_controls=audit,
        remaining=sorted(wanted), download_bytes=bytes_read, status='COMPLETE' if not wanted else 'PARTIAL',
        license='PMC-VQA CC BY-SA; images from PMC commercial-use allowed CC0/CC BY per author card',
        training_use=False, source_recovery_only=True))
    print('RECOVERY_COMPLETE',len(bindings),bytes_read,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('manifest','recover'))
    p.add_argument('--stage15-root', type=Path); p.add_argument('--run-root', type=Path)
    p.add_argument('--manifest', type=Path); p.add_argument('--destination', type=Path)
    a = p.parse_args()
    if a.action == 'manifest': manifest(a.stage15_root, a.run_root)
    else: recover(a.manifest, a.destination)
