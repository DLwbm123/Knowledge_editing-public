"""Retrieve registered source materials; retrieval never grants training admission."""
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen
import xml.etree.ElementTree as ET

BUCKET = 'pmc-oa-opendata'
ASSET_LIMIT = 512 * 1024**2
FILE_LIMIT = 8 * 1024**2


def official_url(url, pmcid, version):
    p = urlsplit(url)
    prefix = f'/{pmcid}.{int(version)}/'
    if (p.scheme != 's3' or p.netloc != BUCKET or not p.path.startswith(prefix)
            or '/' in p.path[len(prefix):] or '..' in p.path):
        raise ValueError('Source object is outside its registered article version')
    return urlunsplit(('https', BUCKET + '.s3.amazonaws.com', p.path, p.query, ''))


def retrieve(metadata, registered_ids, output, media=()):
    pmcid, version = metadata['pmcid'], metadata['version']
    if pmcid not in {item.split(':', 1)[0] for item in registered_ids}:
        raise ValueError('Unregistered candidate; do not extend the candidate cap')
    if media and (not metadata.get('is_pmc_openaccess') or metadata.get('is_retracted')
                  or not metadata.get('license_code') or metadata['license_code'] == 'TDM'):
        raise ValueError('No verified open-access media permission in this metadata')
    available = {Path(urlsplit(u).path).name: u for u in metadata.get('media_urls', [])}
    if len(set(media)) != len(media) or any(name not in available for name in media):
        raise ValueError('Only explicitly listed source media may be retrieved')
    urls = [metadata['xml_url']] + [available[name] for name in media]
    # Validate every object before opening a connection or creating a file.
    urls = [official_url(u, pmcid, version) for u in urls]
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep prior source materials; use a new receipt directory')
    output.mkdir(parents=True)
    paths = []
    try:
        for url in urls:
            used = sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file())
            remaining = min(FILE_LIMIT, ASSET_LIMIT - used)
            if remaining <= 0:
                raise ValueError('Source review asset budget exhausted')
            with urlopen(url, timeout=20) as response:
                if response.geturl() != url:
                    raise ValueError('Unexpected redirect; source URL binding changed')
                data = response.read(remaining + 1)
            if len(data) > remaining:
                raise ValueError('Source object exceeds the bounded asset allowance')
            name = Path(urlsplit(url).path).name
            if name.endswith('.xml') and ET.fromstring(data).tag != 'article':
                raise ValueError('Expected a JATS article, not an HTML access page')
            path = output / name
            with path.open('xb') as stream:
                stream.write(data)
            paths.append(str(path))
    except Exception as error:
        (output / 'ACCESS_FAILURE.json').write_text(json.dumps(dict(
            error=str(error), completed_files=paths, training_admitted=False), indent=2))
        raise
    receipt = dict(pmcid=pmcid, version=version, license_code=metadata['license_code'],
                   files=paths, urls=urls, training_admitted=False)
    (output / 'RETRIEVAL.json').write_text(json.dumps(receipt, indent=2))
    return receipt
