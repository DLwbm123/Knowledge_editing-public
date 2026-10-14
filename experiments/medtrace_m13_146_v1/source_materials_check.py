"""Offline trust-boundary checks for bounded source retrieval."""
import copy
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
import source_materials as source


def check():
    m = dict(pmcid='PMC12345', version=1, is_pmc_openaccess=True,
             is_retracted=False, license_code='CC BY',
             xml_url='s3://pmc-oa-opendata/PMC12345.1/PMC12345.1.xml',
             media_urls=['s3://pmc-oa-opendata/PMC12345.1/figure.jpg'])
    def opened(url, **_):
        data = b'<article/>' if url.endswith('.xml') else b'fixture-image'
        response = io.BytesIO(data); response.geturl = lambda: url
        return response
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        with patch.object(source, 'urlopen', side_effect=opened) as request:
            result = source.retrieve(m, ['PMC12345:case1'], root/'good', ['figure.jpg'])
            assert len(result['files']) == 2 and not result['training_admitted']
            assert request.call_count == 2
            bad = copy.deepcopy(m); bad['license_code'] = 'TDM'
            retracted = dict(m, is_retracted=True)
            escape = dict(m, xml_url='s3://other-bucket/PMC12345.1/article.xml')
            for record, ids, media in [(m, [], []), (bad, ['PMC12345'], ['figure.jpg']),
                                       (retracted, ['PMC12345'], ['figure.jpg']),
                                       (m, ['PMC12345'], ['unknown.jpg']),
                                       (escape, ['PMC12345'], [])]:
                try:
                    source.retrieve(record, ids, root/'bad', media)
                    raise AssertionError('Invalid source accepted')
                except ValueError:
                    pass
            assert request.call_count == 2 and not (root/'bad').exists()
            with patch.object(source, 'FILE_LIMIT', 3):
                try:
                    source.retrieve(m, ['PMC12345'], root/'oversize')
                    raise AssertionError('Oversized object accepted')
                except ValueError:
                    pass
            assert (root/'oversize/ACCESS_FAILURE.json').exists()
            assert not (root/'oversize/PMC12345.1.xml').exists()
    return dict(status='PASS', unregistered_and_unlicensed_media_rejected=True,
                bound_urls_and_asset_limits=True, retrieval_does_not_admit_training=True,
                new_model_calls=0, new_Judge=0)


if __name__ == '__main__':
    print(check())
