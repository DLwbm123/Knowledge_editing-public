"""Acquire relevant original figures when primary evidence corrects a metadata proposal."""
import json
import os
import runpy

api=runpy.run_path(os.environ['ACQUISITION_SOURCE_ENTRY'])
results=[]
for c in json.loads(os.environ['BINDING_CORRECTIONS']):
    paper=c['paper'];record=api['cached'](paper+'.PRIMARY_EVIDENCE.json')
    d=json.loads(api['base64'].b64decode(record['data']));assert d['license_is_CC_BY_or_CC0']
    fig=next(f for f in d['figures'] if f['id']==c['primary_figure_id'])
    assert c['literal_answer_extract'].casefold() in fig['caption'].casefold()
    assert not api['re'].search(r'reproduced|reprinted|courtesy|permission|copyright',fig['caption'],api['re'].I)
    page=api['cached'](paper+'.html');parser=api['Figures']();parser.feed(api['base64'].b64decode(page['data']).decode())
    url=parser.images[c['primary_figure_id']];assert url.startswith('https://cdn.ncbi.nlm.nih.gov/pmc/blobs/')
    data,_=api['fetch'](c['corrected_figure_path'],url,10*1024*1024,'image')
    c.update(source_URL=url,caption=fig['caption'],bytes=len(data),question_or_patient_truth_assigned=False,
             status='ACQUIRED_PRIMARY_BINDING_CORRECTION_PENDING_PANEL_AND_SIX_CHECKS',
             provenance='SOURCE_DERIVED_QA; original author automatic-QA proposal retained separately')
    results.append(c)
api['send']('PRIMARY_BINDING_CORRECTIONS.json',json.dumps(results,ensure_ascii=False).encode(),dict(source_download_bytes=0))
