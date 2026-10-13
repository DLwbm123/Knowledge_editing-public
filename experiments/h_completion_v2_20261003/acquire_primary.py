"""Retrieve selected original article evidence and related figures, through proxy."""
import html.parser
import base64
import json
import os
import re
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET

PROXY='http://127.0.0.1:7897'
OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':PROXY,'https':PROXY}))
RECEIVER=os.environ['RECEIVER_ENTRY']
REQUESTS=json.loads(os.environ['PRIMARY_REQUESTS'])
DEADLINE=float(os.environ['DEADLINE_EPOCH'])
COMPLETION_NAME=os.environ.get('COMPLETION_NAME','PRIMARY_ACQUISITION_COMPLETE.json')


class DownloadLimit(Exception):
    def __init__(self, url, size, kind):
        self.url,self.size,self.kind=url,size,kind
        super().__init__('Response exceeds per-document limit')


def send(name,data,receipt,kind='metadata'):
    header=json.dumps(dict(name=name,kind=kind,receipt=receipt)).encode()+b'\n'
    p=subprocess.run(['ssh','pro5000','python3 '+RECEIVER],input=header+data,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode()[-1500:])
    print(p.stdout.decode(),flush=True)


def cached(name,kind='metadata'):
    p=subprocess.run(['ssh','pro5000','python3 '+RECEIVER],input=json.dumps(dict(mode='read',name=name,kind=kind)).encode()+b'\n',capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode()[-1500:])
    return json.loads(p.stdout)


def fetch(name,url,cap,kind='metadata'):
    c=cached(name,kind)
    if c['exists']:return base64.b64decode(c['data']),dict(reused_existing=True,source_download_bytes=0)
    if c['remaining']<=cap:raise RuntimeError('Remaining download budget below per-document cap; do not start request')
    data,rec=get(url,cap,kind);send(name,data,rec,kind)
    return data,rec


def get(url,cap,kind='metadata'):
    if time.time()>=DEADLINE:raise RuntimeError('Original first-clock deadline reached')
    req=urllib.request.Request(url,headers={'User-Agent':'SourceEvidenceMetadataAudit/2.1'})
    with OP.open(req,timeout=35) as r:
        data=r.read(cap+1)
        if len(data)>cap:raise DownloadLimit(url,len(data),kind)
        return data,dict(source_URL=url,source_download_bytes=len(data),HTTP=r.status,
                         acquisition_epoch=time.time(),proxy=PROXY)


class Figures(html.parser.HTMLParser):
    def __init__(self):super().__init__();self.current=None;self.images={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='figure':self.current=a.get('id')
        if tag=='img' and self.current and 'src' in a:self.images[self.current]=a['src']
    def handle_endtag(self,tag):
        if tag=='figure':self.current=None


def main():
    began=time.time();receipts=[];consecutive=0
    for paper,requested_figures in REQUESTS.items():
        try:
            completed=cached(paper+'.PRIMARY_EVIDENCE.json')
            if completed['exists']:
                receipts.append(dict(paper=paper,status='REUSED_COMPLETED_PRIMARY_EVIDENCE'));continue
            xml,rec=fetch(paper+'.xml',f'https://www.ebi.ac.uk/europepmc/webservices/rest/{paper}/fullTextXML',1500000)
            a=ET.fromstring(xml);licenses=[dict(attributes=e.attrib,text=''.join(e.itertext())) for e in a.findall('.//license')]
            license_ok=any(re.search(r'creativecommons.org/licenses/by/(?!nc)|creativecommons.org/publicdomain/zero/',str(x),re.I) for x in licenses)
            figures=[]
            for f in a.findall('.//fig'):
                cap=f.find('caption');figures.append(dict(id=f.get('id'),label=''.join(f.find('label').itertext()) if f.find('label') is not None else '',
                       caption=''.join(cap.itertext()) if cap is not None else '',
                       graphic=[g.attrib for g in f.findall('.//graphic')]))
            result=dict(paper=paper,primary_article_available=True,license=licenses,
                        license_is_CC_BY_or_CC0=license_ok,figures=figures,requested_figures=requested_figures,
                        diagnosis_or_H_truth_inferred=False,images=[],patient_independence='UNKNOWN',
                        case_relation_evidence='Same paper images are one source group; actual patients require review')
            if license_ok:
                page,hr=fetch(paper+'.html',f'https://pmc.ncbi.nlm.nih.gov/articles/{paper}/',1000000)
                fp=Figures();fp.feed(page.decode())
                for figure in requested_figures:
                    match=re.search(r'_(.+)\.jpg$',figure);fid=match.group(1) if match else ''
                    options=[fid, 'fig'+fid[1:] if fid.startswith('F') else fid]
                    fid=next((s for s in options if s in fp.images),None)
                    if fid is None:
                        result['images'].append(dict(figure=figure,status='FIGURE_ID_BINDING_UNRESOLVED'));continue
                    caption=next((f['caption'] for f in figures if f['id']==fid),'')
                    if re.search(r'reproduced|reprinted|courtesy|permission|copyright',caption,re.I):
                        result['images'].append(dict(figure=figure,status='THIRD_PARTY_RIGHTS_REVIEW_REQUIRED'));continue
                    u=fp.images[fid]
                    if not u.startswith('https://cdn.ncbi.nlm.nih.gov/pmc/blobs/'):
                        result['images'].append(dict(figure=figure,status='SOURCE_IMAGE_URL_NOT_ALLOWED'));continue
                    data,ir=fetch(figure,u,10*1024*1024,kind='image')
                    result['images'].append(dict(figure=figure,primary_figure_id=fid,source_URL=u,
                                                bytes=len(data),caption=caption,status='ACQUIRED_PENDING_GLOBAL_ROLE_AND_QA_REVIEW'))
            send(paper+'.PRIMARY_EVIDENCE.json',json.dumps(result,ensure_ascii=False).encode(),
                 dict(source_download_bytes=0,acquisition_epoch=time.time()))
            receipts.append(dict(paper=paper,status='PRIMARY_ACQUIRED',image_count=sum(x['status'].startswith('ACQUIRED') for x in result['images']),license_ok=license_ok));consecutive=0
        except Exception as e:
            failure=dict(paper=paper,status='ACQUISITION_FAILED_NO_BLIND_RETRY',exception=type(e).__name__,message=str(e),epoch=time.time())
            receipt=dict(source_download_bytes=e.size if isinstance(e,DownloadLimit) else 0,acquisition_epoch=time.time(),
                         failed_transfer=True,attempted_source_URL=e.url if isinstance(e,DownloadLimit) else None)
            send(paper+'.FAILURE.'+str(time.time_ns())+'.json',json.dumps(failure).encode(),receipt,kind=e.kind if isinstance(e,DownloadLimit) else 'metadata');receipts.append(failure);consecutive+=1
            if consecutive>=3:break
    final=dict(status='PRIMARY_SOURCE_ACQUISITION_FINISHED' if len(receipts)==len(REQUESTS) else 'STOPPED_AFTER_THREE_SOURCE_ACQUISITION_FAILURES',
               requested_papers=len(REQUESTS),completed_or_failed=len(receipts),receipts=receipts,
               wall_seconds=time.time()-began,clinical_signoffs=0,H_truth_assigned=0,GPU_hours=0)
    send(COMPLETION_NAME,json.dumps(final).encode(),dict(source_download_bytes=0,acquisition_epoch=time.time()))


if __name__=='__main__':main()
