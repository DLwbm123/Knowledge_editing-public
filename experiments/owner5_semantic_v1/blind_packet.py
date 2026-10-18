"""Prepare a fixed-order blinded packet from frozen outputs; no model calls."""
import json
import random
import sys
from pathlib import Path

run=Path(sys.argv[1])
assert json.loads((run/'COMPLETE.json').read_text())['status']=='COMPLETE'
assert json.loads((run/'EXIT.json').read_text())['code']==0
files=sorted((run/'outputs').glob('*/*.json'))
assert len(files)==27
random.Random(20261018).shuffle(files)
packet=[];mapping={}
for index,path in enumerate(files):
    value=json.loads(path.read_text());key='item_'+str(index+1).zfill(2)
    packet.append(dict(id=key,question=value['binding']['question'],reference=value['binding']['reference'],answer=value['raw_answer']))
    mapping[key]=dict(arm=path.parent.name,query=path.stem)
(run/'BLIND_PACKET.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n')
(run/'BLIND_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
print(json.dumps(packet,indent=2,ensure_ascii=False))
