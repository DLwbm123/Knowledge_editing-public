"""Synthetic CPU admission only; inherits the frozen parent dependencies."""
import json,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['V26_ROOT']);P=Path(os.environ['PARENT_RUN'])
sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
def write(name,value):
    p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
if __name__=='__main__':
    try:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU admission alarm')));signal.alarm(180)
        with (ROOT/'CPU_ONCE').open('x'):pass
        import torch
        torch.set_num_threads(1)
        import module_run as r
        from alignment import selfcheck
        from consumers import selfcheck as consumer_check
        seed=json.loads((P/'private/M13_SOURCE_PACKET.json').read_text())['original_tasks'][4]['seed']
        from attributes import selfcheck as attribute_check,token_groups,weights_for
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(os.environ['M3BENCH_MODEL_PATH'],use_fast=False,local_files_only=True)
        from types import SimpleNamespace
        from m3bench_repro.inference.llava_med import LlavaMedAdapter
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        model_config=json.loads((Path(os.environ['M3BENCH_MODEL_PATH'])/'config.json').read_text())
        adapter=LlavaMedAdapter(os.environ['M3BENCH_MODEL_PATH'],'unused',device='cpu')
        adapter.model=SimpleNamespace(config=SimpleNamespace(mm_use_im_start_end=model_config.get('mm_use_im_start_end',False)))
        units=json.loads((ROOT/'SOURCE_UNITS.json').read_text());mapping={}
        for key,spec in units.items():
            full=tokenizer_image_token(adapter._prompt(spec['source']['question'],spec['source']['reference']),tokenizer,IMAGE_TOKEN_INDEX)
            prefix=tokenizer_image_token(adapter._prompt(spec['source']['question']),tokenizer,IMAGE_TOKEN_INDEX)
            assert full[:len(prefix)]==prefix
            ids=full[len(prefix):]
            assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).lstrip()==spec['source']['reference']+' '
            groups=token_groups(tokenizer,ids,spec['source']['reference'],spec['spans'])
            weights=weights_for(groups);mapping[key]=dict(groups=groups,weights=weights.tolist(),target_ids=ids,template_separator_kept_native_mass=True)
        write('CPU_TOKEN_MAPPING.json',mapping)
        result=selfcheck(r,seed);result['attributes']=attribute_check();result['source_token_mapping_PASS']=True;result['consumers']=consumer_check();result['time']=time.time()
        write('CPU_ADMISSION.json',result)
    except BaseException:
        write('CPU_FAILURE.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
