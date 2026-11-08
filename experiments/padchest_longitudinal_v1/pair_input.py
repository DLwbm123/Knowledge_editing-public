"""Task-local dual-image extension; never changes the frozen single-image runners."""
import hashlib
import inspect
import json
from pathlib import Path
from dataclasses import replace

HEADER='Images in chronological order (first: earlier; second: later):\n<image><image>\n'


def prompt(adapter,question,answer=None):
    from llava.conversation import conv_templates
    assert not adapter.model.config.mm_use_im_start_end
    conv=conv_templates[adapter.conversation_mode].copy()
    conv.append_message(conv.roles[0],HEADER+question)
    conv.append_message(conv.roles[1],answer)
    return conv.get_prompt()


def install():
    import torch
    from PIL import Image
    from m3bench_repro.inference.llava_med import LlavaMedAdapter
    import m3bench_repro.editors.llava_runtime as mod
    from m3bench_repro.inference.preprocessing import deterministic_process_image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    assert not getattr(LlavaMedAdapter,'_longitudinal_installed',False)
    original_black=mod.LlavaMedEditorRuntime.make_black_image
    def prepare(self,image_path,question,answer=None):
        pair=json.loads(Path(image_path).read_text());assert pair['schema']=='ORDERED_CXR_PAIR_V1'
        assert len(pair['paths'])==2 and pair['paths'][0]!=pair['paths'][1]
        rendered=prompt(self,question,answer)
        ids=tokenizer_image_token(rendered,self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(self.device)
        positions=torch.where(ids[0]==IMAGE_TOKEN_INDEX)[0]
        assert len(positions)==2 and int(positions[1]-positions[0])==1
        processed=[deterministic_process_image(p,self.image_processor,self.model.config) for p in pair['paths']]
        assert all(isinstance(x[0],torch.Tensor) and x[0].shape[0]==1 for x in processed)
        pixels=torch.cat([x[0] for x in processed],dim=0).to(self.device,dtype=next(self.model.parameters()).dtype)
        digest=hashlib.sha256(json.dumps(dict(schema=pair['schema'],ordered_images=[x[2] for x in processed]),sort_keys=True).encode()).hexdigest()
        return dict(prompt=rendered,input_ids=ids,attention_mask=torch.ones_like(ids),images=pixels,
            preprocessing_seed=processed[0][1],image_sha256=digest,ordered_image_hashes=[x[2] for x in processed],
            ordered_preprocessing_seeds=[x[1] for x in processed])
    def span(ids,attention,expanded_attention,expanded_length,*,image_token_index):
        valid=ids[0,attention[0].bool()];pos=torch.where(valid==image_token_index)[0]
        assert len(pos)==2 and int(pos[1]-pos[0])==1
        mask=expanded_attention[0].bool() if expanded_attention is not None else torch.ones(expanded_length,dtype=torch.bool,device=ids.device)
        positions=torch.where(mask)[0];n=len(positions)-(len(valid)-2)
        assert n>0 and n%2==0
        start=int(positions[0])+int(pos[0]);end=start+n
        assert end<=expanded_length and mask[start:end].all()
        return start,end
    def black(self,record,output_dir):
        pair=json.loads(Path(record.image_path).read_text());paths=[]
        for i,path in enumerate(pair['paths']):
            rec=replace(record,record_id=record.record_id+f'_time{i}',image_path=Path(path))
            paths.append(str(original_black(self,rec,output_dir)))
        dest=Path(output_dir)/(hashlib.sha256(record.record_id.encode()).hexdigest()[:24]+'.pair.json')
        dest.write_text(json.dumps(dict(schema='ORDERED_CXR_PAIR_V1',paths=paths)))
        return dest
    # These methods have a single-image count guard; their target-only label
    # construction and multimodal expansion are otherwise unchanged.
    import textwrap
    for name in ['build_edit_batch','build_question_batch']:
        source=textwrap.dedent(inspect.getsource(getattr(mod.LlavaMedEditorRuntime,name)))
        assert source.count('if image_count != 1:')==1
        source=source.replace('if image_count != 1:','if image_count != 2:').replace('exactly one image token','exactly two image tokens')
        space={};exec(compile(source,'<task-local-dual-image>','exec'),mod.__dict__,space)
        setattr(mod.LlavaMedEditorRuntime,name,space[name])
    mod.expanded_image_span=span
    mod.LlavaMedEditorRuntime.make_black_image=black
    LlavaMedAdapter._prompt=prompt;LlavaMedAdapter.prepare_inputs=prepare
    LlavaMedAdapter._longitudinal_installed=True
