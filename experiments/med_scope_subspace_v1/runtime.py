"""Bind the stored LLaVA-Med runtime without changing earlier experiment entrypoints."""
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import torch


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


class FrozenRuntime:
    def __init__(self, run, contract):
        self.run = run
        self.contract = contract
        self.outputs = Path(contract['outputs'])
        self.deadline = contract['deadline']
        self.max_forwards = contract['max_forwards']
        self.max_generations = contract['max_generations']
        self.forwards = self.generations = 0
        self.base_outputs = {}
        self.generated_base = set()
        self.allow_base_generation = False
        self.device = torch.device('cuda:0')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose precisely one GPU to a model worker')
        cap = contract['memory_bytes']
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        parent = self.outputs / 'medtrace-curriculum146-v1-20261017/run/private'
        mdu_tools = self.outputs / 'med-mdu150-v1-20261105/run/MDU/tools'
        sys.path[:0] = [str(p) for p in (mdu_tools,self.outputs/'med-fixed-depth-v1-20261030/run/tools',
                        parent/'tools',parent/'source',Path(os.environ['M3BENCH_LLAVA_SOURCE']))]
        from llava.model import LlavaMistralForCausalLM
        from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything
        from m3bench_repro.editors.routing import MemoryRouter
        from methods.medtrace.core import MedTraceLayerHook
        from scripts.editor_paperspec_formal import assert_official_llavamed_source, load_runtime
        from structures import TT4
        from train_pair import state_digest

        self.record_type = EditorRecord
        self.expert_type = TT4
        self.hook_type = MedTraceLayerHook
        self.seed = seed_everything
        self.state_digest = state_digest
        assert_official_llavamed_source()
        original = LlavaMistralForCausalLM.from_pretrained
        old = LlavaMistralForCausalLM.__dict__.get('from_pretrained')
        @classmethod
        def on_cpu(cls, name, *args, **kwargs):
            kwargs.update(low_cpu_mem_usage=True, device_map={'':'cpu'})
            return original(name, *args, **kwargs)
        LlavaMistralForCausalLM.from_pretrained = on_cpu
        try:
            self.runtime = load_runtime(parent/'cpu_gate', str(self.device))
        finally:
            if old is None:
                del LlavaMistralForCausalLM.from_pretrained
            else:
                LlavaMistralForCausalLM.from_pretrained = old
        self.runtime.run_root = run
        self.model = self.runtime.model.eval().half().to(self.device)
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Backbone must be frozen')
        self.layer = self.runtime.get_module('model.layers.21.mlp.down_proj')
        self.prefill = (self.model.model.embed_tokens,self.model.model.mm_projector,self.model.model.get_vision_tower())
        self.eos = self.runtime.adapter.tokenizer.eos_token_id
        self.bank = load(self.outputs/'med150-v1-20261027/run/retained/MED150.pt')
        self.by_order = {t['order']:t for t in self.bank['tasks']}
        self.by_edit = {v['edit_id']:k for k,v in self.by_order.items()}
        self.r0 = MemoryRouter.from_state({'distance':'euclidean','entries':self.bank['R0_entries']},device=self.device)
        self.routing_layer = self.runtime.target_lock['balancedit']['targets'][0]
        if self.runtime.generation_config.get('do_sample') is not False or self.runtime.generation_config.get('max_new_tokens') != 1024:
            raise ValueError('Generation contract differs from Stage 1')
        self.model.float()
        print('MODEL_READY',flush=True)

    def budget(self):
        if time.time() >= self.deadline or self.forwards > self.max_forwards or self.generations > self.max_generations:
            raise TimeoutError('Frozen run budget exhausted')

    def forward(self, batch):
        self.forwards += 1
        self.budget()
        return self.model(**{k:v for k,v in batch.items() if k!='labels'}).logits

    @contextmanager
    def tokenize_images(self):
        for m in self.prefill:
            m.half()
        try:
            yield
        finally:
            for m in self.prefill:
                m.float()

    def record(self, row):
        task = self.by_order.get(row.get('owner'))
        edit = task['edit_id'] if task is not None else row['id']
        image = Path(row['image_path'])
        if not image.is_file() and '/DataP/' in str(image):
            image = Path(os.environ['FROZEN_IMAGE_ROOT']) / str(image).split('/DataP/',1)[1]
        return self.record_type(edit,row['dataset'],row['question'],row.get('answer',''),row['question'],
                                image,row.get('original_image_path',str(image)),row.get('owner') or 0,'SCOPE')

    @staticmethod
    def digest(value):
        return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                 allow_nan=False,separators=(',',':')).encode()).hexdigest()

    def base_binding(self, row):
        return self.digest([row['id'],row['question'],row['image_sha256'],row.get('owner'),
                            self.runtime.generation_config,os.environ.get('M3BENCH_MODEL_PATH')])

    def base_answer(self, row):
        key = row['id']
        if key not in self.base_outputs:
            if not self.allow_base_generation:
                raise ValueError(f'Missing frozen Base continuation for {key}')
            if row.get('owner') is not None:
                self.seed(self.by_order[row['owner']]['seed'])
            self.base_outputs[key] = {'binding':self.base_binding(row),'answer':self.generate(row)}
            self.generated_base.add(key)
        return self.base_outputs[key]['answer']

    def load_prefixes(self, files, rows):
        indexed = {r['id']:r for r in rows}
        for path in files:
            pack = json.loads(Path(path).read_text())
            body = pack['outputs']
            if self.digest(body)!=pack['digest']:
                raise ValueError('Base-prefix payload digest differs')
            for key,value in body.items():
                if key not in indexed or value['binding']!=self.base_binding(indexed[key]):
                    raise ValueError('Base-prefix event or runtime binding differs')
                if key in self.base_outputs and self.base_outputs[key]!=value:
                    raise ValueError('Conflicting frozen Base continuations')
                self.base_outputs[key] = value

    def generated_prefixes(self):
        body = {k:self.base_outputs[k] for k in sorted(self.generated_base)}
        return dict(digest=self.digest(body),outputs=body)

    @torch.no_grad()
    def teacher_batch(self, row, prefix):
        self.budget()
        rec = self.record(row)
        if prefix == 'reference':
            with self.tokenize_images():
                batch = self.runtime.build_edit_batch(rec)
            if batch.image_sha256 != row['image_sha256']:
                raise ValueError('Reference image binding differs')
            answer = batch.forward_kwargs()
            answer['inputs_embeds'] = answer['inputs_embeds'].float()
            return answer
        tokens = row.get('base_tokens')
        if tokens is None:
            result = self.base_answer(row)
            if result['at_cap'] or not result['tokens']:
                raise ValueError('Unusable frozen Base prefix. Inspect the prefix preflight receipt.')
            tokens = result['tokens']
        with self.tokenize_images():
            raw = self.runtime.adapter.prepare_inputs(rec.image_path,rec.question,None)
            if raw['image_sha256'] != row['image_sha256']:
                raise ValueError('Base-prefix image binding differs')
            labels = torch.tensor([tokens],device=self.device,dtype=torch.long)
            if not 0 < labels.numel() <= 1024 or int(labels.min()) < 0:
                raise ValueError('Invalid continuation-only sequence')
            full = torch.cat((raw['input_ids'], labels),-1)
            attention = torch.cat((raw['attention_mask'],torch.ones_like(labels)),-1)
            labels = torch.cat((torch.full_like(raw['input_ids'],-100),labels),-1)
            emb,att,pos,expanded = self.runtime._expand_multimodal(
                raw_input_ids=full,attention_mask=attention,labels=labels,images=raw['images'])
        if not torch.equal(expanded[expanded!=-100], labels[labels!=-100]):
            raise ValueError('Continuation token identity changed after image expansion')
        return dict(inputs_embeds=emb.float(),attention_mask=att,position_ids=pos,labels=expanded,
                    use_cache=False,return_dict=True)

    @torch.no_grad()
    def frozen_routes(self, rows):
        result = {}
        self.model.half()
        try:
            for r in rows:
                self.budget()
                batch = self.runtime.build_question_batch(self.record(r))
                if batch.image_sha256 != r['image_sha256']:
                    raise ValueError('Routing image binding differs')
                key = self.runtime.extract_layer_input_key(batch,module_path=self.routing_layer,pooling='mean')
                decision = self.r0.route(key.to(self.device))
                result[r['id']] = self.by_edit[decision.logical_edit_id] if decision.activated else None
        finally:
            self.model.float()
        return result

    @torch.no_grad()
    def generate(self, row, expert=None):
        self.generations += 1
        self.budget()
        rec = self.record(row)
        model = self.runtime.llava_model()
        with self.tokenize_images():
            raw = self.runtime.adapter.prepare_inputs(rec.image_path,rec.question,None)
            if raw['image_sha256'] != row['image_sha256']:
                raise ValueError('Generation image binding differs')
            expanded = model.prepare_inputs_labels_for_multimodal(
                raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
        original = model.prepare_inputs_labels_for_multimodal
        calls = 0
        def reuse(*args, **kwargs):
            nonlocal calls
            if args[0] is not None and torch.equal(args[0],raw['input_ids']):
                calls += 1
                return tuple(x.float() if isinstance(x,torch.Tensor) and x.is_floating_point() else x
                             for x in expanded)
            return original(*args, **kwargs)
        hook = self.hook_type(self.layer, expert) if expert is not None else None
        model.prepare_inputs_labels_for_multimodal = reuse
        if hook is not None:
            hook.attach()
        try:
            with (hook.generation_request() if hook is not None else nullcontext()):
                answer = self.runtime.adapter.generate_prepared_with_result(raw,self.runtime.generation_config)
        finally:
            model.prepare_inputs_labels_for_multimodal = original
            if hook is not None:
                hook.detach()
        tokens = list(answer.raw_token_ids)
        if calls != 1 or not tokens:
            raise ValueError('Generation prefill contract failed')
        return dict(answer=answer.decoded_text,tokens=tokens,at_cap=len(tokens)>=1024,eos=tokens[-1]==self.eos)

    def checkpoint(self, arm, owner):
        edit = self.by_order[owner]['edit_id']
        if arm == 'MED':
            return self.bank['states'][edit]
        if arm == 'P4' or arm in {'P1','P2','P3'}:
            ck = load(Path(self.contract['p_weights']) / f'{owner}_{arm}.pt')
            if ck['edit'] != edit or ck['seed'] != self.by_order[owner]['seed'] or ck['steps'] != 160:
                raise ValueError('Phase-1 checkpoint binding differs')
            return ck['state']
        if arm == 'MDU':
            # Same three disjoint ledgers used by the frozen reference worker.
            for name in ('med-mdu-boundary-v1-20261105','med-medinit-101-v1-20261104','med-mdu150-v1-20261105'):
                root = self.outputs/name/'run/MDU'
                target = root/'weights'/f'{owner}_MDU.pt'
                if target.is_file():
                    ck = load(target)
                    entries = [row for p in root.glob('TRAIN_COMPLETE_*.json')
                               for row in json.loads(p.read_text())['results'] if row['owner'] == owner]
                    if (len(entries) != 1 or ck['steps'] != entries[0]['steps']
                            or ck['initial_digest'] != self.state_digest(self.bank['states'][edit])
                            or self.state_digest(ck['state']) != entries[0]['final_digest']):
                        raise ValueError('MDU checkpoint binding differs')
                    return ck['state']
        raise KeyError((arm,owner))

    def expert(self, owner, state=None, *, dense=False):
        seed = self.by_order[owner]['seed']
        expert = self.expert_type(seed,8,8)
        if dense or (state is not None and 'A' in state):
            from .dense_writer import DenseReadIn
            expert = DenseReadIn(expert)
        expert.to(self.device)
        if state is not None:
            expert.load_state_dict(state,strict=True)
        return expert
