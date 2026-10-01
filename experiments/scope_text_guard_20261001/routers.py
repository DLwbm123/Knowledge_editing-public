"""Exact legal question guard, with audited background collisions excluded."""
from dataclasses import replace
import unicodedata
import torch
from router_r3 import RejectRouter
from m3bench_repro.editors.routing import distances
def canon(question):
 assert isinstance(question,str)
 return ' '.join(unicodedata.normalize('NFKC',question).casefold().split())
class ScopeRouter(RejectRouter):
 def __init__(self,entries,method):
  super().__init__(1.,0.);assert method in ['R0','NEG0','TXT'];self.method=method;self.prototypes={};self.text={}
  for e in entries:
   self.add(e['edit'],e['key'],e['radius']);self.prototypes[e['edit']]=(e['positive'],e['negative']);self.text[e['edit']]=set(e.get('text_guard_questions',[]))
 def diagnostic(self,q,question=None):
  if self.method=='TXT':assert isinstance(question,str),'Text guard requires the actual input question'
  decision,extra=super().diagnostic(q)
  if self.method in ['NEG0','TXT'] and decision.activated:
   positive,negative=self.prototypes[decision.logical_edit_id];assert negative.numel(),'UNSUPPORTED_NEGATIVE_PROTO'
   dp=float(distances(positive.to(q.device),q,'euclidean').min());dn=float(distances(negative.to(q.device),q,'euclidean').min());quotient=(dn-dp)/(dn+dp+1e-12)
   protected=self.method=='TXT' and canon(question) in self.text[decision.logical_edit_id];active=quotient>=0 or protected
   decision=replace(decision,logical_edit_id=decision.logical_edit_id if active else None,activated=active)
   extra=dict(extra,d_pos=dp,d_neg=dn,q=quotient,tau=0,text_guard=protected,negative_veto=not active)
  return decision,extra
def tensors_to(entries,device):return [dict(e,**{k:e[k].to(device) for k in ['key','positive','negative']}) for e in entries]
def selfcheck():
 e=dict(edit='a',key=torch.tensor([0.,0.]),radius=2.,positive=torch.tensor([[0.,0.]]),negative=torch.tensor([[1.,0.]]),text_guard_questions=['allowed question?'])
 q=torch.tensor([.9,0.]);r=ScopeRouter([e],'TXT');state=torch.get_rng_state()
 assert not ScopeRouter([e],'NEG0').route(q).activated
 assert r.diagnostic(q,question='  ALLOWED   question? ')[0].logical_edit_id=='a'
 assert not r.diagnostic(q,question='other question?')[0].activated
 assert not r.diagnostic(torch.tensor([2.1,0.]),question='allowed question?')[0].activated
 try:r.diagnostic(q)
 except AssertionError:pass
 else:raise AssertionError('Missing input text cannot silently fall back')
 assert torch.equal(state,torch.get_rng_state())
 assert canon('ＡＢＣ?')=='abc?' and canon('abc')!='abc?'
if __name__=='__main__':selfcheck();print('PASS: exact text, no radius/fallback expansion, required input, RNG')
