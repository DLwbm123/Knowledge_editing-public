"""Parameterize only the old/new KL weights in the isolated runtime copy."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(os.environ['RUN_ROOT'])
def main():
 p=ROOT/'worker_r2.py';s=p.read_text();before=hashlib.sha256(s.encode()).hexdigest()
 replacements={
  'def make_protection(runtime, task, run, expert):':'def make_protection(runtime, task, run, expert, beta=.5):\n    assert beta in (0., .125, .25, .5)',
  "if task['order'] <= 2:":"if task['order'] <= 2 and beta == .5:",
  '(.005*value/len(old_u)).backward()':'(.01*(1-beta)*value/len(old_u)).backward()',
  '(.005*new_loss).backward()':'(.01*beta*new_loss).backward()',
  'return .5*sum(old_losses)/len(old_losses)+.5*float(new_loss.detach())':'return (1-beta)*sum(old_losses)/len(old_losses)+beta*float(new_loss.detach())',
 }
 for a,b in replacements.items():assert s.count(a)==1,a;s=s.replace(a,b)
 p.write_text(s);receipt=dict(original_sha256=before,patched_sha256=hashlib.sha256(s.encode()).hexdigest(),changes='beta only; lambda_U=.01; stateless per-step sampler unchanged',historical_source_untouched=True)
 (ROOT/'PROTECTION_PATCH.json').write_text(json.dumps(receipt,indent=2));print(receipt)
if __name__=='__main__':main()
