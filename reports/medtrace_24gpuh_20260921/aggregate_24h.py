import json,hashlib,sys
from collections import defaultdict
def score_key(s,o): return hashlib.sha256(json.dumps([s["image_sha256"],s["question"],s["reference"],o["raw_answer"]],sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def main(outputs,cache):
 rows=[json.loads(x) for x in open(outputs) if x.strip()]; scores=json.load(open(cache))["scores"]; return {"rows":len(rows),"arms":sorted({r["arm"] for r in rows}),"final_rows":{a:sum(r["arm"]==a and r["prefix"]==19 for r in rows) for a in sorted({r["arm"] for r in rows})},"scored":sum(score_key(r["source"],r["output"]) in scores for r in rows)}
if __name__=="__main__": print(json.dumps(main(sys.argv[1],sys.argv[2]),ensure_ascii=False,indent=2))
