import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "isolated_worker"))
from worker_r2 import CPUWorkerR2, EventStore, ContractError, validate_slots

class Spy:
    def __init__(self): self.writes=0; self.steps=0; self.u_calls=0
    def write(self, slot, teacher, masked):
        self.writes+=1
        if not masked and slot.get("roles",{}).get("U_fit",{}).get("state")=="active": self.u_calls+=1
        return f"w{self.writes}"
    def execution_state_id(self, slot, version, inp): return f"state:{version}:{inp['input_fp']}"
    def judge_key(self, inp, slot, version): return f"judge:{inp['input_fp']}:{version}"
    def generate(self, inp, slot, version): return f"out:{inp['input_fp']}:{version}"
    def update(self, slot, teacher, masked):
        self.steps += 1
        if not masked and slot.get("roles",{}).get("U_fit",{}).get("state")=="active": self.u_calls += 1
        return {"u_contribution": 0 if masked else 1}

def plan(slots): return {"slots":slots}
def slot(t,e,sid,**roles): return {"t":t,"expert":e,"slot_id":sid,"roles":roles}
def role(state="inactive", input_fp=None, version=None): return {"state":state,"input_fp":input_fp,"version":version}
def run(worker, judges, teachers, **kw):
    ins=[{"input_fp":"n1"},{"input_fp":"n2"}]
    return worker.run("B1",ins,judges,teachers=teachers,**kw)

class WorkerTests(unittest.TestCase):
    def test_all_slots_duplicate_prefix_and_460_plan_shape(self):
        slots=[{"t":i+2,"expert":i+1,"slot_id":f"c{i}"} for i in range(8)] + [{"t":t,"expert":1,"slot_id":f"o{j}"} for j,t in enumerate([7,9,11,12,13,10,14,15,16,17,18,19,2,3,4])]
        g=validate_slots(slots); self.assertEqual(len(slots),23); self.assertEqual(len(g[7]),2); self.assertEqual(sum(x.get("steps",20) for x in slots),460)
    def test_invalid_zero_and_bool_rejected(self):
        with self.assertRaisesRegex(ContractError,"INVALID_SLOT_RANGE"): validate_slots([{"t":2,"expert":0}])
        with self.assertRaisesRegex(ContractError,"INVALID_SLOT_TYPES"): validate_slots([{"t":True,"expert":1}])
    def test_missing_future_teacher_blocks_before_write(self):
        s=slot(1,1,"s1",U_fit=role("active","u1","v1")); spy=Spy(); w=CPUWorkerR2(plan([s]),spy)
        with self.assertRaisesRegex(ContractError,"MISSING_EXACT_TEACHER"): run(w,{}, {},)
        self.assertEqual(spy.writes,0)
    def test_masked_skips_teacher_and_u_but_other_update_runs(self):
        s=slot(1,1,"s1",U_fit=role("masked",None,None)); spy=Spy(); w=CPUWorkerR2(plan([s]),spy)
        ins=[{"input_fp":"n1"}]; judges={}
        # precompute one output identity by running once with always-true judge from generated key
        judges={w._consumer("B1",1,"B1:p1:s1","n1","state:w1:n1"):True}
        w=CPUWorkerR2(plan([s]),spy); w.run("B1",ins,judges,teachers={}); self.assertEqual(spy.u_calls,0); self.assertEqual(spy.writes,1)
    def test_training_and_maintenance_consume_mask(self):
        active=slot(1,1,"a",U_fit=role("active","u1","v1")); masked=slot(1,1,"m",U_fit=role("masked",None,None)); spy=Spy(); w=CPUWorkerR2(plan([active]),spy)
        teachers={("u1","v1"): {"input_fp":"u1","version":"v1","available_from":1}}
        self.assertEqual(w.train_initial(active, teachers=teachers, prefix=1)["u_contribution"],1)
        self.assertEqual(w.train_initial(masked, teachers={}, prefix=1)["u_contribution"],0)
        self.assertEqual(spy.steps,2)

    def test_native_false_stops_next_write(self):
        slots=[{"t":1,"expert":1,"slot_id":"s1"},{"t":2,"expert":1,"slot_id":"s2"}]; spy=Spy(); store=EventStore(); w=CPUWorkerR2(plan(slots),spy,store)
        ins=[{"input_fp":"n1"}]; judges={w._consumer("B1",1,"B1:p1:s1","n1","state:w1:n1"):False}
        with self.assertRaisesRegex(ContractError,"NATIVE_RELEASE_BARRIER"): w.run("B1",ins,judges,teachers={})
        self.assertEqual(spy.writes,1); self.assertNotIn("B1:p2:s2",store.events)
    def test_execution_state_inconsistency_fails(self):
        class Bad(Spy):
            def generate(self, inp, slot, version): return "different"
        s={"t":1,"expert":1,"slot_id":"s1"}; spy=Bad(); store=EventStore(); w=CPUWorkerR2(plan([s]),spy,store)
        # force same state key by using duplicate input and mutate second output
        orig=spy.generate; n=[0]
        def g(inp,slot,ver): n[0]+=1; return "a" if n[0]==1 else "b"
        spy.generate=g; ins=[{"input_fp":"n1"},{"input_fp":"n1"}]
        judges={w._consumer("B1",1,"B1:p1:s1","n1","state:w1:n1"):True}
        with self.assertRaisesRegex(ContractError,"EXECUTION_STATE_INCONSISTENT"): w.run("B1",ins,judges,teachers={})
    def test_recovery_does_not_repeat_write(self):
        slots=[{"t":1,"expert":1,"slot_id":"s1"},{"t":2,"expert":1,"slot_id":"s2"}]; spy=Spy(); store=EventStore(); w=CPUWorkerR2(plan(slots),spy,store)
        # construct judges lazily for first run
        ins=[{"input_fp":"n1"}]
        judges={}
        for sid,p in [("s1",1),("s2",2)]: judges[w._consumer("B1",p,f"B1:p{p}:{sid}","n1",f"state:w{p}:n1")]=True
        with self.assertRaisesRegex(RuntimeError,"SYNTHETIC_CRASH"): w.run("B1",ins,judges,teachers={},fail_after=1)
        writes=spy.writes
        w.run("B1",ins,judges,teachers={}); self.assertEqual(spy.writes,writes+1)

if __name__=='__main__': unittest.main()
