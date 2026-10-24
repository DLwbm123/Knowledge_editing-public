"""CPU checks for disjoint inference and adoption without replay."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile

HERE=Path(__file__).parent

def function(file,name,scope):
    node=next(n for n in ast.parse((HERE/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),scope)
    return scope[name]

def main():
    tree=ast.parse((HERE/'bank146.py').read_text())
    evaluate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    assigned=next(n.value for n in evaluate.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='assigned' for t in n.targets))
    expression=compile(ast.Expression(assigned),'bank146.py','eval')
    for action,mode,size in [('m13_146_single','single',2956),('m13_146_bank','bank',4538)]:
        rows=[dict(mode=mode,id=i) for i in range(size)]
        groups=[eval(expression,dict(rows=rows,part=p,partitions=4,action=action)) for p in range(4)]
        ids=[r['id'] for group in groups for r in group]
        assert len(ids)==len(set(ids))==size and set(ids)==set(range(size))
    calls=[]
    def launch(file,action,part,gpu):
        calls.append((action,part,gpu));return SimpleNamespace(wait=lambda:0)
    scope=dict(os=os,shared=SimpleNamespace(launch=launch))
    stage=function('curriculum_controller.py','stage',scope)
    with patch.dict(os.environ,{}):
        stage('curriculum_generate_M13P','M13P')
        assert [v[2] for v in calls]==[0,1,3,4] and os.environ['GENERATION_PARTITIONS']=='4'
        calls.clear();stage('curriculum_train_M13P','M13P');assert [v[2] for v in calls]==[0,1]
        calls.clear();stage('curriculum_recover_M13P','M13P');assert [v[2] for v in calls]==[0,1,3,4]
    class ReachedGeneration(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        run=Path(tmp);(run/'private').mkdir()
        for p in (0,1):(run/'private'/f'EXIT_curriculum_train_M13P_{p}.json').touch()
        reads=[]
        def read(path):
            name=path.name;reads.append(name)
            if name.startswith('EXIT_'):return dict(exit_code=0)
            if name.startswith('START_'):return dict(pid=0,start_ticks='0')
            if name=='CURRICULUM_CPU_CHECK.json':return dict(status='PASS')
            if name.startswith('DELETION_'):return dict(status='DELETED')
            return dict(status='COMPLETE',save_load_exact=True,curve=[None]*540,counts={})
        calls.clear()
        def stop_stage(action,arm):
            calls.append(action);assert action=='curriculum_generate_M13P';raise ReachedGeneration
        scope=dict(os=os,RUN=run,Path=Path,read=read,write=lambda *a:None,time=SimpleNamespace(time=lambda:0),f=SimpleNamespace(admit=lambda r:None,CAPS={},METHODS=('C','M1','M13P')),stage=stop_stage)
        adopt=function('curriculum_controller.py','main',scope)
        with patch.dict(os.environ,CURRICULUM_ADOPT_TRAINING='1'):
            try:adopt()
            except ReachedGeneration:pass
            else:raise AssertionError('did not reach generation')
        assert calls==['curriculum_generate_M13P']
        assert all(f'COMPLETE_curriculum_train_M13P_{p}.json' in reads for p in (0,1))
    print('PASS: four disjoint partitions; two training workers; completed training adopted without replay')

if __name__=='__main__':main()
