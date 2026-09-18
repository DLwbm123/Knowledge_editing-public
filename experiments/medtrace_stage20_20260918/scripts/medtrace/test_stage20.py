"""Small CPU contracts: source admission, final-panel coverage and cumulative caps."""
import tempfile
from pathlib import Path
from scripts.medtrace.stage20_admit import self_test
from scripts.medtrace.stage20_closeout import expected,outputs,continuation_receipts
from scripts.medtrace.stage19_fasttrack_budget import remaining


def run():
    self_test()
    def row(i):return dict(image_sha256=str(i),question='q'+str(i))
    stream=dict(tasks=[dict(native=row(i)) for i in range(100)],core_rows=[row(200)],new_rows=[row(300)],positive_rows=[row(400)])
    assert len(expected(stream,19))==20
    assert len(expected(stream,19,final=True))==22
    assert len(expected(stream,50))==53
    assert len(expected(stream,11,final=True,ablation=True))==12
    assert remaining(dict(limit_seconds=28800,sessions=[dict(started_epoch=0,seconds=100),dict(started_epoch=900)]),now=1000)==28600
    assert remaining(dict(limit_seconds=28800,sessions=[dict(started_epoch=0)]),now=30000)==0
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'private').mkdir();(root/'private/OUTPUTS.jsonl').write_text('{"complete":true}\n{"partial":')
        assert outputs(root)==[dict(complete=True)]
        for branch,content in [('CP_W0','{"condition":"CP_W0"}'),('C_FACT','{"branch":"C_FACT"}')]:
            directory=root/'private/edits/e001'/branch;directory.mkdir(parents=True)
            (directory/'TRAINING.json').write_text(content)
        assert [r['branch'] for _,r in continuation_receipts(root/'private')]==['C_FACT']
    import torch
    from methods.medtrace.hsic import select_layer
    from scripts.medtrace.stage20_runtime import hsic_diagnostics
    g=torch.Generator().manual_seed(19)
    f=dict(input=torch.randn(5,8,generator=g),final=torch.randn(5,6,generator=g),layers={i:torch.randn(5,9,generator=g) for i in range(1,31)},sample_ids=list('abcde'))
    selection=select_layer(f);diagnostic=hsic_diagnostics(f,selection)
    assert len(diagnostic['rows'])==30 and max(diagnostic['rows'],key=lambda r:r['score'])['layer']==selection['layer_id']
    print('Stage20 final-panel / cumulative-budget / partial-write checks PASS')


if __name__=='__main__':run()
