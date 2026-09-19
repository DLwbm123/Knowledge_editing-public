"""CPU-only measurement of the completed B artifacts; no historical writes."""
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.medtrace_stage21_20260918.resource_profile import summarize, tensors


def run(root):
    root = Path(root)
    read = lambda p: json.loads(p.read_text())
    rows = []
    for p in sorted((root / 'private/edits').glob('e*/C_FACT/latest.pt')):
        state = torch.load(p, map_location='cpu', weights_only=True)
        assert state['step'] == 320
        training = read(p.parent / 'TRAINING.json')
        initialization = read(p.parent.parent / 'INITIALIZATION_RECEIPT.json')
        assert initialization['writer_layer'] == 31 and not initialization['reused_L30_weights']
        rows.append(dict(position=int(p.parent.parent.name[1:]),
                         file_bytes=p.stat().st_size, writer_tensor_bytes=tensors(state['expert']),
                         initialization_seconds=initialization['seconds'],
                         continuation_seconds=training['session_seconds']))
    assert [r['position'] for r in rows] == list(range(1, 46))
    outputs = [json.loads(x) for x in (root / 'private/OUTPUTS.jsonl').read_text().splitlines()]
    replay_count = sum(r['mode'] == 'integration_replay' for r in outputs)
    outputs = [r for r in outputs if r['mode'] in ('insertion', 'endpoint')]
    assert len(outputs) == 298
    result = dict(status='COMPLETE_B_CPU_PROFILE', edits=45, rows=rows, mechanical_replays_excluded=replay_count,
                  mean_file_bytes=sum(r['file_bytes'] for r in rows)/45,
                  mean_writer_tensor_bytes=sum(r['writer_tensor_bytes'] for r in rows)/45,
                  initialization_seconds=sum(r['initialization_seconds'] for r in rows),
                  continuation_seconds=sum(r['continuation_seconds'] for r in rows),
                  generation_seconds=summarize([r['output'].get('seconds') for r in outputs]),
                  routing_seconds=summarize([r.get('routing_seconds') for r in outputs]),
                  checkpoint_activation_seconds=summarize([r.get('checkpoint_load_seconds') for r in outputs]),
                  peak_gpu_allocated_bytes=read(root/'public/GENERATED.json')['peak_gpu_memory_bytes'],
                  caveat='File includes optimizer/RNG/curve. Timings are observed, not a matched latency benchmark; missing timings are NA, not zero.',
                  historical_artifacts_read_only=True)
    (root/'public/RESOURCE_PROFILE.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    import os
    run(os.environ['PROFILE_ROOT'])
