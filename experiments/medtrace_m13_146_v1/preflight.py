"""Read-only private packet audit; output aggregate counts without clinical identities."""
import argparse
import json
from pathlib import Path
import contract


def main():
    parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();packet=json.loads(args.packet.read_text());contract.selfcheck()
    assert packet['source_recipe']==contract.SOURCE_RECIPE
    gate=contract.source_admission(packet)
    rows=contract.coverage_reservation(packet['original_tasks'],packet['base_mask'])
    import collections
    supported={t['order']:{q for e in t['events'] if e['task']=='T2L' for q in e['all_probe_query_ids'] if packet['base_mask'][q]}
               for t in packet['original_tasks']}
    result=dict(status='PREPARATION_COMPLETE_SOURCE_AND_RUNTIME_ADMISSION_PENDING',original_edits=146,
        qualified_owners=len(gate['qualified_owners']),qualified_anchors=gate['qualified_anchor_count'],
        missing_owners=len(gate['missing_owners']),exhausted_owners=len(gate['exhausted_owners']),
        current_amended_example_not_reused_as_original=True,source_gate=gate['status'],
        scope=dict(steps=160,parameters=7168,prefixes=contract.PREFIXES,methods=contract.METHODS),
        reserved_consumers=len(rows),consumer_counts=dict(collections.Counter(r['method'] for r in rows)),
        historical_T2L_eligible_edits=sum(bool(v) for v in supported.values()),
        historical_T2L_eligible_probes=sum(map(len,supported.values())),
        coverage_is_reservation_not_scientific_output=True,source_recipe=contract.SOURCE_RECIPE,
        source_weights=contract.SOURCE_WEIGHTS,source_material_admission=gate['status']=='PASS',proposed_caps=contract.CAPS,
        actual_full_training_and_bank_integration=False,training_ready=False,
        new_model_calls=0,new_Judge=0,new_downloads=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
