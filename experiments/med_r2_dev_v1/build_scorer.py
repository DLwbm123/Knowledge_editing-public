"""Prepare a bounded one-stage scorer from the existing Astra contract."""
from pathlib import Path
from build_worker import replace_once


def build(source):
    text = replace_once(source, '    import med133_report as report\n', '    import med_r2_report as report\n')
    text = replace_once(text, 'max_attempt_items=22482,workers=4', "max_attempt_items=cfg['max_attempt_items'],workers=4")
    text = replace_once(text, "    for stage in ('A','B'):\n", "    for stage in ('A',):\n")
    start = text.index("        if stage=='A' and not (local/'CANONICAL_BACKUP_RECEIPT.json').exists():\n")
    end = text.index("        packetfile=local/f'PACKET_{stage}.json'\n", start)
    text = text[:start] + text[end:]
    text = replace_once(text, 'import med133_score_packet as score_packet', 'import med_r2_score_packet as score_packet')
    text = replace_once(text, "        consumers=packet.pop('consumers');payloads={}\n",
                        "        consumers=packet.pop('consumers');payloads={}\n"
                        "        assert len(consumers)==3*cfg['single_consumers']\n"
                        "        assert all(v['payload_key'] in decisions and v['payload_key'] not in missing for v in consumers if v['arm'] in ('LEGACY','BASE'))\n")
    text = replace_once(text, "generated_consumers=2956 if stage=='A' else 7494", "generated_consumers=cfg['single_consumers']")
    start = text.index("        meta=remote(prefix+'''\n")
    end = text.index('        cost.update(meta)\n', start) + len('        cost.update(meta)\n')
    text = text[:start] + text[end:]
    text = replace_once(text, '            single_consumers=2956,unknown_consumers=',
                        "            single_consumers=cfg['single_consumers'],unknown_consumers=")
    text = replace_once(text, "        reportname='STAGE_A_REPORT_COMPLETE.json' if stage=='A' else 'FINAL_REPORT_COMPLETE.json'\n",
                        "        reportname='FINAL_REPORT_COMPLETE.json'\n")
    text = replace_once(text, "'reports/med146_single_sequential_v133_20261027'", "'reports/med_r2_dev_v1_20261029'")
    compile(text, 'candidate-scorer', 'exec')
    return text


if __name__ == '__main__':
    import sys
    source, destination = map(Path, sys.argv[1:])
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(build(source.read_text()))
