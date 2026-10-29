"""Build the fixed four-editor worker from the historical V133 source.

The output is preparation only; it needs an admitted evaluator and guards.
"""
from pathlib import Path

OWNERS = (47, 117, 93, 87)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('V133 source differs from the expected integration point')
    return text.replace(old, new, 1)


def build(source):
    text = replace_once(source, '    import med133_curvature as curvature\n',
                        '    import med133_curvature as curvature\n'
                        '    import projected_direction as r2\n'
                        '    from two_direction import build as build_r2\n')
    text = replace_once(text, '                    full_blocks.extend(full);med_blocks.append(med)\n',
                        '                    full_blocks.extend(full)\n'
                        '                    chosen, reason = r2.mapped_block(full, med, audit, curvature.gram, curvature.energy, build_r2)\n'
                        '                    med_blocks.append(chosen)\n'
                        '                    audit["second_direction"] = reason\n')
    text = replace_once(text, "            caches = {name:dict(parts=[fallback,curvature.concatenate(blocks)])\n"
                        "                      for name,blocks in [('FULL_CTX',full_blocks),('MED',med_blocks)]}\n",
                        "            caches = {'FULL_CTX': dict(parts=[fallback, curvature.concatenate(full_blocks)]),\n"
                        "                      'MED': dict(parts=[fallback, r2.pack_mapped(med_blocks)])}\n")
    text = replace_once(text, "    for task in inputs['tasks']:\n",
                        "    for task in [v for v in inputs['tasks'] if v['order'] in (47, 117, 93, 87)]:\n")
    text = replace_once(text, "==23360\n    assert COUNTS['updates']>=23360\n",
                        "==640\n    assert COUNTS['updates']>=640\n")
    text = replace_once(text, '    import med133_evaluate as evaluate\n',
                        '    import med_r2_evaluate as evaluate\n')
    compile(text, 'candidate-worker', 'exec')
    return text


if __name__ == '__main__':
    import sys
    source, destination = map(Path, sys.argv[1:])
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(build(source.read_text()))
