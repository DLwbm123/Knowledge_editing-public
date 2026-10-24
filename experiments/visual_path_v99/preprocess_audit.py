"""CPU replay of the observed official one-pixel padding difference."""
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch


def main(root):
    root=Path(root);env=json.loads((root/'ENV.json').read_text())
    sys.path[:0]=[str(root/'pinned_dependencies'),env['M3BENCH_LLAVA_SOURCE']]
    import torch
    import transformers
    from transformers import CLIPImageProcessor
    from llava.mm_utils import process_images
    from PIL import Image
    from types import SimpleNamespace
    assert transformers.__version__=='4.36.2'
    processor=CLIPImageProcessor.from_pretrained(env['M3BENCH_VISION_PATH'],local_files_only=True)
    config=SimpleNamespace(image_aspect_ratio='pad')
    old=torch.load(root/'C_INPUTS.pt',map_location='cpu',weights_only=True)
    seeds={x['qid']:x['seed'] for x in json.loads((root/'CPU_PREPROCESS_AUDIT.json').read_text())['comparisons']}
    rows=json.loads((root/'CASES_FROZEN.json').read_text())
    parity={route:json.loads((root/(route+'_INPUT_PARITY.json')).read_text()) for route in ['N','R']}
    result=[]
    for row in rows:
        key=str(row['qid']);seed=seeds[row['qid']]
        with Image.open(row['image_path']) as original:
            image=original.convert('RGB');width,height=image.size
        random.seed(seed);aligned=process_images([image],processor,config).half()
        assert torch.equal(aligned,old[key]['pixels'])
        midpoint=abs(width-height)//2
        random.seed(seed);current_offset=random.randint(midpoint,midpoint+1) if width!=height else None
        for route in ['N','R']:
            matched=parity[route][key]['pixels_equal']
            offset=current_offset if matched else 2*midpoint+1-current_offset
            with patch('llava.mm_utils.random.randint',return_value=offset):
                replay=process_images([image],processor,config).half()
            delta=float((replay-old[key]['pixels']).abs().max())
            expected=parity[route][key]['pixel_max_abs_difference']
            assert delta==expected,(route,key,delta,expected)
            result.append(dict(qid=row['qid'],route=route,axis='x' if height>width else 'y' if width>height else None,
                current_padding_offset=current_offset,native_padding_offset=offset,
                native_vs_current_max_abs_difference=delta,replayed_difference_matches=True))
    (root/'NATIVE_PADDING_REPLAY.json').write_text(json.dumps(dict(transformers=transformers.__version__,
        same_seed_pixel_exact=8,source='Observed native inputs retained; no generation repeated',comparisons=result),indent=2))
    print('CPU padding replay verified',len(result),'path/image combinations')


if __name__=='__main__':main(sys.argv[1])
