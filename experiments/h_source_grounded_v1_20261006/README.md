# H source-grounded data build

Use `reports/h_source_grounded_v1_20261006/USER_REQUEST_ZH.txt` as the protocol.
`CONFIG.json` freezes the requested limits. Private paths are supplied in a private
copy of `ASSET_CONFIG.template.json`; do not commit that filled file.

Run CPU tests from this directory with `python -m unittest -v test_build`.
For a new private staging directory, set `RUN_ROOT`, `BUILD_CONFIG`, `ASSET_CONFIG`
and `CUDA_VISIBLE_DEVICES` (empty), then execute `build.py` via a neutral stdin or
neutral temporary launcher. Do not place project or method names in process argv.
Set `STAGING_ROOT` to that directory and `RUN_ROOT` to a new final directory, then
set `SEMANTIC_REVIEW_RULES` to the private reviewed table and execute `finalize.py` in the existing Pillow-enabled environment, also CPU-only.
No output file may be overwritten. Staging and failures are retained.

The actual delivery used the pre-portability source snapshot on the private server.
The published script only externalizes asset paths and source-specific review tables and adds explicit derived-label
and source-role guards; the source binding and finite reviewed mappings are retained.
Unlisted semantic questions never receive an invented PASS. This is a bounded
lexical/template plus individually reviewed source relation build, not embedding
inference, clinical review or an independent GPT API run.

All per-record material, blind literal facts, manifests, source registry, hashes,
coverage CSV and review queues stay private. `sealed_eval` must never be read by
training; `fit_load` enforces role and evidence checks. Grounded records remain
weak-supervision candidates; neither FIT nor EVAL is automatically admitted to an
experiment. The public report gives actual coverage and all unexecuted stages.
