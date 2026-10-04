# Startup status

Run `20261004R1_8` started at 2026-10-04 22:56:17 Asia/Shanghai on pro5000 GPU 7,
with a fixed two-hour deadline. Both arms run serially in one detached process.
The startup check confirmed native model loading and Base generation in progress
(12/123 queries at the captured check), no failure receipt, and 15,676 MiB GPU
memory use. Full process and GPU process names use only the neutral Python entry.

CPU dense-inverse and functional-gradient checks passed. No native edit,
end-of-sequence clean reload, semantic quality score or improvement was confirmed
at this startup checkpoint. The job is background execution; this chat does not
create a monitoring automation. See PROTOCOL.md for the fixed scope and limits.

Raw medical inputs and model outputs are excluded from publication. Judge scoring
is pending and uses the future Qwen default only after generation evidence is ready.
