# Source24 baselines finite comparison

Use the frozen protocol in reports/source24_baselines_v1_20261011. Only pro5000 GPU5.
The inherited runtime/source assets are read-only; only first-party orchestration is published.

Set WORKSPACE to this dedicated checkout and run deploy.gpu through a neutral Python stdin entry.
Installation starts a bounded CPU dependency: preserve the two completed parent candidate matrices,
wait for its existing GPU consumers, then run actual baseline admission and TT/BalancEdit sequentially.
The separate neutral local finish entry runs finish.py through WORKSPACE/TASK_FILE environment variables.
It waits for GPU and parent scoring dependencies, launches four existing isolated scorers, writes the
joint report and pushes anonymous source-only results on this dedicated branch. No recurring automation.

Each committed training step stores optimizer/RNG and journal together. Engineering recovery must
inspect retained failures and recorded PID/start_ticks before explicit adoption; a missing successful
receipt never triggers a blind retry. Already attempted generation and Judge are never repeated.
The original U/H experiment, deadline and result archive are not modified.
