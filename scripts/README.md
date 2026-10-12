# Benchmark harness

The scripts that run an architecture over TerminalBench tasks. The task containers run locally
in Docker; the LLM is served remotely over an OpenAI-compatible API. Full setup is in
[`../SETUP.md`](../SETUP.md).

## Run an architecture

```bash
export BACKEND=runpod
export THESIS_LLM_API_BASE="https://<your-pod-id>-8000.proxy.runpod.net/v1"

bash scripts/switch_stage.sh g2/explorer     # select the architecture
bash scripts/run.sh 32b core_64              # run it on the 64-task subset
```

- `switch_stage.sh <family>/<variant>` points the root runtime symlinks (`agent/`, `llm/`,
  `terminal/`, …) at one system under [`../systems/`](../systems/).
- `run.sh <model> <subset>` runs the active architecture. `<model>` is `7b | 14b | 32b | 72b`;
  `<subset>` is a file in [`task_subsets/`](task_subsets/) (`core_64`, `probe_medium_20`,
  `probe_hard_20`) or `single:<task-id>` for one task.
- `config.sh` resolves the LLM endpoint from `BACKEND` (`ngrok` or `runpod`); edit the
  placeholders there or export `THESIS_LLM_API_BASE` directly.
- `prebuild_tasks.sh <subset>` builds the Docker images ahead of time so the benchmark does not
  build at runtime.

## G1 ablations (no code changes)

The G1 family runs entirely on `g1/baseline`, selected by environment variables:

| Variable | Default | Values | Ablation |
|----------|---------|--------|----------|
| `THESIS_LLM_TEMPERATURE` | `0.0` | `0.2`, `0.6` | sampling temperature |
| `THESIS_LLM_TOOL_CALLING` | `prompt` | `native` | tool-calling mode |
| `THESIS_AGENT_MAX_STEPS` | per model | `32`, `128` | step budget |
| `THESIS_AGENT_PROMPT_VERSION` | `v1` | `stage0` | enhanced prompt (grounding + verify) |

```bash
THESIS_LLM_TEMPERATURE=0.2 bash scripts/run.sh 32b core_64
```

## Task metadata

- `task_subsets/*.json`: the task subsets used in the study.
- `task_meta.json`, `task_census.json`: per-task metadata (category, reference solution length).
