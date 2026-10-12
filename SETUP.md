# Setup & Reproducibility Guide

How to stand up the benchmark on a fresh machine from scratch, for reproducibility, handover,
and running additional model sweeps. Written from a verified end-to-end setup on macOS Apple
Silicon (secondary laptop, 2026-07), but the steps are OS-general; macOS-specific notes are
called out inline.

---

## 0. Architecture: what runs where

The benchmark has **two halves that run on different machines**:

| Component | Where | Notes |
|-----------|-------|-------|
| **TerminalBench task containers** | **Locally**, in Docker | Each task is a Linux (`linux/amd64`) Docker image graded by a test suite. This machine needs Docker + disk, **not** a GPU. |
| **The LLM** (the agent's brain) | **Remote** | Served over an OpenAI-compatible HTTP API: RunPod A100-80GB (for 72B) or Google Colab + ngrok (for 7B/14B/32B). The local machine only makes API calls. |

So a laptop with no GPU is fine: it orchestrates Docker containers locally and talks to a remote
LLM endpoint.

---

## 1. Hardware requirements

- **CPU/arch:** the task images are `linux/amd64`. On Apple Silicon they run **emulated (slow)**;
  a **native amd64 Linux** host is ideal for speed. Either works functionally.
- **GPU:** not needed locally; the LLM is remote.
- **Disk:** **≥ 40-50 GB free**. Prebuilt Docker images for the `core_64` subset are ~50 GB (png-generation alone is 15.2 GB and peaks at ~22 GB while building).
- **RAM:** give **Docker Desktop ≥ 8 GB** (Settings → Resources).

---

## 2. Software prerequisites

- **git**: on macOS, `xcode-select --install` (Xcode Command Line Tools).
- **Docker Desktop**: installed and the daemon **running** (verify with `docker info`).
- **Python 3.12 exactly**: `pyproject.toml` pins `requires-python = ">=3.12,<3.13"`.
  macOS: `brew install python@3.12`.
- **Prevent sleep**: the benchmark runs for **hours**; sleep kills the processes.
  macOS: run `caffeinate -i &` in the session, or disable sleep in System Settings.

---

## 3. Setup steps (in order)

### 3.1 Clone
```bash
git clone https://github.com/harrissfiris/Thesis-Code-Repo.git
cd Thesis-Code-Repo
```

### 3.2 Python environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Installs `terminal-bench`, `pydantic`, `openai`, `openpyxl`.

### 3.3 TerminalBench dataset
The runner reads task definitions from `~/.cache/terminal-bench/repo/original-tasks`.

```bash
# Registers the terminal-bench-core dataset with the tb CLI:
tb datasets download -d "terminal-bench-core==0.1.1"

# IMPORTANT: 0.1.1 only ships ~18 of the 64 tasks we use. Pull the full task set from the
# terminal-bench repo's original-tasks and place it where the runner expects it:
mkdir -p ~/.cache/terminal-bench/repo
git clone --depth=1 https://github.com/laude-institute/terminal-bench /tmp/tb-repo
cp -r /tmp/tb-repo/original-tasks ~/.cache/terminal-bench/repo/original-tasks
```
> Verify: `ls ~/.cache/terminal-bench/repo/original-tasks` should list ~90+ task folders
> (including all of `core_64`).

### 3.4 Prebuild Docker images
So the benchmark doesn't build images at runtime (which would blow the per-task budget):
```bash
bash scripts/prebuild_tasks.sh core_64      # ~30-50 min, ~50 GB disk
```
This pulls the two base images and builds `tb__<task>__client` for every task in
`scripts/task_subsets/core_64.json`. Check the **Built/Failed** summary at the end.

> Available subsets: `core_64` (canonical: 40 solvable + 24 extra), `solvable_40`, `extra_24`,
> `full_64`. Run `bash scripts/prebuild_tasks.sh` with no args to list them.

### 3.5 Reclaim build-cache disk (optional)
```bash
docker builder prune -f       # the build cache is not needed after prebuild
```

---

## 4. Serving the LLM

### Option A: RunPod (used for Qwen2.5-72B)
- **Pod:** GPU **A100-80GB**, image `vllm/vllm-openai:latest`, container disk **80 GB**,
  HTTP port **8000**, env `HF_TOKEN=<your-hf-token>`.
- **Start command:**
  ```
  --model Qwen/Qwen2.5-72B-Instruct-AWQ --host 0.0.0.0 --port 8000 \
  --quantization awq --gpu-memory-utilization 0.90 --max-model-len 32768
  ```
- **Endpoint:** `https://<POD-ID>-8000.proxy.runpod.net/v1`, key `sk-<POD-ID>`
  (the API key is literally `sk-` + the pod id).
- ⚠️ **Every time the pod dies and you restart it, the pod-id changes** → new URL **and** new key.

### Option B: Colab + ngrok (used for 7B/14B/32B and Devstral)
Serve vLLM in a Colab A100 notebook and expose it via ngrok; point `BACKEND=ngrok` at the tunnel
URL (hardcoded in `scripts/config.sh`).

---

## 5. Running the benchmark

### 5.1 Per-session environment
Set this in **every** new terminal before running:
```bash
cd Thesis-Code-Repo
source .venv/bin/activate
unset THESIS_AGENT_MAX_WALLCLOCK              # see Gotchas: a stale export overrides the budget
export BACKEND=runpod SPLIT_TASKS=1
export THESIS_LLM_API_BASE="https://<POD-ID>-8000.proxy.runpod.net/v1"
export THESIS_LLM_API_KEY="sk-<POD-ID>"
caffeinate -i &                              # macOS: prevent sleep
```

### 5.2 Sanity checks (do these before a long run)
```bash
# 1) The endpoint serves the expected model:
curl -s https://<POD-ID>-8000.proxy.runpod.net/v1/models \
  -H "Authorization: Bearer sk-<POD-ID>" | python3 -m json.tool
#    → expect "id": "Qwen/Qwen2.5-72B-Instruct-AWQ"

# 2) One task end-to-end, no connection errors:
bash scripts/switch_stage.sh g1/baseline
bash scripts/run.sh 72b single:hello-world
#    → should finish "resolved"
```

### 5.3 A full sweep
A model sweep is 13 non-baseline architectures × 3 phases (diagnostic `1-5` → `solvable_40 6-40`
→ `extra_22`), with a `switch_stage.sh` before each architecture. Each architecture is invoked as
`switch_stage <stage>` + an env var (see §6). Run the phases chained with `&&` (or `;` for
flaky-backend overnight chains so one failure doesn't abort the rest).

---

## 6. How it works (mental model)

- **`scripts/switch_stage.sh <stage>`** flips the repo's root symlinks (`agent`, `config.py`,
  `tbench_agent_v2.py`, …) to a different architecture's implementation. **Run it before every
  architecture change.** Stages: `g1/baseline`, `g2/explorer|planner|analyst|monitor`,
  `g3/verify` (+`THESIS_STAGE2_VARIANT=a|b|c|d`), `g4/dialogue` (+`THESIS_DIALOGUE_VARIANT=`),
  `g4/orchestra`. G1 ablations run on `g1/baseline` via env vars
  (`THESIS_AGENT_PROMPT_VERSION=stage0` = prompt, `THESIS_LLM_TEMPERATURE=0.2` = temp).
- **`scripts/run.sh <profile> <subset> [range]`** runs the benchmark. `profile` ∈
  `7b|14b|32b|72b|devstral|gemini|llama70b-awq`. `subset` ∈ the task_subsets. `range` (e.g.
  `6-40`) or `single:<task-id>` for one task.
- **`SPLIT_TASKS=1`** runs each task as its own `tb run`, then auto-merges the per-task run dirs
  into `runs/merged-<timestamp>/`. Always set it.
- **Budget:** 60 LLM calls / **1800 s** wall-clock per task. Do not change it.
- **Output:** results land in `runs/`. The cross-model solve matrix in `docs/task_matrix.html` was
  rendered from the full set of run outputs.

---

## 7. Gotchas (learned the hard way)

- **`THESIS_AGENT_MAX_WALLCLOCK`:** if it's exported in your shell it **overrides** the config +
  profile budget → **`unset` it before every run** (otherwise tasks may run at the wrong budget).
- **`switch_stage.sh` before every architecture**: forgetting it runs the previous architecture
  under a new label (silent mis-attribution).
- **Paths with non-ASCII characters or spaces** (this repo's main checkout lives under a Greek
  path): `rm`/`cp`/`cd` in compound bash commands can fail; use Python (`shutil`, `os`) for file
  operations on such paths.
- **iCloud + `switch_stage.sh`** creates duplicate symlinks (`agent 2`, `config 2.py`) and evicts
  file contents. Keep the repo out of iCloud; delete stray ` 2` symlinks with Python `os.remove`.
- **RunPod pod death shows up as HTTP 404/connection errors** mid-run; restart the pod, update
  the URL + key (both change), re-run the sanity check.
- **`runs/` is git-ignored** (transient, large). To move run data between machines use a
  non-git transfer (zip/scp), or temporarily un-ignore it, transfer, then restore `.gitignore`.
