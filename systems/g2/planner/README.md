# Stage 1: Explorer + Planner + Executor (Three-Phase Pipeline)

## Overview

Adds a **think-only PlannerAgent** between the Explorer and Executor. The Planner receives the task description + Data Summary from the Explorer and decomposes the task into 1-4 sequential subtasks. Each subtask runs in a **separate Executor with fresh context**, preventing context window exhaustion on complex tasks.

## Pipeline

```
Task Instruction
        ↓
  ┌──────────────┐
  │ ExplorerAgent │  ← read-only (bash/read_file/think)
  │               │     produces Data Summary JSON
  └──────┬────────┘
         │ Data Summary
         ↓
  ┌──────────────┐
  │ PlannerAgent  │  ← think-only (no side effects)
  │ (planner.py)  │     produces subtask plan JSON
  └──────┬────────┘
         │ Plan (1-4 subtasks)
         ↓
  ┌──────────────────────────────────────────┐
  │ Per-subtask Executor loop                │
  │                                          │
  │  Subtask 1 → Executor (fresh context)    │
  │       ↓ compressed result                │
  │  Subtask 2 → Executor (fresh context)    │
  │       ↓ compressed result                │
  │  Subtask N → Executor (fresh, can submit)│
  └──────────────────────────────────────────┘
         │
         ↓
     submit(answer)
```

## What Changed from Stage 0.5

### New: PlannerAgent (`agent/planner.py`)
- Thin wrapper around ReActAgent (same pattern as ExplorerAgent)
- **Tool access restricted to `think()` only**: no bash, no file ops, no submit
- Receives: task instruction + Data Summary from Explorer
- Outputs: JSON plan with 1-4 subtasks, each containing:
  - `id`: sequential number
  - `description`: what the subtask should accomplish
  - `success_criteria`: how to verify completion
  - `budget_hint`: `"small"` / `"medium"` / `"large"` (drives step allocation)
- **Fallback**: if JSON parsing fails → single-subtask plan (degrades gracefully to Stage 0.5)
- Budget: `planner_max_steps` (14B: 4, 32B: 3) / `planner_max_wallclock` (60s)

### New: Planner Prompt (`planner_v1`)
- Instructs the Planner to assess complexity first, then decompose
- **1-subtask rule**: "Output exactly 1 subtask when the task is straightforward, has a single deliverable, or can be solved in <15 steps"
- **2-4 subtasks**: "Only when the task has genuinely independent phases"
- Budget-aware: tells the Planner how many executor steps are available
- Plan must be emitted in the FIRST `think()` call

### New: Per-Subtask Execution
- Each subtask runs in a **separate Executor with fresh context**
  - System prompt + task + subtask description + Data Summary + compressed results from previous subtasks
  - Fresh context = ~28K usable tokens (vs ~10-15K in late Phase I turns)
- **Budget allocation** via `allocate_budgets()`:
  - `small` = 15%, `medium` = 25%, `large` = 45% of remaining steps
  - Single-subtask plans get 100% of remaining budget
- **Submit gating**: only the last subtask can call `submit()`
- **Result compression**: each subtask's output is compressed into a short summary injected into the next subtask's context

### New: Pipeline Config
- `pipeline` gains third option: `"explorer+planner+react"`
- `planner_max_steps`, `planner_max_wallclock_seconds`, `planner_max_subtasks` added to config

### Budget Split (32B default)
```
Total:    60 steps / 600s
Explorer:  5 steps /  90s  (adaptive, avg usage: 4.1 steps)
Planner:   3 steps /  60s  (think-only, typically 1 step)
Executor: ~52 steps / 450s (split across subtasks)
```

### Budget Split (14B default)
```
Total:    50 steps / 600s
Explorer: 10 steps / 150s  (14B doesn't early-exit well, avg 9.2 steps)
Planner:   4 steps /  60s
Executor: ~36 steps / 390s (split across subtasks)
```

## Technical Decisions

### Why Think-Only Planner?
The Planner must not have side effects. If it could run bash commands, it might accidentally modify the environment before the Executor starts. Think-only ensures the plan is pure reasoning.

### Why Fresh Context Per Subtask?
In Phase I / Stage 0.5, the single Executor's context window fills up over ~30+ steps. By Step 40, the agent operates on heavily summarized history and loses track of earlier work. Fresh context per subtask resets this, each subtask starts with full context capacity.

### Why Sequential (Not Parallel) Subtasks?
Subtasks often have data dependencies (e.g., "read the data" → "build the model" → "evaluate"). Parallel execution would require dependency analysis that's beyond what the Planner can reliably produce.

### Known Limitation: Decomposition Overhead
The planner consumes steps from the executor budget. For simple tasks, this overhead is pure loss: the task would be solved faster without decomposition. The 1-subtask rule mitigates this: simple tasks get the full executor budget as a single continuous run.

## Files Changed from Stage 0.5

| File | Change |
|------|--------|
| `agent/planner.py` | **NEW**: PlannerAgent, PlanResult, allocate_budgets(), compress_result() |
| `agent/prompts.py` | Added `planner_v1` prompt, `get_planner_system_prompt()`, `build_planner_user_message()`, `build_executor_user_message()` |
| `config.py` | Added `planner_max_steps`, `planner_max_wallclock`, `planner_max_subtasks` |
| `tbench_agent_v2.py` | Pipeline routing: `explorer+planner+react` branch with per-subtask executor loop |

## How to Run

```bash
# From project root (stage1 is default):
./scripts/switch_stage.sh stage1
./scripts/run_single.sh hello-world 32b

# Or full benchmark:
./scripts/run_benchmark.sh 32b
```
