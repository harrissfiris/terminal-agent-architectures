# Stage 0.5: Explorer + Executor (Two-Phase Pipeline)

## Overview

First architectural change: a **read-only ExplorerAgent** runs before the main Executor. The Explorer inspects every file in the task directory and produces a structured **Data Summary JSON**, which is injected into the Executor's context. This replaces the fragile prompt-level data-grounding directive from Stage 0 with an architectural guarantee.

## Pipeline

```
Task Instruction
        ↓
  ┌──────────────┐
  │ ExplorerAgent │  ← read-only tools only (bash/read_file/think)
  │ (explorer.py) │     produces Data Summary JSON
  └──────┬────────┘
         │ Data Summary
         ↓
  ┌──────────────┐
  │ ReActAgent   │  ← full tools, stage0 prompt + Data Summary injected
  │ (Executor)   │
  └──────┬────────┘
         │
         ↓
     submit(answer)
```

## What Changed from Stage 0

### New: ExplorerAgent (`agent/explorer.py`)
- Thin wrapper around ReActAgent with restricted config
- **Tool access restricted** to `[bash, read_file, think]`
  - Enforced at two levels: (1) only those schemas passed to LLM, (2) executor rejects non-whitelisted bash commands
- **Read-only bash whitelist**: `ls`, `cat`, `head`, `tail`, `wc`, `file`, `find`, `stat`, `du`, `tree`, `xxd`, `hexdump`
- Runs for `explorer_max_steps` steps (default: 10) or `explorer_max_wallclock` seconds
- Output: structured **Data Summary JSON** containing:
  - Files found (path, type, size)
  - Column names and data types (for CSV/JSON/JSONL)
  - Sample data (first/last rows)
  - Observations and quirks
- **Failure is non-fatal**: if Explorer produces malformed JSON or crashes, Executor still runs without Data Summary

### New: Explorer Prompt (`explorer_v1`)
- System prompt instructs: "You are a data-exploration assistant. Inspect the environment and produce a structured Data Summary JSON."
- **Early-exit directive**: if the directory contains only code files (no data files), produce a minimal summary immediately and stop
- Data Summary must be output in the final `think()` call

### New: Data Summary Injection
- `build_initial_user_message()` gains `data_summary` parameter
- If Explorer produced valid Data Summary, it's injected as a separate section in the Executor's task message
- Executor sees structured file/column/sample info before writing any code

### New: Pipeline Config
- `config.py` adds `pipeline` field: `"single"` | `"explorer+react"`
- `explorer_max_steps` and `explorer_max_wallclock_seconds` control Explorer budget
- Set via `THESIS_AGENT_PIPELINE=explorer+react`

### Budget Split
```
Total:    50 steps / 600s  (14B default)
Explorer: 10 steps / ~150s
Executor: 40 steps / ~450s (continuous, single agent)
```

### What Did NOT Change from Stage 0
- ReAct loop logic (`react.py`): identical
- Executor tools: same 6 tools for the main agent
- Memory: same sliding window
- Safety: same CommandGuard
- Tool calling: still `"prompt"` mode

## Technical Decisions

### Why a Separate Agent (not just prompt)?
Stage 0 showed that prompt-level "inspect first" directives are unreliable: models often skip data inspection under budget pressure. The ExplorerAgent **guarantees** data grounding architecturally: the Executor cannot start until the Explorer has run.

### Why Read-Only?
Prevents the Explorer from accidentally modifying the task environment. If it could write files or run arbitrary commands, Explorer bugs could corrupt the task state before the Executor starts.

### Why Fresh Context for Executor?
The Executor gets a clean context with only: system prompt + task + Data Summary. No Explorer conversation history carried over. This maximizes the Executor's usable context window (~28K tokens).

## Files Changed from Stage 0

| File | Change |
|------|--------|
| `agent/explorer.py` | **NEW**: ExplorerAgent wrapper with restricted tools |
| `agent/prompts.py` | Added `SYSTEM_PROMPT_EXPLORER_V1`, `build_initial_user_message(data_summary=)` |
| `config.py` | Added `pipeline`, `explorer_max_steps`, `explorer_max_wallclock_seconds` |
| `tbench_agent_v2.py` | Pipeline routing: `explorer+react` branch runs Explorer → Executor |

## How to Run

```bash
# From project root:
./scripts/switch_stage.sh stage0_5
THESIS_AGENT_PIPELINE=explorer+react THESIS_AGENT_PROMPT_VERSION=stage0 \
  ./scripts/run_single.sh hello-world 14b
```
