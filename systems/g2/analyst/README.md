# Stage 2.5: "Best of Everything" Multi-Agent Pipeline

## Overview

The final optimized multi-agent configuration. Combines the best components
from every prior stage while removing the elements that caused regressions.

**Key innovation: Analyst = Explorer + Planner merged**
- Explorer (Stage 0.5) provided data grounding but no execution guidance
- Planner (Stage 1) provided plans but consumed extra budget and fragmented context
- Analyst merges both into a single read-only phase: same budget as Explorer,
  produces BOTH a Data Summary AND a data-grounded Execution Plan
- The plan is naturally grounded because the Analyst produced it AFTER
  inspecting the actual files, unlike Stage 1's Planner which planned from
  the task description alone

**Inherited from Stage 2:**
- StagnationDetector (3-tier escalation) + ProgressTracker (unique error tracking)
- Think-loop circuit breaker (2 consecutive max_tokens → force action)
- Write-file identity block (MD5 dedup, hard block after 2 identical writes)
- CommandGuard with inline Python blocking
- SelfValidator with optional Attempt 2 retry

**Inherited from Stage 3:**
- Smart output truncation (pip/apt → last 5 lines, general → head 40 + tail 40,
  error line rescue from truncated middle, ANSI stripping)

**Removed (vs Stage 2):**
- Python syntax checks in SelfValidator: caused false-positive regressions on
  assign-seats and jq-data-processing (compile() flagged working .py files)
- Separate Planner phase: merged into Analyst (zero budger overhead)

## Pipeline

```
Task Instruction
        ↓
  ┌──────────────────┐
  │ Analyst Agent     │  ← read-only (bash/read_file/think)
  │  (analyst_v1)     │     produces Data Summary JSON
  │                   │     + Execution Plan (data-grounded)
  └──────┬───────────┘
         │ Data Summary + Plan
         ↓
  ┌──────────────────────────────────────┐
  │ Attempt 1: Executor (with            │
  │   interventions + plan guidance)     │
  │   - StagnationDetector (3-tier)      │
  │   - ProgressTracker                  │
  │   - Think-loop circuit breaker       │
  │   - CommandGuard (inline blocking)   │
  │   - Write-file identity block        │
  │   - Smart output truncation          │
  └──────┬───────────────────────────────┘
         │ submitted?
         ↓
  ┌──────────────────────┐
  │ SelfValidator         │  ← zero LLM cost
  │  - file existence     │
  │  - format parsing     │
  │  - NO syntax checks   │
  └──────┬────────────────┘
         │ all passed?
    yes ──→ accept
    no  ──→ Attempt 2 (if budget remains)
```

## Key Components

### Analyst Agent (NEW in Stage 2.5)

**agent/explorer.py**: `ExplorerAgent` with `analyst_v1` prompt
- Same read-only tool restrictions as Explorer [bash whitelist, read_file, think]
- Same budget as Explorer (5 steps / 90s for 32B)
- Produces Data Summary JSON with embedded `execution_plan` field:
  - `approach`: brief strategy description
  - `steps`: 3-7 concrete, data-grounded action steps
  - `key_challenges`: gotchas discovered during inspection
- Plan extraction: `_extract_execution_plan()` formats JSON → readable text
- Plan is injected into Executor's user message via existing `plan_guidance` parameter
- Failure is non-fatal: missing plan = Executor runs without guidance (Stage 0.5 behavior)

### From Stage 2: Runtime Interventions

**StagnationDetector** (`agent/stagnation.py`)
- SequenceMatcher similarity on consecutive tool calls (threshold=0.80, window=5)
- 3-tier escalation after 3 consecutive similar calls
- After 4+ warnings: appends recent commands tried (anti-repeat)

**ProgressTracker** (`agent/stagnation.py`)
- Unique error fingerprinting across bash results
- Warning after 5 steps with no new unique error

**Think-loop Circuit Breaker** (`agent/react.py`)
- 2+ consecutive think() calls hitting ≥95% max_tokens → force action

**Write-File Identity Block** (`agent/executor.py`)
- MD5 hash dedup: warn on 1st identical write, hard block on 2nd+

### From Stage 3: Smart Output Truncation

**terminal/session.py**: `_truncate()` method
- pip/apt install output → keep last 5 lines only (install noise removed)
- General output > MAX_LINES (100) → keep first 40 + last 40 lines
- Error rescue: scan truncated middle for Traceback/Error/FAILED lines,
  preserve them with 1 line of context each
- ANSI escape code stripping

### Modified: SelfValidator (no syntax checks)

**agent/self_validator.py**
- File existence checks: ✓ (kept)
- Format parse checks (json/csv/yaml/parquet): ✓ (kept)
- Python syntax checks via compile(): ✗ (REMOVED: caused false positives)

## Budget (32B)
```
Pipeline: explorer+selfval+react
Total:    60 steps / 600s
Analyst:   5 steps /  90s  (produces Data Summary + Execution Plan)
Executor: ~55 steps / ~510s (Attempt 1, with plan guidance)
SelfVal:  ~2s (zero LLM cost)
Retry:    leftover only (min 5 steps / 60s to trigger)
```

## Diff vs Stage 2

| File | Change |
|------|--------|
| `agent/prompts.py` | Added `analyst_v1` prompt (Explorer inspection + planning merged) |
| `agent/explorer.py` | `ExplorerResult.execution_plan` field + `_extract_execution_plan()` + prompt → `analyst_v1` |
| `tbench_agent_v2.py` | Pass `plan_guidance` from Analyst to Executor; log `execution_plan.txt` |
| `terminal/session.py` | Replaced with Stage 3 version (smart truncation) |
| `agent/self_validator.py` | Removed `_check_python_syntax()` and its invocation |
| `config.sh` | Updated description |
| `README.md` | This file |
| All other files | Identical to Stage 2 |

## How to Run

```bash
./scripts/switch_stage.sh stage2_5
./scripts/run.sh 32b solvable_35

# Or single task:
./scripts/run.sh 32b single:hello-world
```
