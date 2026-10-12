# Stage 2: Submit-Intercept Experiments

## Overview

Stage 2 builds on the Phase 1 ReAct agent by adding a **submit-intercept mechanism**. When the agent calls `submit()`, the call is intercepted and control passes to variant-specific verification/retry logic. This tests whether self-verification can improve accuracy without requiring external test oracles.

## Variants

| Variant | Name | Mechanism |
|---------|------|-----------|
| **A** | Clean Verifier | Fresh agent gets only task instruction + current env state |
| **B** | Reflexion Verifier | Fresh agent gets instruction + LLM-generated reflexion from Phase 1 trace |
| **C** | Judge + Resume | Judge LLM evaluates; if rejected, original agent resumes with feedback |
| **D** | Are-You-Sure | Same agent, submit intercepted ×3 (forced self-verification) |

## Pipeline

```
Task Instruction + Initial Observation
        ↓
  ┌─────────────┐
  │  ReActAgent  │  ← Phase 1 agent, full budget
  │  (react.py)  │
  └──────┬───────┘
         │ calls submit()
         ↓
  ┌──────────────────┐
  │  Submit Gate      │  ← intercepts the submit call
  │  (executor.py)    │
  └──────┬───────────┘
         │
    ┌────┼────┬────┐
    A    B    C    D
    │    │    │    │
  fresh  fresh judge  "are you
  agent  agent LLM   sure?" ×3
    │    │    │    │
    ↓    ↓    │    ↓
  submit submit │  same agent
              │  continues
              ↓
         feedback →
         original agent
         resumes
```

## Usage

```bash
# Set the variant
export THESIS_STAGE2_VARIANT=a  # a, b, c, or d

# Run with tb CLI
AGENT_IMPORT_PATH=tbench_agent_v2:Stage2TerminalAgent tb run ...
```

## Key Insight

`/tests/run-tests.sh` does NOT exist in the Docker container during agent execution: the TerminalBench harness copies tests only after the agent finishes. Therefore all verification must be **agent-driven**, not test-driven.
- Design principle: let the LLM reason freely with clear rules and tool descriptions

### Tool Calling
- Mode: `"prompt"` (tool schemas injected into the system prompt; the LLM outputs JSON, parsed by the provider)
- Rationale: fair cross-model comparison (not all model sizes support native tool_calls reliably)
- Fallback: if JSON parsing fails → output wrapped as `think()` call (graceful degradation)

### Memory
- Token-aware sliding window (~28K token budget)
- Oldest turns dropped when budget exceeded
- Key facts extraction: important discoveries persist across window slides
- Output truncation: error lines prioritized over success output

### Safety
- CommandGuard blocks destructive commands (rm -rf /, mkfs, fork bombs)
- Warns on risky commands (sudo, chmod 777, curl|sh)
- Rewrites interactive pagers (less → cat)

### Budget (per model profile)
| Profile | max_steps | max_wallclock | max_tokens | max_context_tokens |
|---------|-----------|---------------|------------|--------------------|
| 7b      | 40        | 600s          | 1024       | 28000              |
| 14b     | 50        | 600s          | 2048       | 28000              |
| 32b     | 60        | 600s          | 2048       | 28000              |
| 72b     | 60        | 600s          | 2048       | 28000              |

### Environment Discovery (front-loaded)
Before the agent starts, the entry point (`tbench_agent_v2.py`) gathers:
1. `pwd && ls -la`: initial directory listing
2. `/tests/` existence check (defines the success criterion)
3. `python3 --version`
4. `cat README*` (head -40): task instructions if present

All injected into the first user message alongside the task instruction.

## Files

| File | Purpose |
|------|---------|
| `tbench_agent_v2.py` | TerminalBench entry point (`ReActTerminalAgent.perform_task()`) |
| `agent/react.py` | Core ReAct loop with budget enforcement and force-submit |
| `agent/executor.py` | Tool execution bridge (bash, write_file, patch_file, read_file, think, submit) |
| `agent/prompts.py` | v1 system prompt + initial message builder |
| `agent/tools.py` | Tool schemas (OpenAI function-calling format) |
| `agent/memory.py` | Sliding window memory with key facts and output truncation |
| `agent/safety.py` | CommandGuard |
| `config.py` | Central configuration (LLMConfig, AgentConfig, env var loading) |
| `model_profiles.py` | Per-model-size profiles (7b/14b/32b/72b) |
| `llm/openai_compat.py` | OpenAI-compatible LLM provider (Ollama, vLLM) |
| `terminal/session.py` | SafeTerminal tmux wrapper |

## How to Run

```bash
# From project root:
./scripts/switch_stage.sh phase1
./scripts/run_single.sh hello-world 14b
```
