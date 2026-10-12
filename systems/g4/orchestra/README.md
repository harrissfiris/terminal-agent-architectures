# G4 Orchestra: Orchestrator with Specialist Sub-Agents

## Overview

Orchestra adds **dynamic routing** to the agent family. Instead of a single generalist (G1) or a fixed pre-execution pipeline (G2), an orchestrator decides at run time which specialist to call next and what it should do. The orchestrator is an LLM with no tools: it never touches the terminal, it only manages the workflow.

This makes it possible to test role specialization against a generalist agent, dynamic routing against a fixed pipeline, and the coordination overhead of a manager that consumes tokens without making direct progress.

## Architecture

```
            ┌──────────────────┐
            │   Orchestrator   │
            │ (LLM, no tools)  │
            └────────┬─────────┘
                     │ {"next": role, "instruction": ..., "reasoning": ...}
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
  ┌──────────┐  ┌──────────┐  ┌──────────────┐
  │ Explorer │  │  Coder   │  │   Verifier   │
  │ read-only│  │  writes  │  │ tests, fixes │
  │          │  │  code    │  │ and submits  │
  └──────────┘  └──────────┘  └──────────────┘
        └─────────────┴─────────────┘
                shared terminal
```

| Role | Tools | Purpose |
|---|---|---|
| Orchestrator | none | Chooses the next role and writes a detailed instruction for it |
| Explorer | `bash`, `read_file`, `think` | Investigates the environment and the task |
| Coder | `bash`, `write_file`, `patch_file`, `read_file`, `think` | Implements the solution |
| Verifier | `bash`, `write_file`, `patch_file`, `read_file`, `think`, `submit` | Runs the checks, fixes what fails, and submits |

## Execution loop

1. The orchestrator receives the task, a summary of the progress so far and the remaining budget, and returns the next role with an instruction.
2. That sub-agent runs a short burst of up to 6 steps with its own tool set. It can end the burst early with a report.
3. A summary of the burst goes back to the orchestrator, and the next cycle starts.
4. Only the verifier can submit. When the explorer or coder call `submit`, it is treated as "assignment done" and their report is passed back to the orchestrator.
5. If the budget runs out without a submission, the current state is submitted automatically.

All sub-agents work in the same terminal session. Each one keeps its own memory across invocations, so an explorer called again on a later cycle remembers its earlier work. The orchestrator is stateless: every routing call gets the full state injected explicitly.

## Budget and fairness

- Routing steps are not counted. The sub-agents share the same 60-step budget as the single-agent baseline, so a difference in results reflects the organisation of the work, not extra compute.
- Safety limits: at most 20 cycles, plus the wall-clock timeout.
- An optional shared knowledge base (a notepad all agents can write to with a `note` tool) is available with `THESIS_USE_KNOWLEDGE_BASE=1`. It is off in the reported experiments.

## Files

| File | Role |
|---|---|
| `agent/orchestra.py` | `OrchestraAgent`: routing loop, sub-agent bursts, burst summaries |
| `agent/prompts.py` | Prompts for the orchestrator, explorer, coder and verifier |
| `agent/knowledge_base.py` | Optional shared knowledge base |
| `config.py` | `burst_steps` (default 6, `THESIS_AGENT_BURST_STEPS`) |
| `config.sh` | Experiment settings: `pipeline=orchestra`, `prompt=orchestrator`, burst of 6 steps |
| `tbench_agent_v2.py` | TerminalBench entry point that uses `OrchestraAgent` |

## Running

```bash
bash scripts/switch_stage.sh g4/orchestra
bash scripts/run.sh 32b core_64
```
