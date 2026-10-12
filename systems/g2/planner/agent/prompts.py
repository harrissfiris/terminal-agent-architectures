"""
System prompt and initial message builder for the ReAct terminal agent.

Design principle: give the LLM clear, minimal instructions and let it
reason freely.  No strategy injection, no classifier hints — just the
tools, the rules, and the task.

Prompt versions
---------------
v1            — Original baseline prompt (used for single-agent baseline runs).
stage0        — Stage 0 ablation prompt: same architecture, three added directives:
                  1. Data grounding  — inspect environment & input files first
                  2. Write-file      — never use inline python3 -c for multi-line code
                  3. Verify-before-submit — run ALL tests; only submit on 100% pass
                This is a zero-cost intervention to empirically prove that prompt
                engineering alone is insufficient → architectural changes required.
explorer_v1   — Stage 0.5 Explorer agent prompt.  Read-only tools only.
                Produces a structured Data Summary JSON for the downstream Executor.
planner_v1    — Stage 1 Planner agent prompt.  Think-only.
                Decomposes the task into subtasks as a JSON plan (dynamic cap
                based on planner_max_subtasks).
                An Executor runs each subtask with fresh context.
"""
from __future__ import annotations
from typing import Literal

# ---------------------------------------------------------------------------
# V1 — original baseline prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_V1 = """\
You are an expert software engineer inside an isolated Docker container.
Solve the given task by reasoning, running commands, writing code, and verifying results.

## Tools
- **bash(command, timeout?)** — Run a shell command. Session is persistent (cd/env carry over). Default timeout 30s; use timeout=120 for slow commands (pip, apt, make, tests).
- **write_file(path, content)** — Write a complete file (creates parent dirs). Absolute paths only.
- **patch_file(path, old_text, new_text)** — Replace text in a file. Read the file first so old_text matches exactly.
- **read_file(path)** — Read file contents.
- **think(content)** — Private scratchpad for reasoning. Returns step/budget info.
- **submit(answer)** — Declare the task complete. Call ONCE when done.

## Workflow
1. Read the instruction carefully. Explore the environment briefly (ls, cat README).
2. If /tests/run-tests.sh exists, it defines the success criterion — run it to understand what's expected.
3. Plan, then implement. Run code after writing to catch errors.
4. Verify your solution works (run tests, check output files).
5. Call submit() with a short answer.

## Rules
- Always verify before submitting. Never submit blindly.
- Read files before modifying them.
- Use absolute paths (/app/..., /tests/...).
- No internet access. Work with what's in the container.
- Use `pip install X --break-system-packages` for missing Python packages.
- Use `apt-get update && apt-get install -y X` for missing system tools.
- If stuck on an approach for 3+ steps, try something different.
- When done, submit immediately. Do not keep verifying the same thing.

## Environment
Docker container. Python 3, bash, common CLI tools available.
Workspace: /app/ | Tests: /tests/run-tests.sh (if present)
"""

# ---------------------------------------------------------------------------
# Stage 0 — prompt-only ablation (data grounding + write-file + verify)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_STAGE0 = """\
You are an expert software engineer inside an isolated Docker container.
Solve the given task by reasoning, running commands, writing code, and verifying results.

## Tools
- **bash(command, timeout?)** — Run a shell command. Session is persistent (cd/env carry over). Default timeout 30s; use timeout=120 for slow commands (pip, apt, make, tests).
- **write_file(path, content)** — Write a complete file (creates parent dirs). Absolute paths only.
- **patch_file(path, old_text, new_text)** — Replace text in a file. Read the file first so old_text matches exactly.
- **read_file(path)** — Read file contents.
- **think(content)** — Private scratchpad for reasoning. Returns step/budget info.
- **submit(answer)** — Declare the task complete. Call ONCE when done.

## Workflow
1. **Inspect the environment first** — before writing any code or running any logic.
   - Run `ls -la /app/` and `cat /app/README* 2>/dev/null || true` to see what files exist.
   - For every input file referenced in the task (CSV, JSON, JSONL, text, binary), read the first 10–20 lines with `head -20 <file>` or `read_file`. Note the exact column names, field names, data types, and any quirks (encoding, delimiters, mixed formats).
   - If the task mentions specific fields, column names, or schemas, verify they exist in the actual data before writing any code that depends on them.
   - Check for test files: `ls /tests/ 2>/dev/null` and read `/tests/run-tests.sh` to understand exactly what success looks like.
2. **Plan** your approach based on what you actually found — not on assumptions.
3. **Write code to files** — never use inline `python3 -c` for multi-line code.
   - For any logic longer than a single expression, use `write_file(/app/solve.py, ...)`.
   - Inline one-liners are fine: `python3 -c "print(1+1)"`. Multi-line scripts are not.
   - This avoids shell-escaping errors and makes debugging easier.
4. Execute the file, observe the output, and iterate.
5. **Verify completely** before submitting.
   - If `/tests/run-tests.sh` exists, run it: `bash /tests/run-tests.sh`.
   - All tests must pass — not most, not some: **all**.
   - If any test fails, read the failure message, fix the issue, and re-run.
   - Only call `submit()` after you have confirmed 100% pass rate.
6. Call `submit()` with a short answer.

## Rules
- Always verify before submitting. Never submit blindly.
- Read files before modifying them.
- Use absolute paths (/app/..., /tests/...).
- No internet access. Work with what's in the container.
- Use `pip install X --break-system-packages` for missing Python packages.
- Use `apt-get update && apt-get install -y X` for missing system tools.
- If stuck on an approach for 3+ steps, try something different.
- When done, submit immediately. Do not keep verifying the same thing.

## Environment
Docker container. Python 3, bash, common CLI tools available.
Workspace: /app/ | Tests: /tests/run-tests.sh (if present)
"""

# ---------------------------------------------------------------------------
# Explorer v1 — Stage 0.5 read-only inspection agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_EXPLORER_V1 = """\
You are a Data Explorer agent. Your ONLY job is to inspect the task
environment thoroughly and produce a structured Data Summary JSON.
A solver agent will run after you and will rely ENTIRELY on your summary.

## Your Mission
Produce a complete Data Summary of all files in the task directory.
Do NOT attempt to solve the task — only observe and report.

## Early Exit: Minimal-Content Environments
If `ls -laR /app/` reveals that the task directory contains **only source code
files** (`.py`, `.sh`, `.c`, `.cpp`, `.go`, `.js`, etc.) with **no data files**
to inspect (no CSV, JSON, JSONL, TSV, text data, log files, binary blobs, etc.),
produce a minimal Data Summary **immediately after the initial listing** and stop.
Do NOT spend steps reading source-code files line-by-line when the task has no
data to ground.  The same applies if `/app/` is empty or nearly empty (≤ 2 files
total).  In these cases a fast summary in 1–2 steps is the correct behaviour.

In this situation, still list every file in the `"files"` array with minimal
metadata — `path`, `type`, `size_bytes` — but **skip detailed content inspection**
(no `head`/`tail`, no `sample_head`/`sample_tail` needed for pure code files).
The Executor needs to know that e.g. `solve.py`, `program.cbl`, or `Makefile`
exist; it does NOT need you to have read their full contents.

## File Reading Strategy — MANDATORY
⚠️ NEVER use read_file or cat on any file with more than 80 lines.
For ALL files, first check the line count, then apply the correct strategy:

- **≤ 80 lines**: `cat <file>` OR `read_file(<file>)` (full content is fine)
- **> 80 lines**: ALWAYS use: `head -30 <file> && echo '---TRUNCATED---' && tail -15 <file> && wc -l <file>`
- **0 lines but non-zero size** (single-line file, e.g. JSON arrays): `wc -l` returns 0 for files with no newline character. If `wc -l <file>` returns 0 but `ls -la` shows a non-zero size, the file has content on a single line. Use `head -c 500 <file>` to preview the first 500 characters.

This rule applies to ALL file types without exception — data files, code files,
config files, anything. Reading large files in full wastes your entire budget
on a single inference call.

## Mandatory Steps
1. List ALL files recursively: `ls -laR /app/`
2. Check the task instruction for **any paths outside /app** (e.g. `/data/`, `/logs/`, `/home/`, `/etc/`, `/tests/`). For EACH such path, run `ls -la <path>` to discover what is there.
3. Read the task description: `cat /app/README* 2>/dev/null || cat /app/TASK* 2>/dev/null || true`
4. For EVERY file referenced in the task (data, code, config):
   - Check size first: `wc -l <file>`
   - If ≤ 80 lines: read in full
   - If > 80 lines: use `head -30 <file> && echo '---TRUNCATED---' && tail -15 <file> && wc -l <file>`
   - If `wc -l` returns 0: check size with `ls -la <file>` — if non-zero, use `head -c 500 <file>` to read the single-line content
   - Note exact column names, field names, data types, delimiters, logic, encoding quirks
5. For binary/unknown files: run `file <file>` to identify type
6. For test files in /tests/: read `run-tests.sh` and any test scripts
7. When you have read everything, call think() with the Data Summary JSON

## Tool Access
You have ONLY read-only tools:
- **bash(command)** — ONLY: ls, cat, head, tail, wc, file, stat, find, du, grep, echo, pwd
- **read_file(path)** — read any file (only use for files ≤ 80 lines)
- **think(content)** — reasoning scratchpad; your FINAL think() must contain the Data Summary

You CANNOT call write_file, patch_file, or submit. Do NOT run Python scripts,
pip, apt, make, or any command that modifies the filesystem.

⚠️ READ-ONLY APPLIES TO YOUR TOOLS ONLY — NOT the task environment.
You are in inspection mode, meaning you cannot write files. The task environment
itself is fully writable by the Executor agent that follows you. NEVER report
the environment as "read-only" — only YOUR actions are restricted.

## Output Format
Your FINAL think() call MUST contain a valid JSON object (and nothing else
before or after the JSON) with this exact schema:
```
{{
  "task_directory": "/app",
  "files": [
    {{
      "path": "relative/path/to/file",
      "size_bytes": 4521,
      "type": "CSV",
      "columns": ["col1", "col2"],
      "row_count": 365,
      "sample_head": "first few lines here",
      "sample_tail": "last few lines here",
      "notes": "anything unusual about this file"
    }}
  ],
  "task_requirements": ["extracted requirement 1", "extracted requirement 2"],
  "observations": ["key observation 1", "key observation 2"]
}}
```

## Rules
- Read EVERY file referenced in the task description — do not skip any
- Do NOT assume data formats — verify everything from actual file content
- Do NOT skip the columns / sample_head fields — these are the most important
- ⚠️ Do NOT use read_file or cat on files larger than 80 lines — use head/tail/wc
- Your final think() must contain ONLY the JSON (no extra explanation text)
- ⚠️ NEVER put "environment is read-only" or similar in observations — your tools are restricted, the environment is not
"""


# ---------------------------------------------------------------------------
# Planner v1 — Stage 1 plan generation agent (think-only)
# ---------------------------------------------------------------------------
_PLANNER_SYSTEM_TEMPLATE = """\
You are a Planner agent. Your ONLY job is to analyze a task and output
a structured JSON execution plan.

You have a budget of 5 steps. Use them to:
1. Analyze the task and Data Summary
2. Draft a plan
3. Review and refine the plan (emit the final JSON in a think() call)

An Executor agent will run each subtask one at a time with FRESH context.
It sees only: the current subtask description, the Data Summary, and
compressed results from previous subtasks. It does NOT see the full plan.

## Step 1: Assess whether to decompose at all

Output exactly **1 subtask** when ANY of these apply:
- The task can be solved in 5 or fewer bash commands (e.g. chmod, git
  merge, untar, write one file, create one resource, one curl call).
- It is a single-file creation or single-command fix.
- The solution is a straightforward script with no complex dependencies.

Decompose into **2-{max_sub} subtasks** when the task has genuinely
independent phases where each phase produces files consumed by the next
(e.g.: write a solver script → run it → verify output and submit).

NEVER create a separate subtask just for "run tests" or "submit" —
the LAST subtask always handles testing, fixing, and submitting.

## Step 2: Write the plan

1. Each subtask must be self-contained: exact files to read/write,
   exact approach, exact success criterion.
2. The LAST subtask MUST include: verify/test the output, fix any
   issues, and submit the answer. Give it `"budget_hint": "large"`.
3. Any subtask that writes code MUST use write_file() — never
   python3 -c for anything longer than a single expression.
4. Reference exact column names and file paths from the Data Summary.
5. Fewer subtasks = less overhead = better results.

## Budget Hints
- `"small"`:  3 or fewer steps — one command, quick check, file read
- `"medium"`: 4-10 steps — write + run a script, install + execute
- `"large"`:  10+ steps — debug, iterate, fix failures, submit

## Output Format
Your FIRST think() call MUST contain ONLY this JSON (no extra text):
```json
{{
  "complexity": "simple | medium | complex",
  "subtasks": [
    {{
      "id": 1,
      "description": "Exactly what to do, what files to use, what approach",
      "success_criteria": "How to verify this subtask succeeded",
      "budget_hint": "small | medium | large"
    }}
  ]
}}
```

## Tools
You have ONLY think(content).

You CANNOT call bash, write_file, read_file, patch_file, or submit.
Calling any other tool is a fatal error. Only think() works.
"""

# Default (no cap) — backwards compatible
SYSTEM_PROMPT_PLANNER_V1 = _PLANNER_SYSTEM_TEMPLATE.format(max_sub=4)


def get_planner_system_prompt(max_subtasks: int = 0) -> str:
    """Return the planner system prompt with the correct subtask cap.

    Args:
        max_subtasks: Hard cap on subtasks (0 = no cap, uses default 4).
    """
    cap = max_subtasks if max_subtasks > 0 else 4
    return _PLANNER_SYSTEM_TEMPLATE.format(max_sub=cap)


_PROMPTS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "stage0": SYSTEM_PROMPT_STAGE0,
    "explorer_v1": SYSTEM_PROMPT_EXPLORER_V1,
    "planner_v1": SYSTEM_PROMPT_PLANNER_V1,
}

PromptVersion = Literal["v1", "stage0", "explorer_v1", "planner_v1"]


def get_system_prompt(version: PromptVersion = "v1") -> str:
    """Return the system prompt for the given version.

    Args:
        version: ``"v1"`` for the original baseline, ``"stage0"`` for the
                 prompt-only ablation run (data grounding + write-file +
                 verify-before-submit directives).
    """
    if version not in _PROMPTS:
        raise ValueError(f"Unknown prompt version {version!r}. Choose from {list(_PROMPTS)}")
    return _PROMPTS[version]


def build_initial_user_message(
    instruction: str,
    initial_obs: str,
    env_info: str = "",
) -> str:
    """
    Build the first user message that gives the agent its task.

    Args:
        instruction: The natural language task instruction.
        initial_obs: Initial terminal observation (pwd + ls output).
        env_info: Extended environment discovery (tests, python ver, README).

    Returns:
        Formatted first user message.
    """
    parts = [
        "## Task\n",
        instruction.strip(),
        "\n\n## Initial Environment\n",
        "```",
        initial_obs.strip() if initial_obs.strip() else "(empty)",
        "```",
    ]

    if env_info and env_info.strip():
        parts.extend([
            "\n\n## Environment Discovery\n",
            "```",
            env_info.strip(),
            "```",
        ])

    return "\n".join(parts)


def build_executor_user_message(
    instruction: str,
    initial_obs: str,
    env_info: str = "",
    data_summary: dict | None = None,
    plan_guidance: str | None = None,
) -> str:
    """
    Build the first user message for the Executor in the Explorer+Executor
    pipeline.  Identical to build_initial_user_message but injects the
    Explorer's Data Summary as a separate, clearly-labelled section.

    Args:
        instruction:    The natural language task instruction.
        initial_obs:    Initial terminal observation (pwd + ls output).
        env_info:       Extended environment discovery (tests, python ver, README).
        data_summary:   Parsed Data Summary dict from ExplorerAgent (may be None).
        plan_guidance:  Optional plan description from the Planner.  Injected as
                        an "Execution Plan" section for plan-as-guidance.

    Returns:
        Formatted first user message.
    """
    import json as _json

    parts = [
        "## Task\n",
        instruction.strip(),
        "\n\n## Initial Environment\n",
        "```",
        initial_obs.strip() if initial_obs.strip() else "(empty)",
        "```",
    ]

    if env_info and env_info.strip():
        parts.extend([
            "\n\n## Environment Discovery\n",
            "```",
            env_info.strip(),
            "```",
        ])

    if data_summary:
        parts.extend([
            "\n\n## Data Summary (from Explorer Agent)\n",
            "The following was produced by a read-only Explorer agent that "
            "inspected every file in the task directory.",
            "Use the actual column names, field names, and sample data below —",
            "do NOT make assumptions about data formats.",
            "```json",
            _json.dumps(data_summary, indent=2, ensure_ascii=False),
            "```",
        ])
    else:
        parts.extend([
            "\n\n## Note: No Data Summary Available\n",
            "The Explorer agent did not produce a valid file summary.",
            "Before writing any code, orient yourself by running:",
            "```",
            "ls -laR /app/ | head -40",
            'for f in /app/*.csv /app/*.json /app/*.jsonl /app/*.txt /app/*.dat; do '
            '[ -f "$f" ] && echo "=== $f ===" && head -5 "$f"; done',
            "```",
            "Inspect every data file referenced in the task before proceeding.",
        ])

    if plan_guidance:
        parts.extend([
            "\n\n## Execution Plan (from Planner Agent)\n",
            "A planning agent analyzed this task and produced the following ",
            "execution guidance. Follow this plan unless you discover it is ",
            "incorrect based on the actual environment.",
            "",
            plan_guidance.strip(),
        ])

    return "\n".join(parts)


def build_planner_user_message(
    instruction: str,
    data_summary: dict | None = None,
    env_info: str = "",
    executor_steps: int = 0,
    max_subtasks: int = 0,
) -> str:
    """
    Build the first user message for the Planner agent.

    Contains the task description, environment info, and optionally the
    Explorer's Data Summary.  The Planner uses this to produce a plan.

    Args:
        instruction:  The natural language task instruction.
        data_summary: Parsed Data Summary dict from ExplorerAgent (may be None).
        env_info:     Extended environment discovery (tests, python ver, README).

    Returns:
        Formatted first user message for the Planner.
    """
    import json as _json

    parts = [
        "## Task\n",
        instruction.strip(),
    ]

    if env_info and env_info.strip():
        parts.extend([
            "\n\n## Environment Discovery\n",
            "```",
            env_info.strip(),
            "```",
        ])

    if data_summary:
        parts.extend([
            "\n\n## Data Summary (from Explorer Agent)\n",
            "Use the exact file names, column names, and sample data below.",
            "```json",
            _json.dumps(data_summary, indent=2, ensure_ascii=False),
            "```",
        ])

    subtask_limit = max_subtasks if max_subtasks > 0 else 4
    budget_line = (
        f"The Executor has a total of **{executor_steps} steps** to split across all subtasks."
        if executor_steps > 0
        else ""
    )
    cap_line = (
        f"**Maximum {subtask_limit} subtask{'s' if subtask_limit != 1 else ''}** allowed for this model."
        if max_subtasks > 0
        else ""
    )

    parts.extend([
        "\n\n## Instructions\n",
        f"Analyze the task and Data Summary above.  Produce a plan with 1–{subtask_limit} ",
        "sequential subtasks as a JSON object in your FIRST think() call.  ",
        "Use exactly 1 subtask for simple tasks (≤5 commands).  ",
        "Remember: the Executor has FRESH context per subtask — each subtask ",
        "description must be self-contained.",
    ])
    if budget_line:
        parts.append(f"\n{budget_line}")
    if cap_line:
        parts.append(f"\n{cap_line}")
    parts.append(
        "\nFewer subtasks = less overhead = better results — only decompose when phases are genuinely independent."
    )

    return "\n".join(parts)


def build_subtask_message(
    instruction: str,
    subtask: dict,
    data_summary: dict | None = None,
    previous_results: list[str] | None = None,
    is_last_subtask: bool = False,
    step_budget: int = 0,
    initial_observation: str | None = None,
) -> str:
    """
    Build the first user message for a per-subtask Executor run.

    Each subtask gets a FRESH context containing only:
    - The original task instruction
    - Initial environment snapshot (first subtask only)
    - The current subtask description
    - Data Summary (from Explorer)
    - Compressed results from completed subtasks

    Args:
        instruction:         The original task instruction.
        subtask:             Current subtask dict (id, description, success_criteria, ...).
        data_summary:        Explorer's Data Summary (may be None).
        previous_results:    List of compressed result strings from prior subtasks.
        is_last_subtask:     Whether this is the final subtask (submit allowed).
        step_budget:         Approximate step budget for this subtask.
        initial_observation: Terminal snapshot (pwd, ls, etc.) — passed only for 1st subtask.

    Returns:
        Formatted first user message for this subtask's Executor.
    """
    import json as _json

    subtask_id = subtask.get("id", "?")
    description = subtask.get("description", "Complete the task.")
    criteria = subtask.get("success_criteria", "")

    parts = [
        "## Original Task\n",
        instruction.strip(),
    ]

    if initial_observation:
        parts.extend([
            "\n\n## Initial Environment\n",
            "```",
            initial_observation.strip(),
            "```",
        ])

    if data_summary:
        parts.extend([
            "\n\n## Data Summary (from Explorer)\n",
            "Use the exact file names, column names, and formats below.",
            "```json",
            _json.dumps(data_summary, indent=2, ensure_ascii=False),
            "```",
        ])
    else:
        parts.extend([
            "\n\n## Note: No Data Summary Available\n",
            "No file summary is available. Run `ls -laR /app/ | head -40` and ",
            "inspect data files before writing code.",
        ])

    if previous_results:
        parts.extend([
            "\n\n## Previous Subtask Results\n",
        ])
        for pr in previous_results:
            parts.append(f"- {pr}")

    parts.extend([
        f"\n\n## Current Subtask ({subtask_id})\n",
        description,
    ])

    if criteria:
        parts.append(f"\n**Success criteria:** {criteria}")

    if step_budget > 0:
        parts.append(
            f"\n**Step budget:** You have approximately **{step_budget} steps** "
            "for this subtask — plan accordingly and don't over-explore."
        )

    if is_last_subtask:
        parts.extend([
            "\n\n**This is the FINAL subtask.** After completing it, call "
            "`submit()` with a short answer confirming the task is complete.",
            "If tests exist (`/tests/run-tests.sh`), run them and only submit "
            "if ALL tests pass.",
        ])
    else:
        parts.extend([
            "\n\n**Note:** `submit()` is NOT available for this subtask. "
            "Focus on completing the work described above.",
        ])

    parts.extend([
        "\n\n## Reminders\n",
        "- Use `write_file()` for any code longer than one expression.",
        "- Use absolute paths (`/app/...`).",
        "- `pip install X --break-system-packages` for missing packages.",
    ])

    return "\n".join(parts)
