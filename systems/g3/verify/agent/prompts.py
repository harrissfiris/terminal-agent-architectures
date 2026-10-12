"""
System prompt and initial message builder for the ReAct terminal agent.

Design principle: give the LLM clear, minimal instructions and let it
reason freely.  No strategy injection, no classifier hints — just the
tools, the rules, and the task.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
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


def get_system_prompt() -> str:
    """Return the system prompt for the ReAct agent."""
    return SYSTEM_PROMPT


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


# ==================================================================
# Stage 2 — Variant-specific prompts
# ==================================================================

VERIFIER_SYSTEM_PROMPT = """\
You are an expert software engineer inside an isolated Docker container.
A previous agent attempted the task below and submitted their work.
Your job is to **verify and fix** the previous agent's solution.

## Tools
- **bash(command, timeout?)** — Run a shell command. Session is persistent (cd/env carry over). Default timeout 30s; use timeout=120 for slow commands (pip, apt, make, tests).
- **write_file(path, content)** — Write a complete file (creates parent dirs). Absolute paths only.
- **patch_file(path, old_text, new_text)** — Replace text in a file. Read the file first so old_text matches exactly.
- **read_file(path)** — Read file contents.
- **think(content)** — Private scratchpad for reasoning. Returns step/budget info.
- **submit(answer)** — Declare the task complete. Call ONCE when verified.

## Workflow
1. Read the task instruction carefully.
2. Examine the current state of the environment (files, code, outputs).
3. Check for correctness — run tests, verify outputs, look for obvious bugs.
4. Fix any issues you find.
5. Call submit() when the solution is correct.

## Rules
- The previous agent's work is already in the environment. Do NOT start from scratch.
- Focus on verification: check before changing.
- If the work looks correct, submit immediately — don't waste budget.
- If issues exist, fix them efficiently and submit.
- Use absolute paths (/app/..., /tests/...).
- No internet access. Work with what's in the container.

## Environment
Docker container. Python 3, bash, common CLI tools available.
Workspace: /app/ | Tests: /tests/run-tests.sh (if present)
"""


def get_verifier_system_prompt() -> str:
    """Return the system prompt for the verifier agent (variants A/B)."""
    return VERIFIER_SYSTEM_PROMPT


def build_verifier_initial_message(
    instruction: str,
    current_obs: str,
    env_info: str = "",
    reflexion: str = "",
) -> str:
    """
    Build the initial message for the verifier agent.

    Args:
        instruction: The original task instruction.
        current_obs: Current environment state (ls, pwd).
        env_info: Extended environment discovery.
        reflexion: Reflexion summary from Phase 1 (variant B only).

    Returns:
        Formatted verifier message.
    """
    parts = ["## Task\n", instruction.strip()]

    if reflexion:
        parts.extend([
            "\n\n## Previous Attempt Analysis\n",
            reflexion.strip(),
        ])

    parts.extend([
        "\n\n## Current Environment State\n",
        "```",
        current_obs.strip() if current_obs.strip() else "(empty)",
        "```",
    ])

    if env_info and env_info.strip():
        parts.extend([
            "\n\n## Environment Discovery\n",
            "```",
            env_info.strip(),
            "```",
        ])

    parts.append(
        "\n\nVerify the previous agent's work. "
        "Fix any issues, then submit()."
    )

    return "\n".join(parts)


# -- Reflexion generation (variant B) ---------------------------------

REFLEXION_SYSTEM = (
    "You are an expert debugging analyst. Given a trace of an agent's "
    "actions on a terminal task, write a concise reflexion that will help "
    "a new agent avoid the same mistakes. Focus on:\n"
    "1. What the agent tried and what went wrong\n"
    "2. Critical errors or wrong assumptions\n"
    "3. What should be done differently\n"
    "Be specific and actionable. Max 150 words."
)


def build_reflexion_user_message(
    instruction: str,
    trace_summary: str,
    intercepted_answer: str,
) -> str:
    """Build the user message for reflexion generation."""
    return (
        f"## Task\n{instruction.strip()}\n\n"
        f"## Agent Trace Summary\n{trace_summary}\n\n"
        f"## Agent's Submitted Answer\n{intercepted_answer}\n\n"
        "Write a concise reflexion for the next agent."
    )


# -- Judge evaluation (variant C) -------------------------------------

JUDGE_SYSTEM = (
    "You are a strict judge evaluating whether a terminal task was completed "
    "correctly. You will see the task instruction, a summary of the agent's "
    "actions, and the current state of key files.\n\n"
    "Respond with EXACTLY one of:\n"
    "- APPROVED — if the solution appears correct and complete\n"
    "- NEEDS_FIX: <specific feedback> — if there are concrete issues to fix\n\n"
    "Be specific about what's wrong. Do not speculate."
)


def build_judge_message(
    instruction: str,
    trace_summary: str,
    env_snapshot: str,
    intercepted_answer: str,
) -> str:
    """Build the user message for the judge LLM call."""
    return (
        f"## Task\n{instruction.strip()}\n\n"
        f"## Agent Actions Summary\n{trace_summary}\n\n"
        f"## Current Environment\n```\n{env_snapshot}\n```\n\n"
        f"## Agent's Answer\n{intercepted_answer}\n\n"
        "Is this solution correct and complete?"
    )
