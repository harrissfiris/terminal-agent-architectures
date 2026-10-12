"""
System prompt and message builders for G4 Dialogue agents.

The dialogue system uses the v1 baseline prompt (no prompt confound)
with additional context injection for turn handoffs.

Two variants:
  cooperation — Agent B continues Agent A's work (relay/tag-team)
  debate      — Agent B critiques Agent A's approach, then tries differently
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# V1 — original baseline prompt (only prompt used in dialogue)
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

_PROMPTS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
}

# ---------------------------------------------------------------------------
# Dialogue variant addenda — appended to the system prompt
# ---------------------------------------------------------------------------
COOPERATION_ADDENDUM = """

## Collaboration Mode
You are collaborating with a partner agent in a tag-team fashion. You take turns
working on this task. When you receive your partner's progress summary, continue
building on their work. Do not redo work they completed successfully.
The terminal session is shared — all files are persistent between turns.
"""

DEBATE_ADDENDUM = """

## Collaboration Mode
You are in a debate-style collaboration with a partner agent. You take turns
working on this task. When you receive your partner's progress:
1. Critically analyze their approach — what assumptions might be wrong?
2. If their approach failed, try a fundamentally different strategy.
3. Do not repeat the same commands they already tried.
The terminal session is shared — all files are persistent between turns.
"""


def get_system_prompt(
    version: str = "v1",
    dialogue_variant: str | None = None,
) -> str:
    """Return the system prompt, optionally with a dialogue variant addendum."""
    if version not in _PROMPTS:
        raise ValueError(
            f"Unknown prompt version {version!r}. Choose from {list(_PROMPTS)}"
        )
    prompt = _PROMPTS[version]
    if dialogue_variant == "debate":
        prompt += DEBATE_ADDENDUM
    elif dialogue_variant == "cooperation":
        prompt += COOPERATION_ADDENDUM
    return prompt


def build_initial_user_message(
    instruction: str,
    initial_obs: str,
    env_info: str = "",
    handoff_context: Optional[str] = None,
) -> str:
    """
    Build the first user message that gives the agent its task.

    When handoff_context is provided (dialogue turns 2+), it is appended
    after the environment section.

    Args:
        instruction:      The natural language task instruction.
        initial_obs:      Initial terminal observation (pwd + ls output).
        env_info:         Extended environment discovery (tests, python ver, README).
        handoff_context:  Formatted handoff section from the previous agent's turn.

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

    if handoff_context:
        parts.extend(["\n\n", handoff_context])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Handoff context builders — one per dialogue variant
# ---------------------------------------------------------------------------

def build_cooperation_handoff(
    turn_number: int,
    agent_label: str,
    action_log: str,
    last_reasoning: str,
) -> str:
    """
    Build handoff context for the COOPERATION variant.

    The incoming agent is told to *continue* the previous agent's work.
    """
    parts = [
        f"## Partner's Progress (Turn {turn_number})",
        f"Your partner ({agent_label}) worked on this task. "
        f"Here is what they did:\n",
        action_log,
    ]

    if last_reasoning:
        parts.extend([
            f"\n**{agent_label}'s last analysis:**",
            last_reasoning,
        ])

    parts.extend([
        "\n**Your job:** Continue their work. Build on what they found and "
        "accomplished. Do not repeat work they already completed successfully. "
        "The terminal session is shared — all files they created or modified "
        "are still present. Focus on making forward progress.",
    ])

    return "\n".join(parts)


def build_debate_handoff(
    turn_number: int,
    agent_label: str,
    action_log: str,
    last_reasoning: str,
) -> str:
    """
    Build handoff context for the DEBATE variant.

    The incoming agent must first CRITIQUE the previous approach in a think()
    call, then try a different strategy.
    """
    parts = [
        f"## Partner's Approach (Turn {turn_number})",
        f"Your partner ({agent_label}) tried to solve this task. "
        f"Here is what they did:\n",
        action_log,
    ]

    if last_reasoning:
        parts.extend([
            f"\n**{agent_label}'s reasoning:**",
            last_reasoning,
        ])

    parts.extend([
        "\n**Your job:** Critically analyze what went wrong or what could be "
        "improved in their approach, then try a DIFFERENT strategy. "
        "Do not repeat the same commands or approach they already tried — "
        "if their approach was working, they would have solved it. "
        "The terminal session is shared — all files they created or modified "
        "are still present.",
    ])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Submit-governance handoffs — used when submission needs partner confirmation
# ---------------------------------------------------------------------------

def build_confirmation_handoff(
    turn_number: int,
    agent_label: str,
    action_log: str,
    last_reasoning: str,
    proposed_answer: str,
) -> str:
    """
    Build handoff for COOPERATION confirmation.

    The incoming agent must verify the proposed solution and either
    confirm (submit) or reject (continue working).
    """
    parts = [
        f"## Partner's Proposed Solution (Turn {turn_number})",
        f"Your partner ({agent_label}) believes the task is solved and "
        f"proposes submitting:\n",
        f"**Proposed answer:** {proposed_answer}\n",
        f"**What they did:**\n",
        action_log,
    ]

    if last_reasoning:
        parts.extend([
            f"\n**{agent_label}'s reasoning:**",
            last_reasoning,
        ])

    parts.extend([
        "\n**Your job:** Verify their solution before accepting.",
        "1. Run tests if available: `bash /tests/run-tests.sh`",
        "2. Check the solution matches the task requirements.",
        "3. If correct, call `submit()` with the final answer.",
        "4. If the solution is wrong or incomplete, explain the problem "
        "in `think()` and continue working to fix it.",
    ])

    return "\n".join(parts)


def build_evaluation_handoff(
    turn_number: int,
    agent_label: str,
    action_log: str,
    last_reasoning: str,
    proposed_answer: str,
) -> str:
    """
    Build handoff for DEBATE evaluation.

    Agent B (critic) evaluates Agent A's proposed solution and decides
    whether to accept (submit) or reject (try differently).
    """
    parts = [
        f"## Proposed Solution for Evaluation (Turn {turn_number})",
        f"Your partner ({agent_label}) proposes this solution:\n",
        f"**Proposed answer:** {proposed_answer}\n",
        f"**What they did:**\n",
        action_log,
    ]

    if last_reasoning:
        parts.extend([
            f"\n**{agent_label}'s reasoning:**",
            last_reasoning,
        ])

    parts.extend([
        "\n**Your job as evaluator:** Critically assess this solution.",
        "1. Check for errors, incorrect assumptions, or missed requirements.",
        "2. Run tests if available: `bash /tests/run-tests.sh`",
        "3. If the solution is correct, call `submit()` to accept it.",
        "4. If flawed, explain the problems in `think()` and implement a fix.",
    ])

    return "\n".join(parts)
