#!/usr/bin/env bash
# Stage 2.5: Analyst + pure Phase I Executor.
#
# Key innovation: Analyst = Explorer + Planner merged into one read-only phase.
#   - Produces BOTH Data Summary AND data-grounded Execution Plan
#   - Plan is naturally grounded because the Analyst inspected actual files
#   - Zero overhead vs separate Planner (Stage 1's lesson)
#
# Executor: identical to Phase I (no deterministic interventions).
#   - Only infra: CommandGuard (env normalizations), output truncation
#   - Analyst output (Data Summary + Plan) injected as context in first message
#   - Prompt: v1 (baseline — context comes from user-message injection, not system prompt)
#
# Budget: CARVED, not additive. The code computes executor_steps = max_steps - explorer_max_steps
# (tbench_agent_v2.py:214-216), so total steps are CONSERVED at the profile's 60. Wall-clock 1800s
# comes from the uniform model profile; the executor gets (1800 - explorer_elapsed). Uniform across
# all models now, so no per-profile overrides are needed beyond the analyst's own 10-step carve.

configure_budgets() {
  export THESIS_AGENT_PIPELINE=explorer+selfval+react
  export THESIS_AGENT_PROMPT_VERSION=v1
  # Analyst (explorer + self-validation, read-only) carves 10 of the 60 steps; executor gets 50.
  # max_steps (60) and max_wallclock (1800) are inherited uniformly from model_profiles.py.
  export THESIS_AGENT_EXPLORER_MAX_STEPS=10

  # No self-validation retry in Stage 2.5 (removed: caused regressions).
}
