# Dataset

Three CSVs, frozen at the end of the study. Every number in the thesis derives from them.

- **`runs.csv`**: one row per run (model × architecture × task), 7,424 rows. The unit of analysis.
- **`cells.csv`**: aggregates `runs.csv` to one row per cell (model × architecture).
- **`tasks.csv`**: aggregates to one row per task, across models and architectures.

## `runs.csv`: the columns that carry the analysis

| Column | Meaning |
|--------|---------|
| `model` | the model id; for the supervised track, the worker |
| `track` | `homogeneous` (one model plays every role) or `supervised` (a strong supervisor over a weaker worker) |
| `worker_model`, `supervisor_model` | the two roles in the supervised track |
| `arch` | one of the 14 architectures (`g1_baseline` … `g4_orchestra`) |
| `group` | the family (`g1`-`g4`) |
| `task` | the TerminalBench task id |
| `resolved` | **1 if the run passed every grader check, else 0**: the primary outcome |
| `n_subtests_passed` / `n_subtests_run` / `subtest_frac` | partial credit: the fraction of grader checks the final state passes |
| `steps` | number of agent steps taken |
| `term_class` | how the episode ended (submitted, or against a budget cap) |
| `submitted` | whether the agent declared itself finished |
| `max_repeat` | longest streak of identical consecutive shell commands (the loop signal) |
| `in_tok`, `out_tok` | input / output tokens |
| `bash_calls`, `write_calls`, `patch_calls`, `read_calls`, `think_calls` | tool-use counts |
| `guard_*` | command-guard actions (allowed / warned / blocked / rewritten) |
| `exp_*`, `plan_*` | explorer / planner artifacts and validity (G2) |
| `s2_*` | verification phase: interception, verdict, feedback (G3) |
| `orch_*`, `dlg_*` | orchestration and dialogue internals (G4) |
| `cfg_*` | the exact configuration of the run (model id, provider, temperature, budgets) |

`resolved` is binary and strict: all of a task’s grader checks must pass. `subtest_frac`
recovers the movement below that threshold and is used only as corroboration, never on its own.

## Reproducing

`analysis/gradient.py` reads `runs.csv` and reproduces the central finding and its robustness
checks with nothing but the Python standard library.
