# Terminal Agent Architectures

Code and data for the diploma thesis **“Agent Architectures for LLM-Based Terminal Agents: A Comparative Study Across Model Capability”** (School of Electrical and Computer Engineering, National Technical University of Athens, 2026).

📄 **Read the thesis:** [`Thesis_Sfyris.pdf`](Thesis_Sfyris.pdf)

The study asks a question the usual single-model evaluation cannot answer: **does the value of an agent architecture depend on the capability of the model it wraps?** To answer it, the same fourteen architectures are run over seven models that span a wide capability range, on the same tasks, under the same budget: a full grid rather than one architecture on one model.

- **14 architectures** in four families, built on one shared ReAct substrate.
- **7 models**: six open-weight (Qwen2.5-Coder 7B/14B/32B, Qwen2.5-72B, Llama-3.1-70B, Devstral-24B; 4-bit AWQ, served with vLLM) and one hosted closed model (Gemini).
- **64 tasks** from [TerminalBench](https://github.com/terminal-bench/terminal-bench).
- **7,424 runs**, one per cell, at temperature 0, under a **carved** 60-step budget that sub-agents *share* rather than add to, so a difference reflects organisation, not extra compute.
- A separate **heterogeneous-supervision** track, where a strong model supervises a weaker worker.

Every number in the thesis is generated from the frozen dataset in [`data/`](data/); nothing is typed by hand.

## The headline, in one command

```bash
python3 analysis/gradient.py
```

This reads `data/runs.csv` and prints the capability gradient: the number of architectures that beat a model’s own baseline falls from 13 of 13 on one of the weakest models to 0 of 13 on the strongest open coder (Spearman ρ = −0.89). Because the baseline is a single run used on both sides of that comparison, the script also reports the robustness checks that remove the coupling: a baseline-free correlation (ρ = −0.83), a split-half correlation (median ρ = −0.62), and a relabelling null (p = 0.037). The direction survives; the single run inflates the magnitude, not the sign.

## What the study finds

- **Architecture is not a universal improvement: its sign depends on the model.** The same scaffold, byte for byte, rescues a weak model and burdens a strong one. The number of architectures that beat a model’s baseline falls as the model gets more capable (ρ = −0.89, and the direction survives every control for single-run noise).
- **No universal default.** Averaged over models the main effect is near zero, because opposite-sign per-model effects cancel. The useful question is not *whether* to add an architecture but *which one for which task*.
- **The architecture that helps is the one matched to how the model fails:** grounding before acting, verification before submitting, routing for genuinely multi-stage work. A worked example (`log-summary` vs `fix-git`) shows the same scaffold helping one task and hurting another.
- **Capability can be injected from outside, but only through certain roles.** In the heterogeneous-supervision track a stronger supervisor transfers capability when it *adds* to the process (a plan, an instruction) and not when it only *judges*.

The thesis states each of these with its uncertainty; the headline above is reproducible from `data/` with `analysis/gradient.py`.

## The architectures

All fourteen are the same ReAct loop (think → act → observe over a Docker-backed terminal) with one thing added or changed. They live in [`systems/`](systems/):

| Family | Idea | Variants |
|--------|------|----------|
| **G1**: single-agent | the bare agent and two ablations of it | `baseline`, `prompt`, `temperature` (env-var driven on `g1/baseline`) |
| **G2**: pre-execution | ground the agent before it acts | `explorer`, `planner`, `analyst`, `monitor` |
| **G3**: post-execution | intercept the submission and verify | `instruction`, `reflexion`, `judge`, `gate` (`g3/verify`) |
| **G4**: multi-agent | several agents coordinate | `cooperation`, `debate` (`g4/dialogue`), `orchestra` |

Each system under `systems/<family>/<variant>/` is self-contained: a ReAct core (`agent/react.py`), tools, memory, a command guard, an LLM client (`llm/`), and a Docker terminal session (`terminal/`). G1 ablations (prompt, temperature, tool mode, budget) are selected by `THESIS_*` environment variables on `g1/baseline`; the other families add components (explorer, planner, verifier, orchestrator).

## Layout

```
systems/      the 14 architectures, grouped by family
data/         the frozen dataset, one row per model × architecture × task
analysis/     standalone scripts that reproduce the thesis numbers from data/
scripts/      the benchmark harness (run, switch architecture, model budgets)
notebooks/    Google Colab notebook for serving the open models with vLLM
docs/         task_matrix.html, the interactive cross-model solve matrix
SETUP.md      full from-scratch reproduction (Docker, task images, LLM serving)
Thesis_Sfyris.pdf   the thesis
```

## Running the benchmark

The benchmark has two halves on two machines: TerminalBench task containers run **locally in Docker** (no GPU needed), and the LLM is served **remotely** over an OpenAI-compatible API (RunPod for the 72B, Colab + ngrok for the smaller models). [`SETUP.md`](SETUP.md) covers the whole path: prerequisites, task images, serving, and the gotchas.

```bash
# point at your LLM endpoint (see scripts/config.sh / SETUP.md)
export BACKEND=runpod
export THESIS_LLM_API_BASE="https://<your-pod-id>-8000.proxy.runpod.net/v1"

bash scripts/switch_stage.sh g2/explorer      # select an architecture
bash scripts/run.sh 32b core_64               # run it on the 64-task subset
```

## Data

`data/runs.csv` is the unit of analysis: one row per run, with the model, architecture, track, task, whether it resolved, step and token counts, termination class, and the per-run behavioural signals used in the thesis (loop length, plan validity, verifier engagement, and so on). `cells.csv` and `tasks.csv` aggregate it by model × architecture and by task. See [`data/README.md`](data/README.md) for the schema.

## Citation

If you use this code or data, please cite the thesis (see [`CITATION.cff`](CITATION.cff)).

## License

Released under the MIT License ([`LICENSE`](LICENSE)). The TerminalBench tasks are the property of their authors and are not redistributed here.
