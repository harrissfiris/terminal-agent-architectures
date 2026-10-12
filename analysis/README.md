# Analysis

Standalone scripts that reproduce the thesis numbers from [`../data/`](../data/). Standard
library only: no dependencies, no network, no serving.

- **`gradient.py`**: the central finding. Prints the capability gradient (architectures that
  beat baseline, per model) and the robustness checks that control for the single-run baseline:
  a baseline-free correlation, a split-half correlation, and a relabelling null.
- **`stats.py`**: the statistical protocol the whole thesis uses. For every model × architecture
  cell it prints the paired difference against that model's own baseline with a 95% bootstrap
  interval, the exact McNemar p-value, and the Holm correction over the 13 architectures.
- **`model_card.py`**: "which architecture for which model". For one model it prints what helps
  and what hurts, whether each architecture's machinery actually engaged, which tasks were gained
  and lost, and where its solves fall by category.

```bash
python3 analysis/gradient.py
python3 analysis/stats.py              # ~30 s (bootstraps)
python3 analysis/model_card.py --list
python3 analysis/model_card.py qwen-32b
```

The scripts are deterministic (fixed seed), so the numbers they print match the thesis.
