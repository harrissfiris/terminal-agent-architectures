#!/usr/bin/env python3
"""
The one statistical protocol the whole thesis uses.

Fixed once, here, so that no two chapters can quietly use different tests,
different pairings, or different correction families.

Design decisions, stated once:

* **Everything is paired within a model.** An architecture is only ever
  compared against *that model's own* baseline, task by task. Pairing removes
  task-difficulty variance, which is by far the largest source of spread in a
  64-task set where some tasks are solved by everything and some by nothing.

* **McNemar is the primary test** for "does architecture A differ from
  baseline on this model". The data are paired binary outcomes, which is
  exactly what McNemar is for, and only the discordant pairs carry information.
  We use the exact binomial version: the discordant counts here are small
  (often < 15), where the chi-square approximation is unreliable.

* **The bootstrap gives the interval.** A paired percentile bootstrap over
  tasks yields a CI on the solve-count difference. The test says "is there a
  signal"; the interval says "how big, and how uncertain" -- and for this study
  the interval is the more honest object, because with n=1 runs most cells are
  not individually significant and the thesis argues direction, not cells.

* **Holm--Bonferroni within a family.** The correction family is the 13
  architectures compared against baseline *for one model*. It is not applied
  across models: those are separate questions, and pooling them would be a
  larger family than any claim requires.

* **No claim rests on a single cell.** These functions exist to put honest
  intervals on the numbers, not to manufacture significance. Where nothing is
  significant after correction -- which is the expected outcome for most cells --
  the thesis says so.
"""

from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(REPO, "data")

N_BOOT = 10000
SEED = 20260805


# ---------------------------------------------------------------- loading ---

def load_runs(path: str | None = None) -> list[dict]:
    with open(path or os.path.join(FROZEN, "runs.csv")) as fh:
        return list(csv.DictReader(fh))


def solve_vectors(runs: list[dict], model: str, arch_a: str, arch_b: str
                  ) -> tuple[list[int], list[int], list[str]]:
    """Aligned 0/1 outcome vectors for two architectures of one model.

    Alignment is by task id, so the pairing is explicit rather than positional.
    Only tasks present in both cells are returned.
    """
    a = {r["task"]: int(r["resolved"]) for r in runs
         if r["model"] == model and r["arch"] == arch_a}
    b = {r["task"]: int(r["resolved"]) for r in runs
         if r["model"] == model and r["arch"] == arch_b}
    tasks = sorted(set(a) & set(b))
    return [a[t] for t in tasks], [b[t] for t in tasks], tasks


# ------------------------------------------------------------------ tests ---

def mcnemar_exact(x: list[int], y: list[int]) -> dict:
    """Exact McNemar for paired binary outcomes.

    b = tasks x solved but y did not; c = the reverse. Concordant pairs carry
    no information about the difference and are excluded by construction.
    """
    b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
    c = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p": 1.0}
    # two-sided exact binomial against p=0.5
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b": b, "c": c, "n_discordant": n, "p": min(1.0, 2 * tail)}


def paired_bootstrap_diff(x: list[int], y: list[int], n_boot: int = N_BOOT,
                          alpha: float = 0.05, seed: int = SEED) -> dict:
    """Percentile CI on (sum(x) - sum(y)), resampling *tasks* with replacement.

    Resampling tasks (not runs) is what makes the interval a statement about
    generalisation to other tasks of this kind, which is the inference the
    thesis actually wants.
    """
    rng = random.Random(seed)
    n = len(x)
    obs = sum(x) - sum(y)
    if n == 0:
        return {"diff": 0, "lo": 0.0, "hi": 0.0, "n": 0}
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(x[i] for i in idx) - sum(y[i] for i in idx))
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"diff": obs, "lo": lo, "hi": hi, "n": n}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm--Bonferroni step-down over one family of comparisons."""
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, rejected_so_far = {}, True
    for i, (key, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        # step-down: once a test fails, all larger p-values also fail
        rejected_so_far = rejected_so_far and (p <= thresh)
        out[key] = {"p": p, "p_adj": min(1.0, p * (m - i)),
                    "threshold": thresh, "significant": rejected_so_far}
    return out


# ------------------------------------------------------------- convenience --

def architecture_family(runs: list[dict], model: str, archs: list[str],
                        baseline: str = "g1_baseline") -> dict[str, dict]:
    """Full protocol for one model: every architecture against its baseline.

    Returns per-architecture diff, bootstrap CI, McNemar p, and the
    Holm-corrected verdict over the family of 13.
    """
    results, pvals = {}, {}
    for arch in archs:
        if arch == baseline:
            continue
        x, y, _ = solve_vectors(runs, model, arch, baseline)
        if not x:
            continue
        mc = mcnemar_exact(x, y)
        bs = paired_bootstrap_diff(x, y)
        results[arch] = {**bs, **mc}
        pvals[arch] = mc["p"]
    for arch, verdict in holm(pvals).items():
        results[arch].update(verdict)
    return results


def fmt_ci(r: dict) -> str:
    return f"{r['diff']:+d} [{r['lo']:+d}, {r['hi']:+d}]"


if __name__ == "__main__":
    ARCHS = ["g1_baseline", "g1_prompt", "g1_temp",
             "g2_explorer", "g2_planner", "g2_analyst", "g2_monitor",
             "g3_instruction", "g3_reflexion", "g3_judge", "g3_gate",
             "g4_cooperation", "g4_debate", "g4_orchestra"]
    MODELS = ["qwen-7b", "llama-70b", "qwen-14b", "devstral-small-2-24b",
              "qwen-72b", "qwen-32b", "gemini"]

    runs = load_runs()
    print("Architecture vs. own baseline, paired over 64 tasks.")
    print("diff [95% bootstrap CI], McNemar exact p, Holm-corrected over 13.\n")
    any_sig = 0
    for model in MODELS:
        res = architecture_family(runs, model, ARCHS)
        sig = [a for a, r in res.items() if r.get("significant")]
        any_sig += len(sig)
        print(f"{model}")
        for arch, r in sorted(res.items(), key=lambda kv: -kv[1]["diff"]):
            star = " *" if r.get("significant") else ""
            print(f"   {arch:16s} {fmt_ci(r):>16s}   p={r['p']:.3f}  "
                  f"disc={r['n_discordant']:2d}{star}")
        print(f"   -> significant after Holm: {len(sig)}/13"
              f"{' (' + ', '.join(sig) + ')' if sig else ''}\n")
    print(f"TOTAL significant cells across the whole grid: {any_sig}/91")


# ------------------------------------------------------- gradient-level ----
# The thesis's actual claim is not about any single cell (none survive
# correction) but about the *relationship* between a model's capability and the
# benefit it derives from architecture. These functions test that relationship.

def spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def spearman_exact_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """Exact permutation p-value. n<=8 here, so all n! orderings are enumerated."""
    import itertools
    rho = spearman(x, y)
    perms = [spearman(x, list(p)) for p in itertools.permutations(y)]
    return rho, sum(1 for r in perms if abs(r) >= abs(rho)) / len(perms)


def permutation_mean_diff(a: list[float], b: list[float], n_perm: int = 20000,
                          seed: int = SEED) -> tuple[float, float]:
    """Two-sample permutation test on the difference of means."""
    rng = random.Random(seed)
    obs = (sum(a) / len(a)) - (sum(b) / len(b))
    pool, na = a + b, len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        m = (sum(pool[:na]) / na) - (sum(pool[na:]) / (len(pool) - na))
        if abs(m) >= abs(obs):
            hits += 1
    return obs, hits / n_perm


def sign_test(values: list[int]) -> dict:
    """Two-sided exact binomial sign test on non-zero deltas."""
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p": 1.0}
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    p = sum(pr for pr in probs if pr <= probs[pos] * (1 + 1e-9))
    return {"pos": pos, "neg": neg, "n": n, "p": min(1.0, p)}
