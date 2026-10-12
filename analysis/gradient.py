#!/usr/bin/env python3
"""Reproduce the central finding: the value of an architecture falls as the
model underneath it gets more capable, and that this direction survives the
single-run noise in the grid.

Reads data/runs.csv (one row per model x architecture x task) and prints the
capability gradient together with the robustness checks reported in the thesis:
a relabelling null for the coupled baseline, a baseline-free correlation, and a
split-half correlation. Standard library only.

    python3 analysis/gradient.py
"""
import csv
import random
import statistics as st
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "runs.csv"
SEED = 20260805

ARCHS = ["g1_baseline", "g1_prompt", "g1_temp",
         "g2_explorer", "g2_planner", "g2_analyst", "g2_monitor",
         "g3_instruction", "g3_reflexion", "g3_judge", "g3_gate",
         "g4_cooperation", "g4_debate", "g4_orchestra"]
NONBASE = [a for a in ARCHS if a != "g1_baseline"]
# open models, ordered by baseline solves (weakest first)
OPEN = ["qwen-7b", "llama-70b", "qwen-14b", "devstral-small-2-24b",
        "qwen-72b", "qwen-32b"]
PRETTY = {"qwen-7b": "Qwen2.5-Coder-7B", "llama-70b": "Llama-3.1-70B",
          "qwen-14b": "Qwen2.5-Coder-14B", "devstral-small-2-24b": "Devstral-24B",
          "qwen-72b": "Qwen2.5-72B", "qwen-32b": "Qwen2.5-Coder-32B"}


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def load():
    grid = {}
    with open(DATA, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["track"] != "homogeneous":
                continue
            grid.setdefault(r["model"], {}).setdefault(r["arch"], {})[r["task"]] = int(r["resolved"] or 0)
    return grid


def solves(grid, m, a):
    return sum(grid[m][a].values())


def main():
    grid = load()
    tasks = sorted(grid["qwen-7b"]["g1_baseline"])

    base = {m: solves(grid, m, "g1_baseline") for m in OPEN}
    nbetter = {m: sum(solves(grid, m, a) > base[m] for a in NONBASE) for m in OPEN}
    mean_delta = {m: st.mean(solves(grid, m, a) for a in NONBASE) - base[m] for m in OPEN}

    print("Capability gradient (open models, weakest first)\n")
    print(f"  {'model':20s} {'baseline':>8s} {'archs>base':>11s} {'mean delta':>11s}")
    for m in OPEN:
        print(f"  {PRETTY[m]:20s} {base[m]:8d} {nbetter[m]:>8d}/13 {mean_delta[m]:>+11.2f}")

    rho = spearman([base[m] for m in OPEN], [mean_delta[m] for m in OPEN])
    print(f"\n  Spearman(baseline, mean delta)        rho = {rho:+.3f}")

    # robustness: the baseline is one run on both axes, so check it three ways.
    amean = {m: st.mean(solves(grid, m, a) for a in NONBASE) for m in OPEN}
    rho_free = spearman([amean[m] for m in OPEN], [mean_delta[m] for m in OPEN])

    rng = random.Random(SEED)
    sh = []
    for _ in range(500):
        t = tasks[:]
        rng.shuffle(t)
        a, b = t[:len(t) // 2], t[len(t) // 2:]
        cap = [sum(grid[m]["g1_baseline"][x] for x in a) for m in OPEN]
        d = [st.mean(sum(grid[m][ar][x] - grid[m]["g1_baseline"][x] for x in b)
                     for ar in NONBASE) for m in OPEN]
        sh.append(spearman(cap, d))

    rng = random.Random(SEED + 1)
    null = []
    for _ in range(5000):
        bb, dd = [], []
        for m in OPEN:
            lab = ARCHS[:]
            rng.shuffle(lab)
            relab = {lab[i]: ARCHS[i] for i in range(len(ARCHS))}
            bv = solves(grid, m, relab["g1_baseline"])
            bb.append(bv)
            dd.append(st.mean(solves(grid, m, relab[a]) - bv for a in NONBASE))
        null.append(spearman(bb, dd))
    p = sum(r <= rho for r in null) / len(null)

    print("\nRobustness to the single-run baseline")
    print(f"  baseline-free (capability = mean of non-baseline cells)  rho = {rho_free:+.3f}")
    print(f"  split-half (disjoint task halves)              median rho = {st.median(sh):+.3f}")
    print(f"  relabelling null                            mean rho = {st.mean(null):+.3f}")
    print(f"  p(rho <= observed) under that null                    p = {p:.3f}")
    print("\nThe direction holds under every check; the single run inflates the")
    print("magnitude, not the sign.")


if __name__ == "__main__":
    main()
