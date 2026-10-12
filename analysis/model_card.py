#!/usr/bin/env python3
"""
Evidence pack for one model: the [A] axis, "which architecture for which model".

Answers, for a single model: what it is, how it fails, which architectures help
and hurt it, whether each architecture's machinery actually functioned on it,
and which specific tasks changed hands.

    python3 analysis/model_card.py --list
    python3 analysis/model_card.py qwen-32b
"""
from __future__ import annotations
import argparse, csv, os, statistics as st
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(REPO, "data")
ARCHS = ["g1_baseline", "g1_prompt", "g1_temp",
         "g2_explorer", "g2_planner", "g2_analyst", "g2_monitor",
         "g3_instruction", "g3_reflexion", "g3_judge", "g3_gate",
         "g4_cooperation", "g4_debate", "g4_orchestra"]
MODELS = ["qwen-7b", "llama-70b", "qwen-14b", "devstral-small-2-24b",
          "qwen-72b", "qwen-32b", "gemini"]
SH = {a: a.split("_")[1][:7] for a in ARCHS}


def load():
    with open(os.path.join(FROZEN, "runs.csv")) as fh:
        runs = [r for r in csv.DictReader(fh) if r["track"] == "homogeneous"]
    with open(os.path.join(FROZEN, "tasks.csv")) as fh:
        tasks = {t["task_id"]: t for t in csv.DictReader(fh)}
    return runs, tasks


def mean(rows, col):
    v = [float(r[col]) for r in rows if r[col] not in ("", "None")]
    return st.mean(v) if v else float("nan")


def pct(rows, col, val="1"):
    v = [r for r in rows if r[col] != ""]
    return 100 * sum(1 for r in v if r[col] == val) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    runs, tasks = load()

    if args.list or not args.model:
        print("models (ordered by baseline capability):\n")
        for m in MODELS:
            b = [r for r in runs if r["model"] == m and r["arch"] == "g1_baseline"]
            print(f"  {m:24s} baseline {sum(1 for r in b if r['resolved']=='1')}/{len(b)}")
        return

    m = args.model
    M = [r for r in runs if r["model"] == m]
    if not M:
        raise SystemExit(f"no such model: {m}")
    T = [r for r in M if r["has_trace"] == "1"]
    bar = "=" * 78
    print(f"{bar}\nMODEL CARD: {m}\n{bar}")

    head = M[0]
    print(f"\nfamily={head['family']}  params={head['params_b']}B  tuning={head['tuning']}  "
          f"serving={head['serving']}  ctx={head['cfg_max_context_tokens']}")

    # ---- 1. the grid --------------------------------------------------------
    solved = {a: sum(1 for r in M if r["arch"] == a and r["resolved"] == "1") for a in ARCHS}
    sub = {a: 100 * mean([r for r in M if r["arch"] == a], "subtest_frac") for a in ARCHS}
    b = solved["g1_baseline"]
    print(f"\n## 1. WHAT HELPS AND WHAT HURTS  (baseline {b}/{sum(1 for r in M if r['arch'] == 'g1_baseline')}, {sub['g1_baseline']:.0f}% subtests)\n")
    print(f"{'arch':12s}{'solved':>8s}{'Δ':>5s}{'subtest%':>10s}{'Δpp':>7s}{'steps':>7s}{'ctx%':>6s}{'budget%':>9s}")
    for a in sorted(ARCHS, key=lambda x: -solved[x]):
        rows = [r for r in T if r["arch"] == a]
        c = Counter(r["term_class"] for r in rows)
        n = len(rows) or 1
        print(f"{SH[a]:12s}{solved[a]:>8d}{solved[a]-b:>+5d}{sub[a]:>9.0f}%{sub[a]-sub['g1_baseline']:>+7.0f}"
              f"{mean(rows,'steps'):>7.1f}{100*c['context_overflow']/n:>5.0f}%"
              f"{100*(c['budget_steps']+c['budget_cycles'])/n:>8.0f}%")

    # ---- 2. failure personality --------------------------------------------
    print("\n\n## 2. HOW IT FAILS (baseline runs only)\n")
    bl = [r for r in T if r["arch"] == "g1_baseline"]
    fails = [r for r in bl if r["resolved"] == "0"]
    c = Counter(r["term_class"] for r in fails)
    print(f"  {len(fails)} failures of {len(bl)} baseline runs; mean {mean(fails,'steps'):.1f} steps "
          f"(solved runs: {mean([r for r in bl if r['resolved']=='1'],'steps'):.1f})")
    for k, v in c.most_common():
        print(f"    {k:22s} {100*v/len(fails):5.0f}%")

    # ---- 3. did the machinery work? ----------------------------------------
    print("\n## 3. DID EACH ARCHITECTURE'S MACHINERY ACTUALLY FUNCTION?\n")
    e = [r for r in T if r["arch"] == "g2_explorer"]
    p = [r for r in T if r["arch"] == "g2_planner"]
    an = [r for r in T if r["arch"] == "g2_analyst"]
    mo = [r for r in T if r["arch"] == "g2_monitor"]
    g3 = [r for r in T if r["group"] == "g3"]
    print(f"  explorer  : valid data-summary {pct(e,'exp_summary_valid'):5.0f}%   "
          f"files inspected {mean(e,'exp_files_inspected'):.1f}")
    print(f"  planner   : valid plan         {pct(p,'plan_valid'):5.0f}%   "
          f"subtasks {mean(p,'plan_num_subtasks'):.2f}")
    print(f"  analyst   : valid summary      {pct(an,'exp_summary_valid'):5.0f}%")
    print(f"  monitor   : injections/run     {mean(mo,'monitor_injections'):.2f}   "
          f"calls {mean(mo,'monitor_calls'):.1f}")
    ran = sum(1 for r in g3 if r["s2_verifier_steps"] not in ("", "0", "None")
              and float(r["s2_verifier_steps"]) > 0)
    print(f"  G3 verify : intercepted {pct(g3,'s2_intercepted'):5.0f}%   "
          f"verifier actually ran {100*ran/len(g3) if g3 else 0:5.0f}%")
    j = [r for r in T if r["arch"] == "g3_judge" and r["s2_judge_approved"] != ""]
    if j:
        print(f"  judge     : verdicts {len(j)}   approved {pct(j,'s2_judge_approved'):.0f}%")
    orc = [r for r in T if r["arch"] == "g4_orchestra"]
    print(f"  orchestra : cycles/run {mean(orc,'orch_cycles'):.1f}   "
          f"hit cycle cap {100*sum(1 for r in orc if r['term_class']=='budget_cycles')/len(orc) if orc else 0:.0f}%")

    # ---- 4. which tasks moved ----------------------------------------------
    print("\n\n## 4. WHICH TASKS EACH ARCHITECTURE GAINED AND LOST\n")
    base = {r["task"] for r in M if r["arch"] == "g1_baseline" and r["resolved"] == "1"}
    for a in ARCHS[1:]:
        got = {r["task"] for r in M if r["arch"] == a and r["resolved"] == "1"}
        gained, lost = sorted(got - base), sorted(base - got)
        if gained or lost:
            print(f"  {SH[a]:10s} {len(got)-len(base):+d}")
            if gained:
                print(f"      + {', '.join(gained)}")
            if lost:
                print(f"      - {', '.join(lost)}")

    # ---- 5. category profile -----------------------------------------------
    print("\n## 5. WHERE ITS SOLVES LIVE (baseline, by category)\n")
    cats = Counter(tasks[t]["category"] for t in base if t in tasks)
    tot = Counter(tasks[t]["category"] for t in tasks)
    for k, v in cats.most_common():
        print(f"    {k:24s} {v}/{tot[k]}")
    print(f"\n{bar}")


if __name__ == "__main__":
    main()
