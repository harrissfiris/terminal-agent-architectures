#!/usr/bin/env python3
"""Merge multiple partial terminal-bench runs into a single unified run folder.

Usage:
    python scripts/_merge_runs.py <run_dir1> <run_dir2> [run_dir3 ...] -o <output_dir>

Task directories are copied (cp -r) into the output.
Summary files (results.json, run_metadata.json, run.log, tb.lock) are merged.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

SUMMARY_FILES = {"results.json", "run_metadata.json", "run.log", "tb.lock"}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {path.name}")


def get_task_dirs(run_dir: Path) -> list[str]:
    """Return task directory names (everything that's not a summary file)."""
    return sorted(
        d.name
        for d in run_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def merge(run_dirs: list[Path], out_dir: Path):
    if out_dir.exists():
        print(f"ERROR: Output directory already exists: {out_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True)
    print(f"Created output: {out_dir}\n")

    # ── 1. Copy task directories ──────────────────────────────────────
    all_tasks = []
    for rd in run_dirs:
        tasks = get_task_dirs(rd)
        print(f"{rd.name}: {len(tasks)} tasks")
        for t in tasks:
            src = rd / t
            dst = out_dir / t
            if dst.exists():
                print(f"  WARNING: duplicate task {t}, skipping from {rd.name}")
                continue
            try:
                shutil.copytree(src, dst, symlinks=True, ignore_dangling_symlinks=True)
            except shutil.Error as e:
                print(f"  WARNING: copy errors for {t} (continuing): {e}")
            all_tasks.append(t)
    print(f"\nCopied {len(all_tasks)} task directories total.\n")

    # ── 2. Merge results.json (dedup by task_id, first occurrence wins) ─
    all_results = []
    seen_task_ids: set[str] = set()
    for rd in run_dirs:
        rj = rd / "results.json"
        if rj.exists():
            data = load_json(rj)
            for r in data.get("results", []):
                tid = r.get("task_id", "")
                if tid in seen_task_ids:
                    print(f"  WARNING: duplicate result for {tid}, skipping from {rd.name}")
                    continue
                seen_task_ids.add(tid)
                all_results.append(r)

    resolved = [r for r in all_results if r.get("is_resolved")]
    unresolved = [r for r in all_results if not r.get("is_resolved")]

    merged_results = {
        "id": run_dirs[0].name,  # use first run's id
        "results": sorted(all_results, key=lambda r: r.get("task_id", "")),
        "pass_at_k": 1,
        "n_resolved": len(resolved),
        "n_unresolved": len(unresolved),
        "resolved_ids": sorted(r["task_id"] for r in resolved),
        "unresolved_ids": sorted(r["task_id"] for r in unresolved),
        "accuracy": round(len(resolved) / len(all_results) * 100, 2)
        if all_results
        else 0,
    }
    save_json(out_dir / "results.json", merged_results)
    print(
        f"  → {merged_results['n_resolved']} resolved / "
        f"{len(all_results)} total = {merged_results['accuracy']}%\n"
    )

    # ── 3. Merge run_metadata.json ────────────────────────────────────
    # Base on first run's metadata, update aggregated fields
    meta = load_json(run_dirs[0] / "run_metadata.json")
    meta["run_id"] = out_dir.name
    meta["dataset_size"] = len(all_results)
    meta["accuracy"] = merged_results["accuracy"]

    # Collect all task_ids from results (authoritative)
    meta["task_ids"] = sorted(r["task_id"] for r in all_results)

    # Timestamps: earliest start, latest end
    starts, ends = [], []
    for rd in run_dirs:
        m = load_json(rd / "run_metadata.json")
        if m.get("start_time"):
            starts.append(m["start_time"])
        if m.get("end_time"):
            ends.append(m["end_time"])
    if starts:
        meta["start_time"] = min(starts)
    if ends:
        meta["end_time"] = max(ends)

    save_json(out_dir / "run_metadata.json", meta)

    # ── 4. Concatenate run.log ────────────────────────────────────────
    with open(out_dir / "run.log", "w") as out_log:
        for i, rd in enumerate(run_dirs):
            log_path = rd / "run.log"
            if log_path.exists():
                if i > 0:
                    out_log.write(f"\n{'='*80}\n")
                    out_log.write(f"=== MERGED FROM: {rd.name} ===\n")
                    out_log.write(f"{'='*80}\n\n")
                out_log.write(log_path.read_text())
    print(f"  Wrote run.log (concatenated)\n")

    # ── 5. Merge tb.lock ─────────────────────────────────────────────
    # Base on first run's lock, update task list and run_id
    lock = load_json(run_dirs[0] / "tb.lock")

    # Rebuild invocation args with all task ids
    base_args = []
    for arg in lock["invocation"]["args"]:
        if arg == "--task-id":
            break
        base_args.append(arg)
    for tid in meta["task_ids"]:
        base_args.extend(["--task-id", tid])
    lock["invocation"]["args"] = base_args

    lock["dataset"]["task_ids"] = meta["task_ids"]
    lock["local_config"]["run_id"] = out_dir.name

    save_json(out_dir / "tb.lock", lock)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE: {out_dir}")
    print(f"  Tasks:    {len(all_tasks)}")
    print(f"  Resolved: {merged_results['n_resolved']}")
    print(f"  Accuracy: {merged_results['accuracy']}%")
    print(f"  Resolved: {', '.join(merged_results['resolved_ids'])}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Merge partial tb runs")
    parser.add_argument("runs", nargs="+", help="Run directories to merge")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    for rd in run_dirs:
        if not rd.is_dir():
            print(f"ERROR: {rd} is not a directory")
            sys.exit(1)

    merge(run_dirs, Path(args.output))


if __name__ == "__main__":
    main()
