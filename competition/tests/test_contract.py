"""The submission must survive every failure path and still score.

On a code-execution platform a non-zero exit means the entrypoint never copies
``submission.csv`` out and the run is discarded. A bad prediction costs you
some log loss; an exception costs you the entire submission. These tests pin
that guarantee.

Run standalone (no pytest needed):

    python tests/test_contract.py --data-dir ../drivendataorg/tutoring-outcomes-runtime/data-demo
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSION_SRC = REPO / "submission_src"

# name -> shell command run inside the staged directory to break something
BREAKAGES: dict[str, str] = {
    "nothing (control)": "true",
    "corrupt model bundle": "mkdir -p model && head -c 500 /dev/urandom > model/bundle.pkl",
    "one transcript missing": "rm -f $(ls data/test_transcripts/*.csv | head -1)",
    "all transcripts missing": "rm -rf data/test_transcripts",
    "src package deleted": "rm -rf src",
    "test_features.csv missing": "rm -f data/test_features.csv",
    "malformed transcript": "f=$(ls data/test_transcripts/*.csv | head -1); "
                            "printf 'garbage,not,a,transcript\\n1,2\\n' > $f",
}


def stage(data_dir: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="contract_"))
    shutil.copytree(SUBMISSION_SRC, work, dirs_exist_ok=True)
    shutil.copytree(data_dir, work / "data")
    return work


def run_case(name: str, breakage: str, data_dir: Path) -> tuple[bool, str]:
    import pandas as pd

    work = stage(data_dir)
    try:
        subprocess.run(breakage, cwd=work, shell=True, capture_output=True)
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=work, capture_output=True, text=True, timeout=600, env=dict(os.environ),
        )
        if proc.returncode != 0:
            return False, f"exit={proc.returncode}: {(proc.stderr or '')[-200:]}"

        out = work / "submission.csv"
        if not out.exists():
            return False, "no submission.csv -> the platform would discard the run"

        sub = pd.read_csv(out)
        fmt = pd.read_csv(work / "data" / "submission_format.csv")
        if list(sub.columns) != list(fmt.columns):
            return False, f"columns {list(sub.columns)} != {list(fmt.columns)}"
        if sub["response_id"].tolist() != fmt["response_id"].tolist():
            return False, "response_id order does not match submission_format"
        p = sub["probability"]
        if not p.notna().all():
            return False, "NaN probabilities"
        if not ((p > 0) & (p < 1)).all():
            return False, "probabilities not strictly inside (0, 1)"
        return True, f"{len(sub)} rows, mean={p.mean():.4f}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=REPO.parent.parent / "drivendataorg/tutoring-outcomes-runtime/data-demo",
        help="A directory shaped like data-demo/",
    )
    args = ap.parse_args()
    if not (args.data_dir / "submission_format.csv").exists():
        print(f"No data at {args.data_dir}. Pass --data-dir pointing at a data-demo-shaped dir.")
        return 2

    print(f"Submission contract, data: {args.data_dir}\n")
    failures = 0
    width = max(len(n) for n in BREAKAGES)
    for name, breakage in BREAKAGES.items():
        ok, detail = run_case(name, breakage, args.data_dir)
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")

    print(f"\n{len(BREAKAGES) - failures}/{len(BREAKAGES)} passed")
    if failures:
        print("A failing case means that scenario would lose the whole submission.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
