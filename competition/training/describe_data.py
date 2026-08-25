"""Print a shareable diagnostic of the competition data. Content never leaves your machine.

    python training/describe_data.py --data-dir "C:\\path\\to\\drivendata_real\\v2"

Emits only schema, shapes, counts and distributions -- no transcript text, no ids,
no labels per row. The official runtime README says "Do not push any actual data
to GitHub", and that applies to pasting it into a chat too. This output is safe
to share; the CSVs are not.

It answers the questions that decide how the model must be built, and that
cannot be answered without the real data:

*   Is there a student id? If yes, grouping by session alone still leaks,
    because student ability is the dominant latent variable.
*   How many responses share a session? That number sets how badly a plain
    KFold overstates your score.
*   Do the test learning objectives appear in train? If not, the per-objective
    prior has nothing to transfer and needs a fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def show_schema(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name}: {df.shape[0]:,} rows x {df.shape[1]} cols ---")
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        n_null = int(df[col].isna().sum())
        example_len = ""
        if df[col].dtype == object:
            lengths = df[col].astype(str).str.len()
            example_len = f", len p50={lengths.median():.0f} p95={lengths.quantile(.95):.0f}"
        print(
            f"  {col:<28} dtype={str(df[col].dtype):<10} "
            f"unique={n_unique:<8,} nulls={n_null:<6,}{example_len}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    args = ap.parse_args()
    d = args.data_dir

    print(f"Data directory: {d}")
    print("Files present:")
    for p in sorted(d.iterdir()):
        kind = "dir" if p.is_dir() else f"{p.stat().st_size / 2**20:.1f} MiB"
        n = f" ({len(list(p.iterdir())):,} files)" if p.is_dir() else ""
        print(f"  {p.name:<32} {kind}{n}")

    train = pd.read_csv(d / "train_features.csv") if (d / "train_features.csv").exists() else None
    labels = pd.read_csv(d / "train_labels.csv") if (d / "train_labels.csv").exists() else None
    test = pd.read_csv(d / "test_features.csv") if (d / "test_features.csv").exists() else None

    for name, df in (("train_features", train), ("train_labels", labels), ("test_features", test)):
        if df is not None:
            show_schema(name, df)

    if labels is not None:
        label_col = [c for c in labels.columns if c != "response_id"][0]
        print(f"\n--- label ---")
        print(f"  column: {label_col}")
        print(f"  base rate: {labels[label_col].mean():.4f}")
        print(f"  value counts:\n{labels[label_col].value_counts().to_string()}")

    if train is not None:
        print("\n--- GROUP STRUCTURE (this decides the CV split) ---")
        # Any column that repeats across responses is a grouping candidate.
        for col in train.columns:
            if col == "response_id" or train[col].nunique() >= len(train) * 0.95:
                continue
            per = train.groupby(col).size()
            print(
                f"  {col:<28} {train[col].nunique():>7,} groups, "
                f"responses/group: mean={per.mean():.2f} max={per.max()} "
                f"p95={per.quantile(.95):.0f}"
            )
        id_like = [c for c in train.columns if "student" in c.lower() or "tutor" in c.lower()]
        print(f"\n  student/tutor id columns found: {id_like or 'NONE'}")
        if id_like:
            print("  >> Group your CV on these too, not just session_id.")
        else:
            print("  >> No student id. GroupKFold on session_id is the strongest split available.")

    if train is not None and test is not None and "learning_objective_id" in test.columns:
        seen = set(train["learning_objective_id"])
        test_obj = set(test["learning_objective_id"])
        overlap = len(test_obj & seen)
        print("\n--- OBJECTIVE TRANSFER (this decides the prior's usefulness) ---")
        print(f"  train objectives: {len(seen):,}")
        print(f"  test objectives:  {len(test_obj):,}")
        print(f"  test objectives seen in train: {overlap:,} / {len(test_obj):,} "
              f"({overlap / max(len(test_obj), 1):.1%})")
        if overlap < len(test_obj):
            print("  >> Some test objectives are unseen. The per-objective prior must fall back")
            print("     to the global rate for those, and text features carry the load instead.")

    tdir = next((p for p in d.iterdir() if p.is_dir() and "transcript" in p.name), None)
    if tdir:
        files = sorted(tdir.glob("*.csv"))
        print(f"\n--- TRANSCRIPTS ({tdir.name}) ---")
        print(f"  files: {len(files):,}")
        sample = files[: min(50, len(files))]
        rows, roles = [], set()
        for f in sample:
            try:
                t = pd.read_csv(f)
                rows.append(len(t))
                roles |= set(t["role"].astype(str).str.lower().unique())
            except Exception as exc:
                print(f"  UNREADABLE: {f.name}: {type(exc).__name__}")
        if rows:
            s = pd.Series(rows)
            print(f"  turns per transcript (first {len(sample)}): "
                  f"min={s.min()} p50={s.median():.0f} p95={s.quantile(.95):.0f} max={s.max()}")
            print(f"  roles observed: {sorted(roles)}")
            print(f"  columns: {list(pd.read_csv(sample[0]).columns)}")

    print("\nPaste this whole output. It contains no transcript text and no per-row labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
