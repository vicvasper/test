"""Offline training + the validation harness that decides whether you land in
the top 15 or around rank 240.

Run:
    python training/train.py --data-dir /path/to/competition/data

The competition scores log loss on a held-out test set. The only thing that
tells you, before you spend a submission, whether a change helps is local
cross-validation -- and local CV is only informative if its split matches the
structure of the real test split. This script is built around three claims:

1.  ``test_features.csv`` carries several ``response_id`` rows per
    ``session_id``. A plain ``KFold`` therefore puts rows from the same session
    on both sides of the split. Session-level features (tutor style, audio
    quality, student identity) then leak, local CV improves, the leaderboard
    does not, and you tune in the wrong direction for days. ``GroupKFold`` on
    ``session_id`` is the fix.

2.  Target-encoding ``learning_objective_id`` on the full training set leaks the
    label into its own feature. It must be refitted inside every fold.

3.  Any model must be compared against the trivial baselines before it is
    believed. A model that cannot beat the per-objective base rate is not a
    model.

The script prints the leakage gap explicitly, because that number is the
diagnosis for a leaderboard result far worse than local CV predicted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "submission_src"))

from src.features import build_feature_frame  # noqa: E402
from src.model import Bundle, PortableLGBM  # noqa: E402

N_SPLITS = 5
SEED = 20260825
# Strength of the shrink toward the global rate when target-encoding an
# objective. Objectives seen a handful of times must not be trusted.
SMOOTHING = 20.0


def smoothed_objective_prior(
    df: pd.DataFrame, label_col: str, global_prior: float, smoothing: float = SMOOTHING
) -> dict:
    """Empirical-Bayes shrink of each objective's success rate to the global rate."""
    grouped = df.groupby("learning_objective_id")[label_col].agg(["sum", "count"])
    shrunk = (grouped["sum"] + smoothing * global_prior) / (grouped["count"] + smoothing)
    return shrunk.to_dict()


def add_prior_feature(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    label_col: str,
) -> tuple[pd.Series, dict, float]:
    """Fit the objective prior on ``train_df`` only, apply it to ``apply_df``."""
    global_prior = float(train_df[label_col].mean())
    prior_map = smoothed_objective_prior(train_df, label_col, global_prior)
    applied = (
        apply_df["learning_objective_id"].map(prior_map).astype(float).fillna(global_prior)
    )
    return applied, prior_map, global_prior


def fit_fold(X_tr, y_tr, X_va, y_va):
    """One fold. LightGBM when available, sklearn otherwise.

    Both are wrapped so the caller gets ``predict_proba`` and a portable export.
    """
    try:
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "verbosity": -1,
            "seed": SEED,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        return PortableLGBM.from_booster(booster)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=5.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        )
        model.fit(X_tr, y_tr)
        return model


def cross_validate(X, y, groups, splitter, label: str, feature_cols, meta):
    """Run one CV scheme and return (oof predictions, fold models)."""
    oof = np.full(len(y), np.nan)
    models = []
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        meta_tr, meta_va = meta.iloc[tr], meta.iloc[va]

        # Refit the objective prior inside the fold. Doing this once outside the
        # loop is the single most common source of an over-optimistic CV score.
        prior_tr, _, _ = add_prior_feature(meta_tr, meta_tr, "label")
        prior_va, _, _ = add_prior_feature(meta_tr, meta_va, "label")

        X_tr = X.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        X_tr["objective_prior"] = prior_tr.to_numpy()
        X_va["objective_prior"] = prior_va.to_numpy()
        X_tr = X_tr[feature_cols]
        X_va = X_va[feature_cols]

        model = fit_fold(X_tr, y[tr], X_va, y[va])
        oof[va] = model.predict_proba(X_va)[:, 1]
        models.append(model)
        print(f"  {label} fold {fold}: logloss={log_loss(y[va], oof[va]):.5f}  n_val={len(va)}")
    return oof, models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory with train_* files")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "submission_src" / "model")
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "training" / ".feature_cache.pkl")
    args = ap.parse_args()

    features = pd.read_csv(args.data_dir / "train_features.csv")
    labels = pd.read_csv(args.data_dir / "train_labels.csv")
    df = features.merge(labels, on="response_id", how="inner")
    label_col = [c for c in labels.columns if c != "response_id"][0]
    df = df.rename(columns={label_col: "label"})
    print(f"Loaded {len(df):,} labelled responses over {df.session_id.nunique():,} sessions")
    print(f"Global base rate: {df['label'].mean():.4f}")

    # Feature building dominates runtime; cache it so tuning iterations are fast.
    if args.cache.exists():
        X = pd.read_pickle(args.cache)
        print(f"Loaded cached features {X.shape} from {args.cache}")
    else:
        def load_transcript(sid):
            return pd.read_csv(args.data_dir / "train_transcripts" / f"{sid}.csv")

        X = build_feature_frame(df, load_transcript)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        X.to_pickle(args.cache)
        print(f"Built and cached features {X.shape}")

    X = X.reindex(df["response_id"]).reset_index(drop=True)
    y = df["label"].to_numpy()
    meta = df[["response_id", "session_id", "learning_objective_id", "label"]].reset_index(drop=True)
    feature_cols = list(X.columns) + ["objective_prior"]

    # --- Baselines. Nothing counts as progress until it beats these. --------
    global_prior = float(y.mean())
    print("\n=== Baselines ===")
    print(f"  constant 0.5      : {log_loss(y, np.full(len(y), 0.5)):.5f}   <-- what the")
    print("                                   unmodified example submission scores")
    print(f"  global base rate  : {log_loss(y, np.full(len(y), global_prior)):.5f}")

    prior_oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, va in gkf.split(X, y, meta["session_id"]):
        p, _, _ = add_prior_feature(meta.iloc[tr], meta.iloc[va], "label")
        prior_oof[va] = p.to_numpy()
    print(f"  objective prior   : {log_loss(y, prior_oof):.5f}")

    # --- The leakage diagnostic -------------------------------------------
    print(f"\n=== Grouped CV (GroupKFold on session_id) -- trust this one ===")
    oof_grouped, models = cross_validate(
        X, y, meta["session_id"], gkf, "grouped", feature_cols, meta
    )
    grouped_score = log_loss(y, oof_grouped)

    print(f"\n=== Naive CV (plain KFold, sessions split across folds) ===")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_naive, _ = cross_validate(X, y, None, kf, "naive", feature_cols, meta)
    naive_score = log_loss(y, oof_naive)

    gap = grouped_score - naive_score
    print("\n=== Verdict ===")
    print(f"  grouped CV logloss : {grouped_score:.5f}  (expect the leaderboard near this)")
    print(f"  naive CV logloss   : {naive_score:.5f}")
    print(f"  leakage gap        : {gap:+.5f}")
    if gap > 0.01:
        print("  >> Naive CV is optimistic by more than 0.01. If you were tuning against")
        print("     it, your leaderboard score was never going to match. Use grouped CV.")
    print(f"  vs objective prior : {log_loss(y, prior_oof) - grouped_score:+.5f} improvement")
    print(f"  vs constant 0.5    : {log_loss(y, np.full(len(y), 0.5)) - grouped_score:+.5f} improvement")

    # --- Calibration check -------------------------------------------------
    print("\n=== Calibration (grouped OOF, decile bins) ===")
    bins = pd.qcut(oof_grouped, 10, duplicates="drop")
    cal = pd.DataFrame({"p": oof_grouped, "y": y}).groupby(bins, observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")
    )
    print(cal.round(4).to_string())
    drift = float((cal["predicted"] - cal["actual"]).abs().max())
    print(f"  max |predicted - actual| = {drift:.4f}"
          + ("  <-- consider isotonic calibration" if drift > 0.05 else "  (well calibrated)"))

    # --- Export ------------------------------------------------------------
    prior_map = smoothed_objective_prior(meta, "label", global_prior)
    bundle = Bundle(models, feature_cols, prior_map, global_prior)
    args.out.mkdir(parents=True, exist_ok=True)
    bundle.save(args.out / "bundle.pkl")
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "grouped_cv_logloss": grouped_score,
                "naive_cv_logloss": naive_score,
                "leakage_gap": gap,
                "objective_prior_logloss": float(log_loss(y, prior_oof)),
                "constant_half_logloss": float(log_loss(y, np.full(len(y), 0.5))),
                "global_prior": global_prior,
                "n_features": len(feature_cols),
                "n_rows": int(len(y)),
            },
            indent=2,
        )
    )
    print(f"\nSaved bundle ({len(models)} fold models) to {args.out / 'bundle.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
