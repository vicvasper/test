"""Model bundle loading and prediction, with a fallback that cannot fail.

The governing principle: on a code-execution platform, an exception is worth
infinitely more log loss than a mediocre prediction. Every path through this
module ends in a valid probability vector.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Log loss is unbounded as p -> 0 or 1. A single confident-and-wrong row can
# cost more than every other row combined, so we never emit a raw 0 or 1.
CLIP_LO, CLIP_HI = 1e-4, 1.0 - 1e-4


class Bundle:
    """Everything the submission needs at inference time, in one picklable object.

    Attributes
    ----------
    models:
        List of fitted estimators exposing ``predict_proba``. A list, not a single
        model, because the fold models are averaged -- refitting on all of train
        and submitting a single model throws away free variance reduction.
    feature_names:
        Exact training column order. Enforced at inference so a feature added
        later cannot silently shift the matrix.
    objective_prior:
        ``learning_objective_id`` -> smoothed historical success rate.
    global_prior:
        Overall base rate; used for unseen objectives and as the last resort.
    """

    def __init__(
        self,
        models: list,
        feature_names: list[str],
        objective_prior: dict,
        global_prior: float,
    ):
        self.models = models
        self.feature_names = list(feature_names)
        self.objective_prior = dict(objective_prior)
        self.global_prior = float(global_prior)

    def save(self, path: Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=5)

    @classmethod
    def load(cls, path: Path) -> "Bundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def objective_prior_column(
    features: pd.DataFrame, objective_prior: dict, global_prior: float
) -> pd.Series:
    """Difficulty prior for each row's learning objective.

    Objectives differ far more in intrinsic difficulty than sessions differ in
    quality, so this is a large share of the achievable signal. It must be
    fitted out of fold -- see ``training/train.py``.
    """
    col = features.get("learning_objective_id")
    if col is None:
        return pd.Series(global_prior, index=features.index, dtype=float)
    return col.map(objective_prior).astype(float).fillna(global_prior)


def predict(bundle: Bundle | None, X: pd.DataFrame, prior: pd.Series) -> np.ndarray:
    """Average the fold models; fall back to the prior on any failure."""
    if bundle is None or not bundle.models:
        return np.clip(prior.to_numpy(dtype=float), CLIP_LO, CLIP_HI)

    matrix = X.reindex(columns=bundle.feature_names, fill_value=0.0).astype(float)
    matrix = matrix.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    preds = []
    for model in bundle.models:
        try:
            p = model.predict_proba(matrix)[:, 1]
            if np.all(np.isfinite(p)):
                preds.append(p)
        except Exception:
            continue

    if not preds:
        return np.clip(prior.to_numpy(dtype=float), CLIP_LO, CLIP_HI)
    return np.clip(np.mean(preds, axis=0), CLIP_LO, CLIP_HI)


class PortableLGBM:
    """A LightGBM model stored as its own text format, not as a pickled object.

    Why this exists: the model is trained in your environment and unpickled in
    the competition runtime. Pickled ``sklearn``/``lightgbm`` estimator objects
    carry class layouts that change between library versions, so a version skew
    between the two environments raises at ``pickle.load`` -- inside the
    container, after the queue, with no way to patch it. The runtime pins its
    own versions and the changelog shows they move (pandas was downgraded 3.0.3
    -> 2.3.3 in July).

    LightGBM's text dump is a stable, documented interchange format. Storing
    that string sidesteps the whole class of failure.
    """

    def __init__(self, model_str: str, best_iteration: int | None = None):
        self.model_str = model_str
        self.best_iteration = best_iteration
        self._booster = None

    @classmethod
    def from_booster(cls, booster) -> "PortableLGBM":
        best = getattr(booster, "best_iteration", None) or None
        return cls(booster.model_to_string(), best)

    def _load(self):
        if self._booster is None:
            import lightgbm as lgb

            self._booster = lgb.Booster(model_str=self.model_str)
        return self._booster

    def predict_proba(self, X) -> np.ndarray:
        booster = self._load()
        p = booster.predict(X, num_iteration=self.best_iteration)
        p = np.asarray(p, dtype=float).ravel()
        return np.column_stack([1.0 - p, p])

    def __getstate__(self):
        # Never pickle the live booster handle.
        return {"model_str": self.model_str, "best_iteration": self.best_iteration}

    def __setstate__(self, state):
        self.model_str = state["model_str"]
        self.best_iteration = state.get("best_iteration")
        self._booster = None
