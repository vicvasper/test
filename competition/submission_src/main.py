"""Trace the Ace -- inference entrypoint.

Runtime contract (from runtime/entrypoint.sh in the official runtime repo):

*   This file must sit at the ROOT of submission.zip. The entrypoint does
    ``cd /code_execution`` then ``python main.py``.
*   Competition data is mounted read-only at ``./data``.
*   The only accepted output is ``./submission.csv``, written next to this file.
*   There is no internet. Nothing may be downloaded, installed, or called out to.
*   A non-zero exit means the entrypoint never copies the result out: the
    submission is lost. So this module catches everything and always writes a
    valid file.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_ROOT))

DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submission.csv")
MODEL_PATH = SRC_ROOT / "model" / "bundle.pkl"

# Used only if the data itself is unreadable. Overridden by the trained bundle's
# fitted base rate whenever the bundle loads.
FALLBACK_PRIOR = 0.5

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru ships in the runtime
    class _Logger:
        def _emit(self, level, msg):
            print(f"{level:8} | {msg}", flush=True)

        info = lambda self, m: self._emit("INFO", m)          # noqa: E731
        warning = lambda self, m: self._emit("WARNING", m)    # noqa: E731
        error = lambda self, m: self._emit("ERROR", m)        # noqa: E731
        success = lambda self, m: self._emit("SUCCESS", m)    # noqa: E731

    logger = _Logger()


def load_transcript(session_id: str) -> pd.DataFrame | None:
    path = DATA_DIR / "test_transcripts" / f"{session_id}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_submission(submission_format: pd.DataFrame, probabilities) -> None:
    """Write predictions in exactly the shape the scorer expects.

    Row order and the ``response_id`` set come from submission_format, never
    from our own frame. Misalignment here scores like noise while looking
    completely healthy in the logs -- it is the failure mode that silently
    buries an otherwise good model.
    """
    out = submission_format.copy()
    out["probability"] = probabilities
    out = out[list(submission_format.columns)]
    out.to_csv(SUBMISSION_PATH, index=False)
    logger.success(
        f"Wrote {len(out):,} predictions to {SUBMISSION_PATH} "
        f"(min={out['probability'].min():.4f}, "
        f"mean={out['probability'].mean():.4f}, "
        f"max={out['probability'].max():.4f})"
    )


def main() -> None:
    submission_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    logger.info(f"Loaded submission_format.csv of shape {submission_format.shape}")

    # Baseline that is always available, so every later step is an improvement
    # on something valid rather than a prerequisite for producing anything.
    probabilities = pd.Series(FALLBACK_PRIOR, index=submission_format.index, dtype=float)

    try:
        from src.features import build_feature_frame
        from src.model import Bundle, objective_prior_column, predict

        features = pd.read_csv(DATA_DIR / "test_features.csv")
        logger.info(f"Loaded test_features.csv of shape {features.shape}")

        bundle = None
        if MODEL_PATH.exists():
            try:
                bundle = Bundle.load(MODEL_PATH)
                logger.info(
                    f"Loaded bundle: {len(bundle.models)} model(s), "
                    f"{len(bundle.feature_names)} features, "
                    f"global_prior={bundle.global_prior:.4f}"
                )
            except Exception:
                logger.error(f"Bundle failed to load, using prior only:\n{traceback.format_exc()}")
        else:
            logger.warning(f"No model bundle at {MODEL_PATH}; predicting the prior.")

        global_prior = bundle.global_prior if bundle else FALLBACK_PRIOR
        objective_prior = bundle.objective_prior if bundle else {}
        prior = objective_prior_column(features, objective_prior, global_prior)

        X = build_feature_frame(features, load_transcript)
        logger.info(f"Built feature matrix of shape {X.shape}")

        X = X.reindex(features["response_id"])
        prior.index = features["response_id"]
        preds = predict(bundle, X, prior)

        # Align to the submission format by response_id. Any id we somehow
        # failed to predict falls back to the prior rather than to NaN.
        by_id = pd.Series(preds, index=features["response_id"])
        probabilities = (
            submission_format["response_id"].map(by_id).fillna(global_prior).astype(float)
        )
        logger.info(f"Predicted {probabilities.notna().sum():,} responses")

    except Exception:
        logger.error(f"Prediction path failed; falling back to prior.\n{traceback.format_exc()}")

    write_submission(submission_format, probabilities.to_numpy())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Absolute last resort: emit the submission format unchanged so the run
        # still produces a scoreable file and exits 0.
        logger.error(f"Fatal error in main():\n{traceback.format_exc()}")
        fmt = pd.read_csv(DATA_DIR / "submission_format.csv")
        fmt.to_csv(SUBMISSION_PATH, index=False)
        logger.warning("Emitted submission_format.csv unchanged as an emergency fallback.")
