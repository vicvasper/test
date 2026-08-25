"""Transcript -> feature table for the Trace the Ace tutoring-outcomes challenge.

Design constraints that come straight from the official runtime contract:

*   Pure ``pandas`` / ``numpy`` / ``re``. No network, no optional imports, no
    model downloads. This module must never be the reason a submission dies.
*   Deterministic. The same transcript always yields the same row.
*   Built per ``response_id``, not per session. One session is probed on several
    learning objectives, so session-global features alone throw away the signal
    that distinguishes two rows of the same session.

The feature groups are deliberately pedagogically legible: the challenge awards
prizes on write-up quality as well as leaderboard score, so every column here is
one we can defend as a claim about tutoring, not an anonymous embedding
dimension.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- Lexicons -------------------------------------------------------------
# Tutor evaluative feedback. The single strongest in-session proxy for whether
# the student was actually getting items right before the follow-up question.
PRAISE = re.compile(
    r"\b(correct|exactly|that'?s right|well done|perfect|excellent|brilliant|"
    r"fantastic|very good|great|good job|spot on|nice work|lovely|yes that'?s)\b",
    re.I,
)
CORRECTION = re.compile(
    r"\b(not quite|not right|isn'?t right|try again|almost|have another look|"
    r"let'?s check|careful|remember that|actually|close but|not exactly|"
    r"think again|have a look again)\b",
    re.I,
)
# Accountable-talk "press for reasoning" moves. Tutor asks the student to
# justify rather than just answer.
PRESS = re.compile(
    r"(how did you|how do you know|can you explain|why do you|why is|"
    r"tell me how|talk me through|what made you|explain how|explain why|"
    r"how did that|show me how)",
    re.I,
)
# Student epistemic hedging / low confidence.
HEDGE = re.compile(
    r"\b(um+|uh+|erm+|i think|i guess|not sure|maybe|dunno|don'?t know|"
    r"is it|would it be|i'?m not|probably)\b",
    re.I,
)
# Student causal / self-explanation connectives (Chi's self-explanation effect).
EXPLAIN = re.compile(r"\b(because|so that|therefore|since|that'?s why|which means|so the)\b", re.I)
UNCLEAR = re.compile(r"\[unclear\]|\[inaudible\]", re.I)
NUMERIC_ONLY = re.compile(r"^[\s\d.,:/x*+\-=%$£]+$")
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or that the to with
    up down out over under how what when which who why using use used than then""".split()
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in STOPWORDS and len(w) > 2}


def _to_seconds(ts) -> float:
    """Parse HH:MM:SS. Transcripts are ASR output, so be forgiving."""
    if not isinstance(ts, str):
        return np.nan
    parts = ts.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return np.nan
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] if nums else np.nan


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if b else default


def _turn_stats(turns: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Feature block computed over an arbitrary slice of a transcript.

    Called once for the whole session and once for the objective-relevant
    window, so the same vocabulary describes both scopes.
    """
    student = turns[turns["role"] == "student"]
    tutor = turns[turns["role"] == "tutor"]
    n_s, n_t = len(student), len(tutor)

    s_text = " ".join(student["content"].astype(str))
    t_text = " ".join(tutor["content"].astype(str))
    s_lens = student["content"].astype(str).str.split().str.len()

    praise = len(PRAISE.findall(t_text))
    correction = len(CORRECTION.findall(t_text))
    press = len(PRESS.findall(t_text))

    out = {
        f"{prefix}n_turns": float(len(turns)),
        f"{prefix}n_student": float(n_s),
        f"{prefix}n_tutor": float(n_t),
        f"{prefix}student_share": _safe_div(n_s, n_s + n_t),
        f"{prefix}student_words": float(s_lens.sum() if n_s else 0.0),
        f"{prefix}tutor_words": float(len(_words(t_text))),
        f"{prefix}student_words_per_turn": float(s_lens.mean()) if n_s else 0.0,
        f"{prefix}student_words_max": float(s_lens.max()) if n_s else 0.0,
        f"{prefix}word_ratio": _safe_div(s_lens.sum() if n_s else 0, len(_words(t_text)), 0.0),
        # Evaluative feedback -- the correctness proxy.
        f"{prefix}praise": float(praise),
        f"{prefix}correction": float(correction),
        f"{prefix}praise_rate": _safe_div(praise, n_s),
        f"{prefix}correction_rate": _safe_div(correction, n_s),
        f"{prefix}praise_minus_correction": float(praise - correction),
        f"{prefix}praise_ratio": _safe_div(praise, praise + correction, 0.5),
        # Accountable talk.
        f"{prefix}press": float(press),
        f"{prefix}press_rate": _safe_div(press, n_t),
        f"{prefix}tutor_questions": float(t_text.count("?")),
        f"{prefix}tutor_question_rate": _safe_div(t_text.count("?"), n_t),
        # Student epistemic state.
        f"{prefix}hedge": float(len(HEDGE.findall(s_text))),
        f"{prefix}hedge_rate": _safe_div(len(HEDGE.findall(s_text)), n_s),
        f"{prefix}explain": float(len(EXPLAIN.findall(s_text))),
        f"{prefix}explain_rate": _safe_div(len(EXPLAIN.findall(s_text)), n_s),
        f"{prefix}student_questions": float(s_text.count("?")),
        f"{prefix}student_question_rate": _safe_div(s_text.count("?"), n_s),
        # Transcription quality is a confounder, not a learning signal -- but it
        # correlates with both, so the model needs it explicitly to avoid
        # attributing audio noise to pedagogy.
        f"{prefix}unclear": float(len(UNCLEAR.findall(" ".join(turns["content"].astype(str))))),
        f"{prefix}unclear_rate": _safe_div(
            len(UNCLEAR.findall(" ".join(turns["content"].astype(str)))), len(turns)
        ),
    }

    # Bare numeric answers = recall; prose = reasoning.
    if n_s:
        numeric = student["content"].astype(str).str.strip().str.match(NUMERIC_ONLY).sum()
        out[f"{prefix}numeric_answer_rate"] = _safe_div(numeric, n_s)
    else:
        out[f"{prefix}numeric_answer_rate"] = 0.0
    return out


def _response_after_press(turns: pd.DataFrame) -> dict[str, float]:
    """How substantively does the student answer when pressed to explain?

    A student who produces a long causal answer after 'how did you work that
    out?' has understanding a bare correct answer does not evidence.
    """
    roles = turns["role"].tolist()
    contents = turns["content"].astype(str).tolist()
    lengths, explained = [], 0
    for i, (role, text) in enumerate(zip(roles, contents)):
        if role != "tutor" or not PRESS.search(text):
            continue
        for j in range(i + 1, min(i + 3, len(roles))):
            if roles[j] == "student":
                lengths.append(len(contents[j].split()))
                explained += bool(EXPLAIN.search(contents[j]))
                break
    return {
        "press_reply_words_mean": float(np.mean(lengths)) if lengths else 0.0,
        "press_reply_words_max": float(np.max(lengths)) if lengths else 0.0,
        "press_reply_explained_rate": _safe_div(explained, len(lengths)),
        "press_answered": float(len(lengths)),
    }


def _trajectory(turns: pd.DataFrame) -> dict[str, float]:
    """Late-session state beats session-average state.

    The follow-up question comes after the session, so the student's condition
    in the final third is a better predictor than the mean over the whole hour.
    """
    n = len(turns)
    if n < 6:
        return {
            "late_praise_rate": 0.0,
            "late_correction_rate": 0.0,
            "praise_rate_delta": 0.0,
            "late_student_words_per_turn": 0.0,
            "student_words_delta": 0.0,
            "late_hedge_rate": 0.0,
            "hedge_rate_delta": 0.0,
        }
    third = max(1, n // 3)
    early = _turn_stats(turns.iloc[:third], "e_")
    late = _turn_stats(turns.iloc[-third:], "l_")
    return {
        "late_praise_rate": late["l_praise_rate"],
        "late_correction_rate": late["l_correction_rate"],
        "praise_rate_delta": late["l_praise_rate"] - early["e_praise_rate"],
        "late_student_words_per_turn": late["l_student_words_per_turn"],
        "student_words_delta": (
            late["l_student_words_per_turn"] - early["e_student_words_per_turn"]
        ),
        "late_hedge_rate": late["l_hedge_rate"],
        "hedge_rate_delta": late["l_hedge_rate"] - early["e_hedge_rate"],
    }


def _relevance(turns: pd.DataFrame, objective: str) -> np.ndarray:
    """Per-turn lexical relevance to the learning objective.

    Deliberately lexical rather than neural: it costs nothing at inference, it
    is reproducible, and it is the fallback if the embedding path is disabled.
    """
    target = _content_words(objective or "")
    if not target:
        return np.zeros(len(turns), dtype=float)
    scores = np.array(
        [len(_content_words(str(t)) & target) for t in turns["content"]], dtype=float
    )
    return scores / len(target)


def _window(turns: pd.DataFrame, rel: np.ndarray, radius: int = 12) -> pd.DataFrame:
    """Contiguous span of the session most about this learning objective.

    This is what makes two responses from the same session differ. Without it,
    every row of a session gets identical features and the model can only ever
    learn a session-level average.
    """
    if len(turns) == 0 or rel.max() <= 0:
        return turns
    kernel = np.ones(min(radius, len(turns)))
    smoothed = np.convolve(rel, kernel, mode="same")
    centre = int(np.argmax(smoothed))
    lo, hi = max(0, centre - radius), min(len(turns), centre + radius + 1)
    return turns.iloc[lo:hi]


def transcript_features(
    turns: pd.DataFrame, objective: str, n_objectives_in_session: int = 1
) -> dict[str, float]:
    """Full feature row for one (session, learning objective) pair."""
    turns = turns.copy()
    turns["role"] = turns["role"].astype(str).str.strip().str.lower()
    turns["content"] = turns["content"].fillna("")

    feats: dict[str, float] = {}
    feats.update(_turn_stats(turns, "s_"))          # whole session
    feats.update(_response_after_press(turns))
    feats.update(_trajectory(turns))

    rel = _relevance(turns, objective)
    feats.update(_turn_stats(_window(turns, rel), "w_"))  # objective-relevant window

    secs = turns["timestamp"].map(_to_seconds) if "timestamp" in turns else pd.Series(dtype=float)
    secs = secs.dropna()
    feats["duration_s"] = float(secs.max() - secs.min()) if len(secs) > 1 else 0.0
    feats["turns_per_min"] = _safe_div(len(turns), max(feats["duration_s"], 1.0) / 60.0)

    feats["rel_max"] = float(rel.max()) if len(rel) else 0.0
    feats["rel_mean"] = float(rel.mean()) if len(rel) else 0.0
    feats["rel_covered"] = float((rel > 0).sum())
    feats["rel_covered_rate"] = _safe_div((rel > 0).sum(), len(rel))
    feats["objective_words"] = float(len(_content_words(objective or "")))
    feats["n_objectives_in_session"] = float(n_objectives_in_session)
    return feats


def build_feature_frame(features: pd.DataFrame, load_transcript) -> pd.DataFrame:
    """Vectorised over the features table; one transcript read per session.

    ``load_transcript(session_id) -> DataFrame | None``. A missing or unreadable
    transcript yields an all-zero row rather than an exception: on the platform
    a single bad file must not cost the whole submission.
    """
    per_session = features.groupby("session_id")["response_id"].transform("size")
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for (_, row), n_obj in zip(features.iterrows(), per_session):
        sid = row["session_id"]
        if sid not in cache:
            try:
                cache[sid] = load_transcript(sid)
            except Exception:
                cache[sid] = None
        turns = cache[sid]
        if turns is None or len(turns) == 0:
            rows.append({"response_id": row["response_id"], "transcript_missing": 1.0})
            continue
        try:
            feats = transcript_features(turns, str(row.get("learning_objective", "")), int(n_obj))
            feats["transcript_missing"] = 0.0
        except Exception:
            feats = {"transcript_missing": 1.0}
        feats["response_id"] = row["response_id"]
        rows.append(feats)

    frame = pd.DataFrame(rows).set_index("response_id")
    return frame.astype(float).fillna(0.0)
