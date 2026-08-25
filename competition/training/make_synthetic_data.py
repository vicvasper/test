"""Generate a synthetic Trace the Ace dataset with known ground-truth structure.

Two uses:

*   Smoke-test the whole pipeline (features -> CV -> bundle -> inference) before
    the real data is downloaded, or on a machine that will never hold it.
*   Demonstrate the leakage the grouped CV is there to catch. Sessions here get
    a latent quality shared by every response they contain -- exactly the
    structure the real data has -- so a naive KFold can memorise a session from
    one of its rows and score the rest for free.

The generated transcripts are not a model of real tutoring and must never be
used to draw conclusions about tutoring. They exist to exercise code paths.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

OBJECTIVES = [
    ("hkrpmns", "Multiplying and dividing whole numbers and decimals by 10 and 100."),
    ("njclpqd", "Adding and subtracting fractions with different denominators."),
    ("wllbgmz", "Finding fractions of an amount."),
    ("mcrpwxl", "Rounding numbers to the nearest 100 and 1,000."),
    ("rpkdltf", "Comparing and ordering fractions."),
    ("qzmtnvb", "Converting between improper fractions and mixed numbers."),
    ("plkwsdr", "Identifying common factors and common multiples."),
    ("tzvbnmq", "Solving problems involving percentages of amounts."),
]
PRAISE_T = ["Exactly right.", "Very good.", "Perfect, well done.", "Correct.", "Excellent thinking."]
CORRECT_T = ["Not quite, try again.", "Almost, let's check that.", "Careful, have another look."]
PRESS_T = ["How did you work that out?", "Can you explain how?", "How do you know?"]
CONFIDENT_S = ["It is {n}.", "{n}.", "I got {n}."]
HEDGED_S = ["Um, is it {n}?", "I think maybe {n}?", "Not sure, {n}?"]
EXPLAIN_S = [
    "Because {a} times {b} is {n}, so that works.",
    "Since {a} goes into {n} exactly {b} times, so the answer is {b}.",
]


def _rng_choice(rng, seq):
    return seq[rng.integers(len(seq))]


def make_transcript(rng, mastery: float, objective_text: str, n_exchanges: int) -> pd.DataFrame:
    """Transcript whose surface features track ``mastery`` monotonically."""
    p_praise = 1.0 / (1.0 + math.exp(-(mastery * 1.6)))
    rows, t = [], 0
    key_words = [w for w in objective_text.rstrip(".").split() if len(w) > 4][:4]

    def push(role, content):
        nonlocal t
        rows.append({"utterance_id": len(rows), "role": role, "content": content,
                     "timestamp": f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"})
        t += 8

    push("tutor", f"Hello, today we are working on {objective_text.lower()}")
    push("student", "Okay.")
    for i in range(n_exchanges):
        n = int(rng.integers(2, 99))
        a = int(rng.integers(2, 12))
        b = max(1, n // a)
        # Sprinkle objective vocabulary so the relevance window has something
        # to lock onto, as it would in a real multi-objective session.
        topic = _rng_choice(rng, key_words) if key_words and rng.random() < 0.3 else ""
        push("tutor", f"Here is a question about {topic}. What is {a} times {b}?".replace(" about . ", " "))
        got_it = rng.random() < p_praise
        if got_it and rng.random() < 0.4 + 0.4 * p_praise:
            push("student", _rng_choice(rng, EXPLAIN_S).format(a=a, b=b, n=n))
        elif got_it:
            push("student", _rng_choice(rng, CONFIDENT_S).format(n=n))
        else:
            push("student", _rng_choice(rng, HEDGED_S).format(n=n))
        push("tutor", _rng_choice(rng, PRAISE_T) if got_it else _rng_choice(rng, CORRECT_T))
        if rng.random() < 0.25:
            push("tutor", _rng_choice(rng, PRESS_T))
            push("student", _rng_choice(rng, EXPLAIN_S).format(a=a, b=b, n=n)
                 if got_it else "Um, I am not sure.")
        if rng.random() < 0.05:
            push("background", "[unclear]")
    push("tutor", "Good work today, see you next time.")
    push("student", "Bye.")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sessions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out
    (out / "train_transcripts").mkdir(parents=True, exist_ok=True)

    # Objective difficulty: a real, learnable per-objective effect.
    difficulty = {oid: float(rng.normal(0, 0.8)) for oid, _ in OBJECTIVES}

    feat_rows, label_rows = [], []
    for s in range(args.n_sessions):
        session_id = f"s{s:05d}"
        # Latent quality shared by every response in this session. This is the
        # leak a naive KFold exploits.
        student_ability = float(rng.normal(0, 1.0))
        tutor_quality = float(rng.normal(0, 0.6))
        n_obj = int(rng.integers(1, 5))
        chosen = rng.choice(len(OBJECTIVES), size=n_obj, replace=False)

        session_mastery = student_ability + tutor_quality
        transcript = make_transcript(
            rng, session_mastery, OBJECTIVES[chosen[0]][1], int(rng.integers(12, 40))
        )
        transcript.insert(0, "session_id", session_id)
        transcript.to_csv(out / "train_transcripts" / f"{session_id}.csv", index=False)

        for k in chosen:
            oid, otext = OBJECTIVES[k]
            latent = session_mastery + difficulty[oid] + float(rng.normal(0, 0.7))
            p = 1.0 / (1.0 + math.exp(-latent))
            rid = f"r{len(feat_rows):06d}"
            feat_rows.append(
                {"response_id": rid, "session_id": session_id,
                 "learning_objective_id": oid, "learning_objective": otext}
            )
            label_rows.append({"response_id": rid, "correct": int(rng.random() < p)})

    pd.DataFrame(feat_rows).to_csv(out / "train_features.csv", index=False)
    pd.DataFrame(label_rows).to_csv(out / "train_labels.csv", index=False)
    print(f"Wrote {len(feat_rows):,} responses over {args.n_sessions:,} sessions to {out}")
    print(f"Base rate: {pd.DataFrame(label_rows)['correct'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
