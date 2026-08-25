# The submission workflow, and the ways it silently costs you 200 places

Everything in this file is derived by reading the official runtime repository
(`drivendataorg/tutoring-outcomes-runtime`, commit `ab8055d`) directly —
`runtime/entrypoint.sh`, `justfile`, `runtime/pyproject.toml` — not from
memory. Each trap below was reproduced locally before being written down.

## What the platform actually does with your upload

From `runtime/entrypoint.sh`, in order:

1. `zip -sf submission/submission.zip` and `grep -q main.py` on the result.
2. `unzip submission/submission.zip -d ./` into `/code_execution/`.
3. `cd /code_execution` (done at the top), then **`python main.py`**.
4. If `submission.csv` exists in that same directory, copy it out. If not, log
   an error and exit non-zero.

The whole thing runs under `set -euxo pipefail` with **no internet access**.

Four consequences, and each is a real failure mode:

- `main.py` must be at the **root of the zip**.
- `submission.csv` must land in the **current directory**, not in `output/`.
- **A non-zero exit discards the run entirely.** No file is copied out.
- Nothing may download at inference time.

---

## Trap 1 — the nested zip (reproduced, and it is nastier than it looks)

The classic mistake is `zip -r submission.zip submission_src` (zipping the
folder) instead of packing its *contents*.

Here is the part that makes this expensive. The platform's check is a
**substring** grep:

```bash
submission_files=$(zip -sf ./submission/submission.zip)
if ! grep -q main.py <<<$submission_files; then ... fi
```

`submission_src/main.py` **contains** the substring `main.py`, so this check
**passes**. The archive is accepted. Then `python main.py` runs at
`/code_execution/`, where no `main.py` exists, and the job dies.

Verified locally:

| archive | contains | platform entrypoint grep | `just check-submission` | `python main.py` |
| --- | --- | --- | --- | --- |
| `good.zip` | `main.py` | ACCEPTS | PASSES | runs |
| `bad.zip` | `submission_src/main.py` | **ACCEPTS** | **FAILS** | **crashes** |

The local recipe uses `grep -F -x -q -- "main.py"` — an *exact line* match — and
does catch it. The platform's does not. So the local check is not a formality
duplicating a server-side guard; it is the **only** thing standing between you
and a dead submission.

Correct packing — note the `cd`, which is the entire point:

```sh
cd submission_src && uvx rpzip -r ../submission/submission.zip ./*
```

or just `just pack-submission`, which does exactly that.

## Trap 2 — a submission that runs perfectly and scores 0.69315

`examples/minimal/main.py` ends with:

```python
predictions = submission_format.copy()
predictions.to_csv(SUBMISSION_PATH, index=False)
```

`submission_format.csv` is all `0.5`. So the example **is a fully valid
submission that predicts 0.5 for every row**. It exits 0, produces a
well-formed file, logs success, and scores exactly `ln(2) = 0.69315`.

There is no error anywhere to tell you something is wrong. If a leaderboard
position looks like "ran fine, scored near the bottom", this is the first thing
to rule out — a constant submission is indistinguishable from a healthy one
right up until the score appears. `preflight.py` raises a WARN when every
predicted probability is identical, which is the only automated signal you get.

Even predicting the **global base rate** instead of 0.5 beats it, and a
per-objective base rate beats that. On the synthetic benchmark in this repo:

| submission | log loss |
| --- | --- |
| constant 0.5 (the unmodified example) | 0.69315 |
| global base rate | 0.69276 |
| per-objective smoothed base rate | 0.67545 |
| the model in `submission_src/` | **0.58359** |

## Trap 3 — row misalignment

The scorer joins on `submission_format.csv`. If you write your own frame's row
order instead of the format's, every prediction lands on the wrong response.
The file looks perfect: right columns, right length, sensible probabilities.
It scores like noise.

`main.py` here maps predictions back by `response_id` and reindexes onto
`submission_format` explicitly, and `preflight.py` asserts the id order matches.

## Trap 4 — probabilities at exactly 0 or 1

Log loss is unbounded as `p → 0` or `p → 1`. One confident, wrong row can cost
more than every other row combined. Predictions are clipped to
`[1e-4, 1 - 1e-4]` in `src/model.py`.

## Trap 5 — anything that touches the network

No internet at inference. That rules out `pip install`, any API call, and
`from_pretrained("Qwen/Qwen3-8B")` with a bare hub id — the last one looks like
ordinary code and is a download.

Hugging Face models must (a) already be listed in `runtime/huggingface_models.txt`
via a **merged PR to the runtime repo**, and (b) be loaded from the mounted path:

```python
model_path = "/code_execution/huggingface_models/sentence-transformers/all-MiniLM-L6-v2"
AutoModel.from_pretrained(model_path, local_files_only=True)
```

Getting a new model added needs a PR, CI, admin approval, and a republish —
that is a multi-day path, so with the deadline close, **plan only around models
already on the list**. Currently staged: `Qwen3-14B-AWQ`, `Qwen3-8B`,
`Qwen2.5-14B-Instruct`, `Qwen3-Embedding-4B`, `Mistral-Small-24B-Instruct-2501`,
`ModernBERT-large`, `bge-large-en-v1.5`, two `gemma-4` AWQ builds, `phi-4`,
`all-MiniLM-L6-v2`, and `saroyehun/Talkmove-bert`.

## Trap 6 — version skew between your machine and the runtime

You train locally and unpickle inside the container. Pickled `sklearn` /
`lightgbm` estimator objects encode class layouts that shift between library
versions, so a mismatch raises at `pickle.load` — inside the container, after
the queue, unfixable without a new submission.

This is not hypothetical for this competition: the runtime `CHANGELOG.md`
records `pandas` being downgraded **3.0.3 → 2.3.3** on 2026-07-22 (a `gradio`
pin pulled in by `ms-swift`), plus `pillow` and `aiofiles` downgrades. That
entry also warns that copy-on-write is no longer mandatory in `pandas` 2.3.3,
so code mutating DataFrame slices can behave differently there than locally.

Mitigations, in order of preference:

1. Train **inside the runtime image** (`just interact-container`).
2. Export models in a version-stable text format. `src/model.py` provides
   `PortableLGBM`, which stores LightGBM's own text dump instead of a pickled
   object, so it survives a version bump.
3. At minimum, run `just test-submission` against the real image before upload.

---

## Pre-upload checklist

```sh
# 1. Package correctly (packs the CONTENTS of submission_src/)
just pack-submission

# 2. Exact-match root check — the one the platform does NOT do for you
just check-submission

# 3. Fast local contract check: static scan + real execution + output validation
python training/preflight.py --zip submission/submission.zip --data-dir data-demo

# 4. Authoritative check — the actual competition image
just pull
just test-submission

# 5. Spend a smoke test before a scored submission.
```

`preflight.py` is a fast pre-check that needs no Docker and catches the traps
above. It is **not** a substitute for step 4: it runs in your local Python, so
it cannot detect a package that exists on your machine but not in the runtime,
or vice versa. Only the real image proves that.

Smoke tests run on a small slice of training data, are not prize-evaluated, and
exist precisely so correctness bugs cost you nothing. Use one before every
scored submission you care about.
