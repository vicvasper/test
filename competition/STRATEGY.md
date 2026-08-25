# Trace the Ace — the path to first place

## The single most important thing about this competition

**The leaderboard does not decide the winner.** From the competition's own
description of how prizes work: if you are in the **top 15** when model
submissions close, you become *eligible* for a prize, and prizes are then
awarded on **a combination of leaderboard performance and write-up quality** —
"winners will be evaluated not only quantitatively, who makes the most accurate
predictions, but also on the quality of their insights into the task."

That changes the whole strategy:

- Top 15 is a **gate**, not the goal. Rank 9 with an excellent write-up beats
  rank 2 with a thin one.
- Going from rank 240 to rank 12 is a **workflow problem**. Going from rank 12
  to first is a **writing and insight problem**.
- Time spent squeezing the fourth decimal of log loss after you are safely
  inside the top 15 is worth less than the same hours spent on the write-up.

So the plan is: clear the gate decisively, then spend the remaining time on
insight.

## Where rank 240 most likely came from

You said it was "an incorrect workflow", and this competition has an unusually
punishing one — it is a **code-execution** competition. You do not upload
predictions; you upload a `submission.zip` containing `main.py`, and the
platform runs it inside a Docker container with **no internet**.

`WORKFLOW.md` documents six traps, each reproduced locally. Two produce exactly
the "it ran but I placed terribly" signature:

- **The unmodified example scores 0.69315.** `examples/minimal/main.py` writes
  `submission_format.csv` straight back out, and that file is all `0.5`. It
  exits 0, logs success, and lands near the bottom. Nothing warns you.
- **Row misalignment** — writing your own frame's order instead of the
  submission format's — scores like noise while looking completely healthy.

And one that loses the submission outright:

- **A nested zip passes the platform's check and then crashes.** The entrypoint
  greps for the *substring* `main.py`, which `submission_src/main.py` satisfies;
  `python main.py` then fails because the entrypoint has `cd`-ed to
  `/code_execution`. The local `just check-submission` uses an exact-line match
  and catches it — which is why skipping that step is what costs you.

`training/preflight.py` in this repo turns all of these into an automated
gate. Run it before every upload.

## Two-day plan (submissions close **August 27**)

The deadline is the binding constraint. This ordering front-loads everything
that guarantees a valid, non-embarrassing score, so that every later hour is
upside rather than risk.

### Day 1 morning — establish a floor you cannot fall below

1. Download the competition data. Confirm what `train_features.csv` carries
   beyond the test columns — specifically whether there is a **student id**,
   a tutor id, or a timestamp. Grouping depends on it (see below).
2. Run `python training/train.py --data-dir <data>`. This prints the baselines,
   the grouped-CV score, and the leakage gap.
3. Package and submit **the per-objective prior alone**. It takes minutes, it
   cannot crash, and it beats a constant. Now you have a floor, and every
   subsequent submission is measured against a real number instead of against
   the risk of having nothing.

### Day 1 afternoon — the model that clears the gate

4. Full feature model (`submission_src/src/features.py`, 76 features) with
   LightGBM and grouped CV. On the synthetic benchmark this is **0.110 better
   than the constant baseline and 0.092 better than the objective prior**.
5. Check the leakage gap the harness prints. If naive CV is optimistic by more
   than ~0.01, that gap is the mechanism by which local progress fails to
   transfer to the leaderboard.
6. Smoke test, then submit. Compare leaderboard to grouped CV. **If they
   disagree by much more than fold-to-fold variance, stop modelling and find
   out why** — that discrepancy is worth more than any feature.

### Day 2 morning — the upgrades that actually move log loss

In descending order of expected value per hour:

7. **Semantic relevance windows.** The current window is lexical
   (`_relevance` in `features.py`), and on the demo data one of five sessions
   scores `rel_max = 0` — the objective's words simply never appear literally.
   Swapping in `bge-large-en-v1.5` or `all-MiniLM-L6-v2` embeddings (both
   already staged in the runtime) to locate the objective-relevant span should
   help most on exactly the rows the lexical version fails on.
8. **`saroyehun/Talkmove-bert`.** This is already on the runtime's model list,
   and it is a classifier for *accountable-talk moves* in classroom dialogue —
   a purpose-built encoder for this exact data type. Per-utterance talk-move
   distributions, aggregated over the session and over the relevant window, are
   both a strong feature block and the backbone of a genuinely interesting
   write-up. Highest combined score-and-insight value on the list.
9. **A fine-tuned `ModernBERT-large`** over the relevant transcript window,
   blended with the GBDT. Text models and feature models make different errors,
   so the blend usually beats both. Weight the blend by optimising log loss on
   grouped OOF predictions, never on the leaderboard.
10. **Isotonic calibration** on grouped OOF predictions, if the calibration
    table `train.py` prints shows drift above ~0.05. Log loss is a proper
    scoring rule: calibration is free score.

### Day 2 afternoon — the part that decides the actual placing

11. Freeze the model. Submit the best validated blend with time to spare.
12. **Write the write-up.** Details below.

Keep a hard reserve: make your final scored submission **well before** the
close, not against it. A queue delay at the deadline is an unforced loss.

## What actually predicts the follow-up answer

The features are organised around claims that can be defended in a write-up,
not around anonymous embedding dimensions — because insight quality is scored.

1. **Tutor evaluative feedback is the correctness proxy.** Praise and
   correction counts directed at student turns approximate whether the student
   was getting items right *during* the session. Strongest single block.
2. **Recency beats averages.** The follow-up comes after the session, so the
   final third matters more than the session mean. Hence `late_praise_rate`,
   `praise_rate_delta`, `hedge_rate_delta` — the trajectory, not the level.
3. **Self-explanation.** Causal connectives in student turns ("because", "so")
   distinguish understanding from a lucky correct answer. Chi's self-explanation
   effect, and directly measurable here.
4. **Press-for-reasoning, and the answer to it.** A tutor asking "how did you
   work that out?" is an accountable-talk move; what matters is what the student
   produces next. `press_reply_words_mean` and `press_reply_explained_rate`
   capture the response, not just the prompt.
5. **Objective difficulty dominates session quality.** The smoothed
   per-objective base rate is a large share of achievable signal and must be
   fitted out of fold.
6. **The objective-relevant window is what separates two rows of one session.**
   A session probed on three objectives gets three rows; session-global features
   give all three identical values. `_window()` restricts the feature computation
   to the span of the transcript actually about that objective. Without it the
   model can only ever learn a session-level average.
7. **Transcription quality is a confounder, not a signal.** `[unclear]` rate
   correlates with both the features and the outcome. It is in the model
   explicitly so that audio noise is not attributed to pedagogy — and that
   distinction is itself write-up material.

## The validation discipline that separates top-15 from rank 240

`training/train.py` enforces three things:

- **`GroupKFold` on `session_id`.** Several responses share a session. A plain
  `KFold` splits them across folds, session-level signal leaks, local CV
  improves, the leaderboard does not, and you tune in the wrong direction for
  days. The script runs *both* splits and prints the gap, so the leak is a
  number you can see rather than a suspicion.
- **Out-of-fold target encoding.** The objective prior is refitted inside every
  fold. Fitting it once on all of training data leaks the label into its own
  feature and is the most common source of an over-optimistic CV.
- **Baselines first.** Constant 0.5, global base rate, and per-objective prior
  are printed every run. A model that does not beat all three is not a model.

**If the data has a student id, group on student as well** (or at minimum check
whether grouping by student widens the gap). The same student across sessions
leaks harder than the same session, because student ability is the dominant
latent variable. This is the first thing to check when the real data lands.

## The write-up — this is what wins

Prizes weight insight quality alongside score, and there is an additional award
for teams that publish their findings. The write-up is not documentation of your
model; it is an argument about tutoring. Structure it as findings, evidenced:

- **Lead with what predicts learning, not with your architecture.** "Late-session
  student explanation length predicts the follow-up better than total session
  length" is a finding. "We used LightGBM with 3000 rounds" is a hyperparameter.
- **Quantify each claim.** Ablate the feature blocks and report grouped-CV log
  loss with each removed. That table *is* the insight section.
- **Report what did not work.** Negative results are unusually credible and
  almost nobody writes them. Session duration, turn count, and raw praise count
  are all plausible-sounding features; if they add nothing over the trajectory
  features, say so and show it.
- **Name the confounders honestly.** Transcription quality, objective difficulty,
  and session length are entangled with outcome. Showing you measured the
  confounder rather than letting it inflate a pedagogical claim is exactly the
  judgment the organisers are selecting for.
- **Be explicit about what the model cannot support.** This data cannot identify
  what *causes* learning — it is observational, the outcome is a single
  follow-up item, and tutors are not randomly assigned. A write-up that states
  its own limits reads as more trustworthy than one that overclaims, and the
  audience here is researchers who will notice.
- **Make it reproducible.** Reproducibility is standard for DrivenData prize
  verification: the winning submission has to be re-runnable from your code.
  Pin your seeds and keep the training script runnable end to end.

## Rules compliance

The rules that matter, and none of them are in tension with anything above:

- **One account per participant.** No multi-accounting, and therefore no
  submitting from a second account to get extra attempts.
- **No private sharing of code or data outside your team.** Public forum posts
  are fine; a private DM of a solution is not.
- **Submission limits are per-competition and enforced.** Any attempt to
  circumvent them is disqualifying. Budget your remaining submissions
  deliberately — this is precisely why the smoke test exists.
- **The runtime is fixed and offline.** New packages or Hugging Face models go
  through a PR to the runtime repo, CI, and admin approval. With the deadline
  this close, treat the current package and model list as fixed.
- **Prize claims require the write-up**, and reproducibility of the winning
  submission.

There is no shortcut worth taking here. The gap between rank 240 and top 15 in
this competition is not something a rule-bend would buy you — it is a valid
submission plus honest cross-validation, both of which are fully within reach
in the time remaining.

## Confirm these on the platform before you rely on them

I could not reach `platform.k12-ai-infrastructure.org` — this environment's
egress policy returns 403 for that host — so the competition pages themselves
were not readable. Everything above about the **runtime** is first-hand from the
runtime repository. The following came from search results and should be
verified on the competition pages before you plan around them:

- The **August 27** close date, and the exact closing time and timezone.
- The **top-15** eligibility rule and the prize/write-up weighting.
- The **daily submission limit** and how many you have left.
- The **execution time limit** and hardware for a scored run (this determines
  whether an LLM-based approach fits at all — check it before starting step 8).
- The **external data policy**, if you intend to use anything beyond the
  provided data.
- Whether `train_features.csv` includes a **student id** — it changes the
  grouping, and the grouping is the whole ballgame.
