# E-X8 — Speculative self-drafting acceptance measurement

Status: **harness built and smoke-tested; no measurement run yet.**
Date: 2026-07-17.  Branch: `experiments/e-x1-phase-baseline`.

## Step 0 — `REF` / `teacher_*` semantics (answered)

- `REF` ([qwen36b.c:5226](../../../../src/qwen36b.c#L5226)) is only the
  legacy tiny-fixture oracle JSON path (`c/ref_qwen36.json`); it is not a
  scoring harness and is irrelevant to E-X8.
- `--teacher-corpus/--teacher-output` (`run_teacher_capture`,
  [qwen36b.c:4237](../../../../src/qwen36b.c#L4237)) **is** a teacher-forcing
  scorer: for every position of every corpus sequence it records the target
  token, its logit, the logsumexp, and the top-5 argmax ids to a
  SHA-256-sealed little-endian binary stream (`QWTFM001`).  That is exactly
  the verify-side measurement E-X8 needs, with one gap: it accepted only
  *text* prompts (re-templated and re-tokenized), and decode→encode does not
  round-trip token ids in general.
- The MoE policy is process-global (`g_moe_policy`), so `MOE_K=1` applies to
  chat, serve, and teacher modes alike: the draft and the full model are the
  same binary under different env.
- `ROUTE_TRACE` v2 records `position` and `token_id` for every routed input
  token ([qwen36b.c:961](../../../../src/qwen36b.c#L961)), so a draft
  generation run under `ROUTE_TRACE` yields its exact token-id sequence with
  no engine change.  Caveat: a *resumed session* does not re-route its
  restored context, so draft runs for this experiment must send the full
  prompt (the extractor rejects non-contiguous positions).

## Harness added (this commit)

- [src/qwen36b.c](../../../../src/qwen36b.c): a corpus item may now be
  `{"tokens": [ids...]}` instead of `{"prompt": ...}` — forced verbatim, no
  chat template, no re-tokenization, ids validated against the vocab.  The
  text-prompt path is unchanged.
- [tools/spec_accept.py](../../../../tools/spec_accept.py):
  `extract-trace` (route-v2 JSONL → tokens corpus) and `analyze` (QWTFM001
  stream → per-position agreement α, p(draft token), leading-run
  distributions, sequential window simulation for W ∈ {4,6,8}, and the
  modeled speedup curve when `--t-full/--t-draft/--t-verify` are supplied).
- [tests/test_spec_accept.py](../../../../tests/test_spec_accept.py): 7
  self-contained tests over byte-exact synthetic streams and traces; wired
  into `make test` (all green, Python 3.14, 2026-07-17).

## Real-model smoke runs (validation only — not measurements)

Machine state: no model process running; swap 1,100.94 MB before and after
(zero growth); `pmset -g therm` no warnings.  Portable single-thread build.

1. Token corpus (10 arbitrary valid ids): engine forced 9 positions in
   3.237 s; `spec_accept.py analyze` parsed the stream, verified its SHA-256,
   and reported α = 2/9 with a coherent leading-run histogram.  α is
   meaningless here by design (the ids were not model-generated); the run
   validates only the pipeline.
2. Text-prompt regression corpus (`"Hi"`, no_thinking): 12 positions in
   3.610 s — the pre-existing path is intact.

`make` and `make omp` both build clean; full `make test` passes.

## Measurement plan (next; needs the machine idle per the standing rules)

All runs greedy, `seed=1729`, thinking off, 4T `make omp` build, warm
protocol per the common protocol; full safety telemetry each leg.
Baseline inputs already measured by E-X1/E-X4:
`t_full = 131.6 ms/token` (4T W-DECODE median), expert unions for W=4/6/8 =
21.7 / 28.6 / 34.7 experts·layer⁻¹ (E-X4 Phase A, non-baseline host).

1. **Draft speed** — W-DECODE with `MOE_K=1` (serve path, warm, ×3 + warmup):
   decode tok/s, `[moe-policy]`, `[stats]`, `[ecache]`.  This bounds
   everything: max speedup ≤ t_full/t_draft even at α=1.
2. **Draft generations for acceptance** — 5 diverse ~512-token continuations
   (chat, code, summary; suite-adjacent), each sent as a **full prompt** (no
   resumed session) with `MOE_K=1 ROUTE_TRACE=<file>`; archive traces.
3. **Verify capture** — `spec_accept.py extract-trace` → tokens corpus →
   `--teacher-corpus` under default (full) MoE, `--teacher-calibration 1`.
   Teacher forcing is S=1 decode-shaped (~130 ms/token warm ⇒ ≈3–4 min per
   1.5k-token sequence at 4T).  E-X1's thermal lesson applies: watch the
   live powermetrics trace and stop on sustained non-Nominal pressure;
   sequences are independent, so run them one at a time with cooldowns.
4. **Verify cost** — batched forward of W tokens approximated by a W-token
   prefill continuation on a warmed engine (state the approximation).
5. **Model the curve** — `spec_accept.py analyze --draft-start ... --windows
   4,6,8 --t-full --t-draft --t-verify` per the card; report the whole
   curve.  Go/no-go: modeled end-to-end speedup ≥ 1.4×.

Also owed by the card: one paragraph on whether `REFINE_*` base planes could
be a cheaper draft than `MOE_K=1`.

Not run: everything in this section.  No performance claim is made by this
report.
