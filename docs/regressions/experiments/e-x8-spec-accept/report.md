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

## Addendum 2026-07-18 — checkpoint has a native MTP draft head; plan deltas from colibrì evidence

The card gained an addendum on `main` (commit `5bd41bc`); this section
records the verification evidence behind it and the resulting plan deltas.
No measurement was run.

### Verified: the active checkpoint ships an MTP layer the engine loads but never runs

Command and output (2026-07-18, active snapshot):

```
$ python3 -c "
import json
cfg=json.load(open('$HOME/.samosa/current/model/config.json'))
tc=cfg.get('text_config',cfg)
for k in ['num_hidden_layers','num_experts','num_experts_per_tok',
          'moe_intermediate_size','hidden_size','mtp_num_hidden_layers',
          'shared_expert_intermediate_size']:
    print(k, tc.get(k))
"
num_hidden_layers 40
num_experts 256
num_experts_per_tok 8
moe_intermediate_size 512
hidden_size 2048
mtp_num_hidden_layers 1
shared_expert_intermediate_size 512
```

Engine cross-references (code read, same date): expert offset/size tables
are sized for `n_layers + mtp_layers`
([qwen36b.c:1284](../../../../src/qwen36b.c#L1284),
[:2345](../../../../src/qwen36b.c#L2345)); the refine store requires MTP
experts to be `passthrough-int8` ([:1679](../../../../src/qwen36b.c#L1679));
`expert_views` maps layer index `n_layers` (i.e. 40) as int8
([:1845-1850](../../../../src/qwen36b.c#L1845-L1850)); the forward loop
runs `i < c->n_layers` ([:3092](../../../../src/qwen36b.c#L3092)), so layer
40 is parsed and loadable but never executed.  Grepping `mtp` over
`src/qwen36b.c` shows only config/geometry/loading sites — no forward-pass
call.  **Conclusion: the model carries its own draft head, int8, unwired.**

### Plan deltas (from the card addendum; external colibrì numbers are GLM-5.2 on other hardware, directional only)

1. Steps 2–3 gain a `MOE_K=2` leg beside `MOE_K=1` (draft-quality
   sensitivity: colibrì saw MTP acceptance collapse 39–59% → 0–4% between
   int8 and int4 heads — acceptance can sit on a cliff).
2. Step 4 additionally diffs a batched forward's per-position argmax
   against the S=1 teacher capture on one already-captured sequence —
   colibrì #100 measured that shape-dependent integer kernels alone can
   fork greedy output; this is the free early warning for the
   "greedy acceptance keeps token identity" assumption.
3. Step 6's model is reported warm and cold separately (colibrì measured
   cold-cache expert-loads/token inflating ~660 → ~1100 under
   speculation); draft legs archive their `[ecache]`/`[stats]` lines.
4. If go: the follow-up implementation card's leading design candidate is
   wiring the native MTP head (donor design:
   [colibri/c/glm.c](../../../../colibri/c/glm.c) — MTP drafting,
   batch-union verify, rejection sampling), with `MOE_K` drafts as the
   measured lower bound it must beat.

Runner's note: the owner's live privileged capture (started 2026-07-18) is
`sudo /usr/bin/powermetrics --samplers cpu_power,gpu_power,thermal -i 1000
-o /tmp/samosa-e-x10-m0-powermetrics.log`; `/tmp` does not survive reboot —
copy each leg's slice beside this report immediately after the run.
