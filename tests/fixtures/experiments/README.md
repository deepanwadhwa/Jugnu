# E-X1 deterministic workloads and quality suite

These files are deliberately plain UTF-8 text.  They are versioned inputs, not
claimed token counts: the reference tokenizer is the authority.  Before a
baseline run, record `wc -w` and the encoded-token count in the evidence
report.  If a tokenizer/model update moves a workload materially away from its
target shape, adjust the fixture in a reviewable commit rather than silently
changing the command.

All experiment invocations use `--greedy --no-thinking --seed 1729` and a
freshly built `qwen36b`.  The runner must archive the command plus every
`[stats]`, `[phase]`, `[ecache]`, and `[seqio]` line.

## Workloads

- `workloads/w_decode_context.txt` seeds a saved session.  Resume that session
  and generate 256 tokens for W-DECODE; use the tokenizer count to tune the
  seed turn to approximately 1,000 saved-context tokens.
- `workloads/w_prefill_document.txt` is the W-PREFILL source document.  Ask
  for a concise summary and generate 32 tokens.
- W-SESSION is the same saved-session procedure after extending the context to
  at least 4,096 tokens, then generating 128 tokens.
- W-SUSTAIN repeats the W-DECODE command for ten minutes under the thermal
  protocol in `docs/TASKS_EXPERIMENTS.md`; it is never an unattended loop.

## Quality suite

Run every file in `prompts/` in lexical order.  `quality_source.md` is the
committed source for the two summary prompts.  The long-document QA prompt
uses the repository's Jobs corpus fixture when that fixture is present; until
then it is explicitly not run rather than substituted with an untracked file.

The suite is a compatibility baseline: archive exact outputs before comparing
any numerics- or policy-changing experiment.
