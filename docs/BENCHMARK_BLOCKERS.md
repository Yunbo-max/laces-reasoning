# Benchmark Blockers

This file records which benchmark requests were temporarily excluded from the
bundled Qizhi evaluation jobs and why.

## Submitted Runnable Subset

The current bundled jobs only include benchmarks whose entrypoints and known
data roots are present in the workspace:

- `gsm8k`
- `cola8`
- `mcpack`

`cola8` expands to the existing Cola-DLM generative task set currently backed by
local task JSONL files under:

- `/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM/generate_task_data`

Specifically used tasks:

- `mmlu`
- `obqa`
- `race`
- `squad`
- `story_cloze`

`mcpack` expands to locally prepared multiple-choice / exact-match evaluation
surfaces backed by files under `DiffRwkv/preprocessed_data`:

- `mmlu`
- `openbookqa`
- `race`
- `siqa`
- `story_cloze`
- `piqa`
- `hellaswag`
- `arc_easy`
- `arc_challenge`
- `winogrande`
- `sciq`
- `lambada`

## Verified Next Safe Group

The next locally validated group is:

- `paperpack`

It has passed dry-run command generation and is backed by existing local data:

- `mmlu_pro/test`
- `xsum/validation`
- `wmt14_de_en/validation`

`paperpack` is not yet included in the currently queued jobs; it is ready for the
next resubmission wave.

## Deferred Benchmarks

These were requested in the larger plan but were not included in the bundled
jobs because the current `laces-reasoning` workspace does not yet provide a
verified runnable path with local data/dependencies.

### Missing or mismatched evaluation entrypoints in `laces-reasoning`

- `mmlu_redux`
- `gpqa_diamond`
- `c_eval`
- `ifeval`
- `bfcl`
- `truthfulqa`
- `known_unknowns`
- `babilong`
- `glue`

### Why they were excluded

1. The configured entrypoint scripts did not exist in `laces-reasoning`, or the
   mapped script name was incorrect.
2. Some available scripts require external local dependencies not present in the
   current environment:
   - `lm_eval` is not installed for `run_lmeval_relay.py`
   - `bfcl_eval` is not installed for BFCL relay evaluation
   - LongBench / BABILong local staging directories are not present
3. Some datasets or local prepared bundles expected by the scripts are absent:
   - local BABILong JSON exports
   - LongBench config/data checkout
   - benchmark-specific local bundles for `mmlu_redux`, `gpqa_diamond`, `c_eval`

## Next Recovery Steps

To expand beyond the runnable subset, the next steps are:

1. Add/install the missing evaluation dependencies (`lm_eval`, `bfcl_eval`,
   LongBench checkout) in the execution image or job bootstrap.
2. Stage the missing benchmark data locally or point the scripts to the correct
   existing locations.
3. Replace placeholder benchmark mappings with validated, repo-local commands,
   then re-run dry-run verification before submit.
