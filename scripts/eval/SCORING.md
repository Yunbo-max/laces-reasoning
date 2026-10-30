# Scoring and evaluation protocols (LACES / LaLaLanD)

Every number in the paper comes from one of the protocols below. Scripts live in
`research/laces-reasoning/scripts/eval/` unless stated otherwise; results land in
`research/laces-reasoning/results/benchmarks/<variant>/…`.

## 1. Generate-then-match (the main protocol)

`scripts/eval/score_gen_match.py` — scores a `*.jsonl` whose records carry
`prompt / generate / ground_truth / choices` (the schema all our runners write).

Rule set, applied identically to every model:

1. truncate the generation at the first newline (mirrors Cola-DLM's `process_line`);
2. try to match a **choice** against the text: exact (normalised) → prefix in either
   direction → token-F1 argmax with a 0.5 floor;
3. if no choice matches, fall back to parsing an option **letter**, with the answer
   label (`A:`, `(A)`, `Answer:`) stripped first, case-insensitively;
4. resolve the ground-truth choice by the same matching, then score 1 if the picked
   choice is the gold one;
5. `squad*` tasks are free-form: token-F1 and EM, no choice matching;
6. `lambada*` tasks: the metric is the **last word** — the generation's opening words
   must contain the gold word (`acc_*_last`), with per-token and exact-suffix variants
   reported separately for diagnosis.

```bash
python scripts/eval/score_gen_match.py results/benchmarks/<variant>/gen results/benchmarks/baseline-<name>
```

## 2. mc-pack multiple choice (NLL ranking)

`scripts/eval/eval_prefix_suffix_cfg_multichoice.py` (single-z) and
`scripts/eval/eval_mcpack_trajectory_cfg.py` (chunked/trajectory models). Reports
`acc_raw_suffix` (no injection) and `acc_cfg{scale}` (injected), scored by ranking the
candidate continuations' NLL and taking the argmin.

**Use the trajectory variant for any checkpoint with `trajectory_mode=true`.** Running a
chunked checkpoint through the single-z script feeds a one-shot plan to a model trained
on per-chunk plans, which silently depresses the injected numbers (measured: 0.33 vs
0.49–0.51 average).

## 3. LAMBADA last-word

`scripts/eval/eval_prefix_suffix_cfg_lambada.py` / `eval_lambada_trajectory_cfg.py`.
Three metrics for both raw and injected: `acc_*_suffix` (whole-suffix exact — kept only
as a diagnostic; it is ~0 by construction), `acc_*_last` (standard LAMBADA), `acc_*_token`.

## 4. Perplexity, and bits-per-byte across tokenizers

`scripts/eval/eval_lm_ppl.py`.

- `--impl relay` (our models): prefix/suffix split; `--mode raw` scores with no plan,
  `--mode prefix_cfg` writes a sampled plan before scoring the suffix. Same checkpoint,
  so the table isolates what writing memory does.
- `--impl hf` (baselines): the stored shards are tokenised with the **RWKV world
  tokenizer**, so for a model with a different vocabulary the script decodes back to text
  and re-encodes with that model's tokenizer, then reports **bits-per-byte** as well as
  PPL. PPL is not comparable across tokenizers; BPB is.

## 5. Efficiency and memory

`scripts/eval/eval_efficiency.py` — load time, prefill over 512 tokens, greedy decode
latency per token at batch 1 and batch 8 (batch 8 = the one-device parallel setting),
plan sampling + writing time, peak allocated memory. Input is built from a **real
tokenizer** (random ids fall outside some vocabularies and trigger CUDA asserts).

## 6. Task-data construction (must be reproducible)

- `scripts/preprocess/build_gen_task_data_from_npz.py` — rebuilds generative MC task
  data from the tokenised shards. **Slices each row with its own `choice_start[j]`**;
  using `choice_start[0]` for every row silently corrupts the choices (symptom: strong
  models score near zero while weak ones look fine). Strips the RWKV end-of-text marker,
  drops duplicate/empty choices and rows with no shared prompt.
- `scripts/preprocess/build_winogrande_from_hf.py` — winogrande shards are cloze pairs
  with `choice_start = 0`, so no prompt can be recovered from them; the generative form
  is rebuilt from `allenai/winogrande`.

## 7. Failure modes we have already paid for

| Symptom | Cause | Fix |
|---|---|---|
| injected score collapses to ~0.3 | chunked checkpoint scored by the single-z script | use the trajectory variants |
| every model scores near zero on a MC task | `choice_start[0]` used for all rows | per-row `choice_start[j]` |
| RACE far below its historical range | whole-suffix exact match on sentence choices | choice matching, not string equality |
| LAMBADA 0/200 in both modes | whole-suffix exact match | last-word metric |
| PPL = 1.0 on short corpora | skip rule dropped every sequence (0 tokens) | FIXED 2026-09-21: `--split_frac` splits short sequences proportionally; a 0-token corpus now reports `null`, never `exp(0/max(1,0))==1.0` |
| relay PPL in the 1e5 range (`prefix_cfg`) | FIXED 2026-09-21: `score_relay` scored `out.logits` against `ids[:, split:]` with no shift, so the logit *at* token t was scored against token t instead of t+1. 12.0 nats/token > log(65536)=11.09, i.e. worse than uniform -> PPL 164023 | build the prefix cache from `ids[:, :split-1]`, feed `ids[:, split-1:]`, score `logits[:, :-1]` vs `ids[:, split:]` -- same convention as training's `_singlez_coadapt_ce_from_z` |
| relay `bits_per_byte` ~5x the HF rows | FIXED 2026-09-21: relay passed `nbytes = ntok`, so the column was bits-per-TOKEN (it equalled `log2(PPL)` exactly) | decode the scored span and count real UTF-8 bytes |
| PPL reported with `--cfg_scale 3.0` | CFG extrapolates epsilon, so the guided plan is not a draw from the conditional posterior and the induced token distribution is not a normalised likelihood | default is now `--cfg_scale 1.0`; anything else hard-errors unless `--allow_cfg_ppl` |
| baseline PPL in the thousands | RWKV-tokenised shards fed to another vocabulary | decode→re-encode, report BPB |
| `steps` rejected by `generate` | transformers validates unused kwargs (raises ValueError, not TypeError) | pass only keywords in `inspect.signature(model.generate)` |
| remote checkpoint will not load | `tie_weights` signature drift; missing `all_tied_weights_keys`; `KeyError: 'default'` from `ROPE_INIT_FUNCTIONS` (the key was removed in transformers 5.x); `AttributeError: config has no attribute use_cache` | filter kwargs by signature; build the config from its own class first; fall back to the `auto_map` class; restore `ROPE_INIT_FUNCTIONS['default']`; default `config.use_cache` |
| efficiency job dies with a CUDA assert | random token ids outside the vocabulary | real tokenizer input |
| raw baseline fails instantly | `--ckpt_dir` is required even in `--method raw` | pass the checkpoint |
| log unavailable from QZ | `GetJobLog` returns InternalError | every job writes its own logs to disk |

## 8. Where results live

```
results/benchmarks/<variant>/gen/<task>.jsonl     generate-then-match outputs (ours, raw)
results/benchmarks/baseline-<name>/<task>.jsonl   same, external baselines
results/benchmarks/<variant>/mcpack/<task>.json   mc-pack NLL ranking (+ trajectory variants)
results/benchmarks/<variant>/cola8/<task>.jsonl   Cola-DLM protocol generations
results/benchmarks/ppl/<variant>_<mode>.json      perplexity / bits-per-byte
results/benchmarks/efficiency/<variant>.json      latency, memory, plan time
```

Job logs: `/inspire/hdd/project/multimodal-diffusion-model/zhangjiaquan-253108540222/qz_eval_logs/`.

## 9. Long-context suites (`run_open_context_bench.py`)

One runner for all five: `--suite {longbench,babilong,ruler,infinitebench,memeval}`, data read
from the shared HF cache (compute nodes are offline), model from `--model_dir` (HF baseline)
or `--ckpt_dir` (relay).

Metrics, and how much of each is actually official:

| suite | metric | provenance |
|---|---|---|
| `babilong` | label-aware `compare_answers` | official, for qa1/qa2/qa3/qa6/qa9 only -- other tasks carry no label set and raise |
| `ruler` | substring, max over refs except `vt` (mean) | official-style; see the caveat below |
| `longbench` | QA F1 / ROUGE-L F1 (`rouge==1.0.1`) / `Paragraph N` match | official LongBench `metrics.py` |
| `infinitebench` | per-task rules | **approximation** -- the official `eval_utils.py` was not reachable offline |
| `memeval` | containment, else token-F1, per question | **approximation** -- official is a gpt-4o yes/no judge |

Two things to keep in mind before any of these numbers reach the paper:

- The `memeval` and `infinitebench` rows are *approximations and are named as such in the
  code*. The official LongMemEval score is LLM-judged; the substitute cannot score abstention
  at all, so the 30 `_abs` questions per split are a systematic zero.
- RULER's `max`-over-refs rule is generous on the multi-reference tasks: `cwe` carries 10
  reference words, `fwe` 3, `niah_multivalue`/`niah_multiquery` 4. Any generation containing
  one of them scores 1.0. Whether the official split is max/mean per task was not verifiable
  offline; the current rule is asserted by `tests/test_open_context_bench.py`.
