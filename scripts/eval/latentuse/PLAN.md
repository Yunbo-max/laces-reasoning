# D5 -- GPU job plan for `tab:latent-use-pending`

Fills the five rows of `tab:latent-use-pending` in
`DiffRwkv/paper_ICLR27/lalaland/appendix.tex`. Verified positions in the live
tree on 2026-09-23: section heading `\subsubsection{Matched latent-use controls}`
at **line 711**, the arm definitions at **714**, `\begin{table}` at **716**,
`\label{tab:latent-use-pending}` at **719**, `\end{table}` at **731**, the
protocol paragraph (200 examples, paired CIs, first-token logits) at **733**.
The briefing said "table ~718, prose ~712-737"; the real span is 711-733 and 737
is `\label{app:historical}`, i.e. the next subsection. Close enough to have been
read off a live tree, but cite 719 for the label.

Everything in this file was prepared on CPU. Nothing here was run by the author
of this file; the three GPU job groups below were executed by separate agents and
their status is recorded in **Status** at the end.

The five arms, exactly as the appendix prose defines them:

| arm | condition | write |
|-----|-----------|-------|
| A0 | Same-loop backbone, no adapter | adapter NOT called; principal memory untouched |
| A1 | Same adapter, original principal memory | adapter called, then original `recurrent_state` restored bit-exactly |
| A2 | Training-mean latent | `R(mean_train_latent)` |
| A3 | Shuffled sampled latent | the sampled latent of a DIFFERENT prompt, within-task derangement |
| A4 | Correct prefix-conditioned latent | the sampled latent of the correct prompt |

One checkpoint for all five:
`outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000/model.pt`,
sha256 `55bc5c7bf8d0a242c0a007d57c7e4d8114ce8b80ea6331ba43865125d6b2a573`,
1492023287 bytes, 189 trainable tensors, step 50000. Pinned in `provenance.json`.

## Environment for every job

```bash
export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning:/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
```

Write that `PYTHONPATH` line exactly as shown: a bare `$PYTHONPATH` under `set -u`
fails silently. Never `pkill -f`. Redirect job output to a file rather than piping
through `grep` — piping a 1.4 GB load has been OOM-killed before (exit 137).

`RELAY_INJECT_ZERO_RNN_CACHE` is left at its default `0`. It does not matter for
this checkpoint: `global_denoiser_type` is ABSENT from the config, so the attribute
defaults to `"dit"`, `is_rnn_denoiser` is False, and `conv_state`/`ffn_state` are
zeroed on every write regardless. Recorded as
`injection_branch: "non-rnn(zero conv/ffn)"` in `provenance.json`.

## Deliverables already on disk (CPU, complete)

| file | what | status |
|------|------|--------|
| `provenance.py` -> `provenance.json` | checkpoint sha256/size/step, git HEAD+dirty, writer shapes, decoder geometry, `global_denoiser_type` presence, `state_scale` | ran, exit 0 |
| `fix_examples.py` -> `examples_fixed.json` | the pinned 200/200/200 with per-row prompt sha256 and a 64-hex prompt-set hash per task | ran, exit 0 |
| `run_five_arms.py` | the five-arm runner | written; statically validated on CPU; executed on GPU by job groups G2/G3 |
| `sanity_checks.py` | 9 checks, 34 positive controls, TP=34/34 FP=0/9 | ran, exit 0 |
| `PLAN.md` | this file | -- |

## Step 1 (was BLOCKING for A2; now RESOLVED) -- the training-mean latent

A2 could not run without a `mean_train_latent`, and one did not exist. The runner
refuses to substitute anything:

```
A2 requested but --mean_latent is missing. The training-set mean latent
does not exist yet; see PLAN.md step 1. A zero vector is a DIFFERENT
control, not a substitute, and this runner will not silently use one.
```

**Resolved 2026-09-23.** `scripts/eval/latentuse/g1_mean_latent.pt` exists with
`g1_mean_latent_provenance.json` recording `n_spans_used=4096`,
`n_spans_requested=4096`, `split="train"`, `split_examples_count=304052`,
`prefix_encoding_rule="posterior_mean"`, and
`mean_taken_in="z space (posterior means). R(E z) != E R(z): the writer R is NOT
applied here and no states were averaged."` A2 is no longer blocked.

Requirements, all of which come from the appendix prose at `appendix.tex:714`:

* **Training split only.** `preprocessed_data/owt_rwkv_tokens/train` (304052 npz
  files), which is the checkpoint's own `cfg.data.token_dir` — confirmed as
  `corpus_revision.token_dir` in the provenance. Never from the evaluation
  questions and never from gold answers; the appendix requires "a training-only
  mean latent, not an evaluation-set or target-informed mean".
* **Mean in z space, not in state space.** `R(E z)`, never `E R(z)`. The appendix
  states this explicitly because the writer is nonlinear: `R` is
  `s1_trunk -> (s1_u_head, s1_v_head) -> (U @ V^T) * state_scale`
  (`models/state_hijacking_dit.py:1167-1177`). The per-layer product
  `U[:,l] @ V[:,l].T` is a product of two z-dependent factors, so it does not
  commute with averaging.
* 4096 spans, fixed seed, encoded with the same `encode_prefix`
  (`scripts/eval/sample_prefix_suffix_cfg.py:62`) the runner uses. The runner
  imports it rather than reimplementing it
  (`run_five_arms.py:83-84`), so the pooling rule cannot drift between the mean
  and the arms that consume it.
* A zero vector is a **different control**, not a fallback. If step 1 had failed,
  A2 would be reported blocked and the other four arms would still run.

### Finding from step 1 that changes how the table must be read

The latent is collapsed: the encoder is close to input-independent on this
checkpoint. From the full 4096-span run and from the sampled task latents:

* `||mean|| = 9.178606` against `mean_of_individual_l2_norms = 9.178618`, a ratio
  of `0.9999987`. For iid zero-mean vectors that ratio would be
  `1/sqrt(4096) = 0.0156`.
* `cos(z_i, mean)` median `0.99999881`; `||z_i - mean||` median 0.16% of `||mean||`.
* Across 12 DDIM-CFG task latents from **different prompts**, pairwise
  `cos` is at minimum `0.999988`.
* `cos(training mean, sampled mean) = 0.999997`.
* Effective rank of the **centered** training latents is 8.4 / 32, so residual
  structure exists, but it lives in ~0.2% of the vector magnitude.

Mechanism, verified on CPU from the weights alone (no forward pass):

* `mu_head.weight` has spectral norm 1.22 and Frobenius norm 1.417 over a
  10240-dim input, while `mu_head.bias` has norm 0.0277.
* `logvar_head.bias` is exactly `-3.0` on all 32 dimensions, i.e. still at its
  initialisation value, so `sigma = exp(-1.5) = 0.2231` per dimension is injected
  unconditionally — `_encode_pooled`'s variational branch reparameterises with no
  `self.training` guard.

This is the known champion-KL collapse, independently reproduced here from three
directions (training spans, sampled task latents, and the 200-example seed-42 z
cache).

**Sampling noise exceeds prompt-dependent variation.** Measured on G2's seed-42 z
cache over 200 sciq prompts: `mean||z_i - mean z|| = 0.0285` against
`||mean z|| = 9.2041` (0.31%), `cos` to mean `0.99999565`. Across seeds on the
**same** prompt, `mean||z_42 - z_43|| = 0.0344`.

⚠ **SMOKE-LEVEL ONLY, DO NOT PUT NEAR THE PAPER.** The comparison that would turn
this into a claim — across-prompt spread `0.0231` vs across-seed spread `0.0344`,
ratio **1.49** — was computed on **n = 6 shared examples**. Six pairs cannot
separate two variances: there is no usable CI, and the ratio is not distinguishable
from 1.0 at that n. It is recorded here as an observation to be *re-measured by the
real run*, not as a result. Any sentence of the form "`z` carries less information
about which question was asked than about which seed drew it" is **not yet
supported** and must not be written until the ratio is recomputed on the full
200-example overlap with an interval attached. The n must appear inline wherever
the ratio is quoted; a ratio without its n is how this one nearly escaped.

What *is* supported at this point is only the single-seed geometry above (200
prompts, 0.31% relative spread, cos 0.99999565).

**`erank` does not rebut this.** The centered training latents have effective rank
8.4/32 and the sampled ones up to ~22.7/32, but that structure lives in ~0.2-0.3%
of the vector magnitude. A high `erank` alongside a 0.3% relative spread is not
evidence against collapse, so any collapse claim must report **magnitude and
`erank` together**.

**Consequence for the table, which must be stated and not left for the reader to
infer: A2, A3 and A4 differ by less than 0.5% in `z`, so they are nearly the same
intervention** — at bf16, A2 and A4 write almost the same vector. If those three
rows come out near-identical, that is the encoder, not a bug in any runner, and not
evidence that the write mechanism is sound. Conversely A3 (the derangement) is a
**weak manipulation by construction**. Report the observed `cos(z_i, z_donor)`
distribution next to the A3 row so the strength of the manipulation is visible
rather than assumed.

Correction to an earlier framing of mine: I called A2 "the arm most likely to emit
degenerate text". That does not follow — since A2 and A4 write vectors differing by
under 0.5%, the empty-string scorer hole matters about **equally** for both. What
*is* true, and is the binding constraint, is that A2 has **no rows at all**, so no
exposure measurement can speak for it.

## Step 2 -- `prefix_replay` is a measured axis, not a setting to pick

`--prefix_replay {none,refeed}` applies identically to all five arms and is
recorded in `run_meta` and in every row. Default `none`.

* `none` = prefill `prompt[:-1]`, write, feed `prompt[-1:]` once. Every prompt
  token consumed exactly once; 0 tokens re-fed; first-token logits come from the
  written cache. This is the training-shaped path: `_singlez_coadapt_ce_from_z`
  (`models/state_hijacking_dit.py:1972`) prefills the prefix, injects, then
  forwards the suffix.
* `refeed` = forward the whole prompt, write, forward the whole prompt again.
  This is what `generate_answer`
  (`scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py:239-246`) does — the same
  `input_ids` is passed to `rwkv_model` at line 243, then again at line 245 with
  the injected cache — and it is therefore what **every published number for this
  checkpoint** came from.

**Run both. Do not pick one.**

### The `none` result, and the control that decides what it means

Measured on GPU, seed 42, 6 examples per task:

| task | arm | n | correct | distinct generations |
|------|-----|---|---------|----------------------|
| sciq | A0 | 6 | **5** | 6/6 |
| sciq | A1 | 6 | 0 | 5/6 |
| sciq | A3 | 6 | 0 | 3/6 |
| sciq | A4 | 6 | 0 | 2/6 |
| commonsenseqa | A0 | 6 | **5** | 6/6 |
| commonsenseqa | A1 | 6 | 1 | 6/6 |
| commonsenseqa | A3 | 6 | 0 | 3/6 |
| commonsenseqa | A4 | 6 | 0 | 2/6 |

Smoke macro3: A0 77.8, A1 5.6, A3 0.0, A4 0.0. A0 gives 18/18 distinct 50-char
prefixes, A4 gives 7/18. A0 and A4 differ on 12/12 examples.

**A0 does not collapse, and that asymmetry is the finding.** A3/A4 emit
prompt-independent boilerplate ("The New York Times, the Washington Post and...")
while A0 answers the question ("(B) bookstore\nExplanation: Magazines are usually
found in bookstores"). The mechanism is `inject_into_cache`
(`models/state_hijacking_dit.py:2489-2492`): `state["recurrent_state"] = st` is an
unconditional full overwrite, so under `none` the arms that **write** lose the
prefix and the arm that does **not** write keeps it. The write is not inert; it is
actively destructive. A0 vs A4 differing on 12/12 examples rules out "the runner
is broken under `none`", because a broken `none` path would damage A0 too.

This ordering matters for how the claim is written. The earlier draft of this file
said "under `none`, A3 and A4 return 0/18" and left open whether all five arms
collapse. A0 was the only arm that could falsify the general claim, and it had not
been run. It has now, and it **refutes** the general form: do not write "all arms
collapse under `none`". Write "every arm that writes collapses under `none`; the
no-write arm does not" — which is a statement about the write, not about the
harness. (The general form would have been the "tested arm != all arms" trap: the
two arms originally tested were both overwriting arms, i.e. the same arm twice.)

Two things follow, and both belong in the appendix rather than in a choice of flag:

1. "Training-consistent" and "produces a usable eval decode" are different
   properties. At training the suffix carries the content; at eval under `none`
   there is no suffix, so nothing restores prompt conditioning after the
   overwrite.
2. The headline table should be `refeed` (comparable to every published number)
   with `none` reported beside it as the training-consistent control. All seeds
   must match at each setting; the two settings are not comparable to each other.
   **Do not drop the `none` rows** — they are the measurement that shows the write
   destroys prefix conditioning, which no `refeed` row can show, because `refeed`
   repairs the damage before decoding.

**Coverage gap to close.** Seed 42 has `refeed` for A0 and A4 only (a side
measurement appended to the main run), not for A1/A2/A3. A five-arm `refeed` run
at seed 42 is a **separate job** and is not yet authorised. Until it exists, the
headline `refeed` column cannot be filled for A1/A2/A3 at seed 42 and those cells
stay pending rather than being back-filled from another seed.

## Step 3 -- the five-arm runs

```bash
python3 scripts/eval/latentuse/run_five_arms.py \
  --ckpt_dir outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000 \
  --examples scripts/eval/latentuse/examples_fixed.json \
  --provenance scripts/eval/latentuse/provenance.json \
  --arms A0,A1,A2,A3,A4 --tasks sciq,commonsenseqa,race \
  --prefix_replay refeed --seed 42 --seed_stream additive \
  --mean_latent <step-1 output> \
  --z_cache <dir>/z_cache_seed42.pt \
  --out <fresh dir>
```

Then the same with `--prefix_replay none`, and both with seeds 43 and 44.
`--out` refuses an existing non-empty directory and refuses any path containing
`results` or `outputs` unless it also contains `latentuse`, so no measurement
directory can be overwritten.

Seeds: A3 and A4 share one sampled `z` per example via `--z_cache`, so the two
arms are paired by construction and differ only in which row's latent is written.
Per-example outputs are written to JSONL so the paired confidence intervals the
appendix asks for can be computed afterwards.

**`--seed_stream` matters for what may be claimed about seeds.** The default
`additive` stream is `seed + 100003*(task+1) + i`. It is collision-free within one
seed (600/600 distinct) but NOT across seeds: `(s, i)` and `(s+1, i-1)` map to the
same integer, so seeds 42/43/44 share 597/600, 597/600 and 594/600 of their noise
draws at relabelled positions. Per-example `z` still differs across seeds because
`cond` is fixed, so the observed spread is real, but **the three runs are not
independent and the appendix must not claim they are.** `multiplicative`
(`seed*1000003 + ...`) shares 0/600 and is available for any future run; the two
are not comparable, so do not mix them within one table.

## Step 4 -- what to check before believing any number

Run `sanity_checks.py` on the same node, then verify from the emitted rows:

* Every row carries `prefix_replay`, `seed_stream`, `adapter_name`,
  `effective_lambda`, `state_scale`, `prefill_tokens`, `prefix_replay_tokens`,
  `cache_policy`, and `first_token_logits_top10`.
* A1's `a1_layers_restored` is 32 on every row. The runner raises if the restore
  is not bit-exact (`torch.equal`) and raises if it restored 0 layers, because a
  check with no denominator cannot fail.
* A3's donor differs from the recipient **as a tensor**, not merely by index —
  with a near-constant encoder an index-level derangement can still hand back a
  numerically identical vector.
* Denominators are exactly 200 per task, 600 macro, with invalid rows counted as
  wrong and never dropped.

### Two instrument defects found and fixed during preparation

Both were found by the calibration, not by reading the code, and both are now
positive controls:

1. **The scorer credited empty predictions.** `letter_of` tested
   `t.upper() in LETTERS` where `LETTERS` is the *string* `"ABCDEFGH"`, so
   `"" in LETTERS` is True and `LETTERS.index("") == 0`: an empty generation was
   scored as choice A with `invalid=False`. Because `truncate_at_newline` runs
   first, any generation merely *starting* with a newline was truncated to `""`
   and scored as A too — so `"\n(B) library"` and `"\nbookstore"`, which are
   *correct answers*, were scored as choice A. Measured on the pinned 200s, an arm
   that emits nothing scored 49/200 SciQ, 34/200 CQA, 37/200 RACE — macro
   **20.0% with zero invalids**. The direction matters: the defect could only
   **inflate** an arm, and it inflated the most degenerate arms hardest, which are
   exactly the controls whose collapse the table exists to detect. Fixed
   (`letter_of` returns `None` on empty and only ever one character;
   `letter_index` replaces every `LETTERS.index` call site). Now: empty -> 0/600
   correct, 600/600 invalid; gold-as-prediction -> 60/60 correct.

   **Correction to the mechanism, and it narrows the blast radius.** An earlier
   draft of this file said two-letter runs like `"AB"`/`"CD"` were mis-scored as
   A/C. That is wrong about *why*. `str.index` is a **substring search**, so
   `LETTERS.index("AB") == 0`, `LETTERS.index("CD") == 2`, `LETTERS.index("GH") == 6`
   — every contiguous two-letter run already mapped to the index of its **first**
   letter, which is exactly what `letter_index()` now does deliberately. Verified
   on CPU: `all(LETTERS.index(LETTERS[i:i+2]) == i)` is True. So the two-letter
   path was accidentally *correct*, not wrong, and a **non-contiguous** run like
   `"BA"` raised `ValueError` rather than mis-scoring. The empty-string hole is the
   real defect; the two-letter behaviour is a latent fragility that would break the
   moment `LETTERS` were reordered or made non-contiguous.

   **Consequence: no previously published or in-flight number needs rescoring**
   because of the two-letter path. The empty-string hole is the only behavioural
   change, and its exposure is measured below.

### Scorer exposure on the rows that exist (differential rescore, read-only)

Both scorers were run over every stored row and compared on `score_row`. **These
counts move as jobs write rows — re-run before quoting.** Last re-measured
2026-09-23 18:2x, when the row set had grown from 108 to 144:

```
DENOMINATOR files=31 rows=144
arm counts:           A0 18   A1 18   A3 54   A4 54   (A2: 0)
prefix_replay counts: none 144   (refeed: 0)
seed counts:          42 -> 72   43 -> 72   (44: 0)
empty-after-truncate      = 0/144
two-letter (non-single)   = 0/144
SCORER DISAGREEMENTS      = 0/144
per-arm: A0 0/18  A1 0/18  A3 0/54  A4 0/54
POSITIVE CONTROL: plant "ab" into 20 real rows -> 20/20 flagged
NEGATIVE CONTROL: same scorer twice            -> 0/144  (must be 0)
```

**The comparator was proved able to fail before that 0 was believed.** Injecting
`"ab"` into real rows flags 20/20, and the same-scorer-twice negative control gives
0/144. A zero from a comparator that has not been shown to separate the two
scorers is indistinguishable from a vacuous scan, and one such scan was in fact
reported and retracted during this work.

**Scope limit on the corpus sweep.** The finding that the 145 triggering rows are
"all of kind `''`, zero two-letter" is a statement about the **145,636-row
published corpus at the time it was scanned**, not about rows that did not yet
exist. The five-arm rows are re-scored on every aggregation run, so if A2 ever
contributes a two-letter hit it will surface in that count rather than being
assumed absent.

**One residual divergence exists.** The frozen and fixed scorers disagree on
lowercase two-letter input: `"ab"` -> frozen `(True, False, 0)` vs fixed
`(False, True, None)`, because `t.upper() in LETTERS` uppercases before the
substring test while `letter_index` requires a single character. A 544-case sweep
that reported "behaviourally identical" did not cover lowercase runs. Exposure on
real rows is 0/144 (last re-measure), so nothing moves today, but the two scorers are **not**
identical and must not be described as such.

**Coverage limits on that zero — this is the part that must not be dropped:**

* **A2 has 0 rows.** All `latentuse_g2_out/seed42/main/*.jsonl` are empty; the rows
  are smoke only. A2's mean latent is loaded and live (`latent_norm` 9.1786,
  matching G1's provenance, so not a zero vector) but has produced no scored rows,
  so **no exposure measurement can speak for it.**
* **`prefix_replay` is `none` on every row.** There is no `refeed` coverage.
* Arms covered: 4 of 5. Settings covered: 1 of 2. Seeds present: 42 and 43.

### The arms are mechanically distinct — checkable from the data, not just the source

The recorded cache fingerprints live **nested inside `cache_policy`**, not at the
top level: `cache_policy.cache_fp_before_sha256` / `cache_fp_after_sha256`,
alongside `deepcopy_before_mutation`, `conv_ffn_zeroed_by_adapter`,
`seen_tokens_reset_by_adapter` and the `cache_seen_tokens_*` counters. Verified
over all 144 rows:

```
arm: before == after / total
  A0: 18/18 identical   (missing fp = 0)
  A1:  0/18 identical
  A3:  0/54 identical
  A4:  0/54 identical
```

with `A0 adapter_name='none'`, `effective_lambda=None`, `latent_norm=None` versus
`A4 adapter_name='inject_into_cache'`, `effective_lambda=1.0`. So **A0 is a true
no-op and every writing arm actually mutates the cache** — the lambda=0 trap is
excluded from the *data*, not merely from a source reading. This is the strongest
available evidence that the `none`-mode collapse is destructive rather than inert.

A1's restore is exact where it must be and inexact where it should be: on a
3-layer cache, 3/3 layers restored with `recurrent_state` bit-identical on every
layer, and the only differing fingerprint fields are L0-L2 `conv_state` and
`ffn_state`. Negative control (same object twice) equal; positive control (one
scalar flipped) detected and named `L0.conv_state`. **So A1 minus A0 isolates the
adapter's auxiliary-cache damage, which is exactly what that row is for.**

Two harness traps encountered while establishing that, worth avoiding on re-check:
a fake cache declaring `layers = None` as a class attribute makes `hasattr` True so
`_cache_layer_state` takes the wrong branch; and `cache_fingerprint` returns its
per-field data under a `'fields'` key, so a naive top-level diff collapses to the
single name `'fields'` and reports "all differing fields are aux: False". Both were
instrument bugs, not defects in the code under test.

### ⚠ A1 scores 0.00 on sciq ONLY BECAUSE OF A SCORER BUG — see the two corrections below

> **⛔ THIS SUBSECTION IS SUPERSEDED TWICE.** (1) Scope: the "A1 is the floor"
> generalisation was drawn from sciq alone and is retracted. (2) Mechanism: the
> `0.00` figures for A1 below are a **scoring defect, not model behaviour** — with
> the defect repaired A1 scores **42.00 on sciq and 29.50 on cqa**. Both
> corrections are documented at the end of this file
> ("SCORER DEFECT: A1's 0.00 was never model behaviour"). The numbers below are
> faithful to the shipped scorer, which is why they remain the table of record,
> but **no claim about the model may be built on A1's `0.00`.**
> **What survives unchanged: A2/A3/A4 are 0.00 under both scorings with 0 rows
> changed, so latent content remains unanswerable from this run.**

**The real sciq run is on disk and I verified every number below myself** from
`latentuse_g2_out/seed42/main/sciq__A*.jsonl` (5 files x exactly 200 rows):

```
arm   n   correct     acc   distinct preds   empty
A0   200      166   83.00%            198        0
A1   200        0    0.00%             78        0
A2   200        0    0.00%             29        0
A3   200        0    0.00%             37        0
A4   200        0    0.00%             36        0
```

**A1 restores `recurrent_state` bit-exactly on 32/32 layers and still scores 0.00
against A0's 83.00** *as measured by the shipped scorer* — but that 0.00 is a
scorer artefact (A1 verbalises "the answer is X" on 188/200 sciq rows and the
scorer never strips that lead-in before letter extraction; corrected value
**42.00**). A1 differs from A0 in *exactly* the conv/ffn zeroing (see the
runner's own arm table above; `restore_principal_memory` deliberately leaves
auxiliary fields and counters as the adapter left them, and raises if the restore is
not bit-exact). The inference below was therefore **wrong** and is kept only to
mark what it was:

> **⛔ WITHDRAWN OUTRIGHT — On sciq, zeroing `conv_state`/`ffn_state` alone is
> already sufficient for the full collapse.** False: after zeroing, the model still
> answers at **42%** (corrected scoring). The real cost of zeroing conv/ffn is
> about **−41 points** (83.00 → 42.00), not −83.
> The rest of this box still stands, and is the part that matters:
> A2, A3 and A4 all sit at or below a floor that auxiliary-field damage has already
> imposed. Their mutual near-equality says **nothing** about whether latent
> *content* is used — that comparison is **saturated**, and reporting "A2 ≈ A3 ≈ A4"
> as evidence about latent content would be reading a floor as a result.

This is not an aliasing artefact: A0 vs A1, A0 vs A2/A3/A4 and A1 vs A2/A3/A4 all
differ on **200/200** predictions, so the restore is real and the arms are genuinely
distinct; only A2/A3/A4 resemble each other (A3 vs A4 differ 77/200, A2 vs A4
127/200).

**Prompt conditioning is destroyed, not merely unhelped.** Distinct predictions over
200 distinct prompts: A0 **198**, A1 78, A2 **29**, A3 37, A4 36. A2 collapses
hardest, as expected for a single mean latent. An arm that emits 29 distinct outputs
for 200 distinct questions is not conditioning on the prompt.

Mechanism confirmed at source: `inject_into_cache`
(`models/state_hijacking_dit.py:2503-2509`) zeroes `conv_state`/`ffn_state` when
`force_zero or not is_rnn_denoiser`. Two legs; **leg 2 is the one that fires.** Leg 1
(`force_zero`) is eliminated three ways: `RELAY_INJECT_ZERO_RNN_CACHE` defaults to
`"0"`, the as-run provenance records `'0'`, and the submitted job command contains 0
occurrences of the variable.

**State leg 2 by the DEFAULT, not by this checkpoint's key.** `:912` is
`str(getattr(config, "global_denoiser_type", "dit")).lower()` and `:2504` accepts
only `("rwkv", "birwkv")`, so **any** `dit` global denoiser zeroes the auxiliary
fields — whether the key is absent *or* present with value `dit`. I had written "this
checkpoint has no `global_denoiser_type` key", which is true here but is falsified by
a checkpoint that carries the key explicitly; the default-value phrasing cannot be.
**Second-level form of "quote the gate that actually fired": also quote the right leg
for a non-contingent reason.** A cause that happens to hold is not the same as the
cause that must.

Recorded per row as `cache_policy.conv_ffn_zeroed_by_adapter = true` for A1-A4 and
`false` for A0.

Controlled-comparison invariants verified single-valued across all five arms:
`checkpoint_sha256`, `boundary_ids_sha256`, `seed` (42), `sampler`
(`ddim_cfg(steps=100,cfg=3.0)`), `prefix_encoding_rule`, `git_commit`, `task`.
`cache_policy_hash` is 5-distinct **by design** (one per arm). A3 derangement audit:
**0/200 fixed points, 200 distinct donors**. A0 confirmed non-writing
(`adapter_name='none'`, `latent_norm=null` on all 200 rows). `prefix_replay='none'`
on every arm, `state_scale=0.1845703125` (the bf16 value) everywhere.

**Consequence for the appendix table.** The headline contrast that survives is
**A0 83.00 vs A1 0.00** — a same-checkpoint, same-prompt, same-seed demonstration
that the adapter's auxiliary-cache handling is destructive on its own. The
latent-content question (A2 vs A3 vs A4) is **not answerable from this run** because
all three are below the floor. Say that explicitly rather than printing three
indistinguishable numbers and letting the reader infer that latent content does not
matter.

### refeed vs none is a controlled contrast, not a confounded re-sample

The side `refeed` run passes the **same** `z_cache_main.pt`, and the runner skips
re-sampling any key already present, so `refeed` and `none` differ **only** in
`prefix_replay` with identical latents. That makes the pair a controlled contrast —
worth stating explicitly, because a re-sampled latent would have confounded the
one comparison the `none` rows exist to support.

So the honest statement is "0/N on the smoke rows that exist at the time of the
scan (144 at last re-measure), comparator
verified live, A2 and `refeed` unmeasured" — not "unexposed". Re-run this
differential over the full output before any cell is reported, and if it is ever
non-zero, **rescore from the stored `prediction` field; never re-decode**, because
re-decoding silently changes the decode path and would make the rescored rows
incomparable to the rest of the table.

### The same hole in production, and its real blast radius

`scripts/eval/score_gen_match.py` carries the same `letter_of` (definition at
:63, the substring test at :80). Two facts have to be stated **together** or each
one misleads:

* The `LETTER_FIRST` path at :195 is gated behind `SCORE_LETTER_FIRST`, which
  defaults to `"0"` at :52 — so it is OFF by default.
* The fallback at **:212-214** (`lt = letter_of(gen)` /
  `pred = LETTERS.index(lt)`) is **not gated at all**, and that is the path every
  published number went through.

"The bug is at :80" understates it; "LETTER_FIRST is off, so we are safe" would
overstate the safety.

**My characterisation of the direction was wrong, and the correction runs against
us.** I wrote that the defect "inflated hardest exactly the degenerate control
arms". Swept over the real corpus — 833 jsonl files, 145,636 rows, 124,603 with
choices — **145 rows trigger it, all of kind `''` (zero two-letter)**, and the hits
are overwhelmingly **baselines** (`baseline-133b-chunked`, `baseline-gemma-2-2b`,
`baseline-rwkv-6-world21`), not our arms. The defect was inflating the models we
compare **against**, so fixing it moves numbers **in our favour** — which is
precisely why it must be reported rather than quietly fixed.

**The filter that matters, and that neither I nor G2 applied: "file lives under
`results/`" is not "this number is in the paper."** Judged against
`results/paper_ledger/cell_sources.json` (231 source files), of the 45 affected
files only **three** are published cell sources:

| row | cell | as printed | corrected |
|-----|------|-----------|-----------|
| RWKV-6 World2.1 | winogrande | 52.5 | 51.0 |
| RWKV-6 World2.1 | race | 25.0 | 24.5 |
| ELF-M | sciq | 4.0 | 3.5 |

11-task Avg: RWKV-6 37.7 -> 37.5; ELF-M 4.3 -> 4.3 (printed value unchanged).
Every delta is negative and on a baseline.

Related trap, worth recording as a general rule: `baseline-gemma-2-2b/winogrande`
looks affected but scores 8.00 against the paper's 52.5, because Gemma's six fix2
columns actually come from `fix2choices/baseline-gemma-2-2b/`. **A file being
affected does not make it the file the printed cell came from** — resolve the cell
to its source before claiming a delta.

**Ledger to cite: `table_capability_14col_gluefix.json`, not
`table_capability_14col.json`.** Verified on CPU.

⚠ **CORRECTION — my earlier "the two differ only in ELF-M's sciq cell (3.0 -> 4.0)"
was FALSE.** They differ in **52 of 238 cells** (17 rows x 14 cols). I derived "one
cell" from a targeted three-row lookup and then described it as a diff.

**Rule this cost: before reporting "A and B differ in N places", enumerate all of
them and print the count. A sampled look is not a diff.** The same rule applies to
existence claims (see the 108 -> 144 row staleness note) — a targeted probe answers
only about what it probed.

Enumerated (`acc` per cell, plain -> gluefix): 52 differing, of which **28 are
float-formatting noise** (|d| <= 0.01, e.g. `56.49999999999999` vs `56.5`) and
**24 are real**:

(Reconciliation of a competing count, settled by dumping members on both sides. A
peer first reported 40 differing / 16 noise and attributed my 52 to counting
top-level `old`/`new`/`mean14` keys; that attribution was wrong — my comparator only
ever reads `cells[col]["acc"]`, and all 52 differences are in that field. The peer
re-enumerated with `repr(a) != repr(b)` and **independently arrived at 52**: its
`abs(d) > 1e-9` cutoff had silently dropped the 12 repr-only cells whose true delta
is ~7e-15. Final agreed classification, each class enumerated in full:

  * **16 `squad`-precision** — plain keeps full float precision, gluefix rounds to
    4 dp (`50.286524440471815` -> `50.2865`). This is **every** row that has a
    `squad` accuracy: 16 of 16 differ, 0 identical. (`STAR-LDM` has no `squad` cell
    in gluefix at all.)
  * **12 repr-tidy** — `56.49999999999999` -> `56.5`, `28.999999999999996` -> `29.0`.
  * **23 real** (|d| > 0.01) **+ 1 presence** (`STAR-LDM arc_challenge`
    `2.631578947368421` -> `None`) = the load-bearing **24**.

  16 + 12 + 23 + 1 = **52**. Both sides now agree on all four class sizes.

⚠ **I had written "13" for the squad class. It is 16, and the members were sitting in
my own printed dump — I counted a 16-line block by eye instead of by `len()`.** Same
defect as the original one-cell claim, one level down: I enumerated, then summarised
the enumeration without measuring it. If a count is worth asserting, take it from
`len()`, not from looking at the list.

Two lessons, one each way. Mine: when two tallies disagree, dump the disputed class's
members rather than inferring the other side's denominator. The peer's, which is the
better half: **its epsilon WAS a denominator choice.** `1e-9` looks neutral and
silently converted a whole class of real differences into nonexistent ones. A
tolerance in a diff is a claim about what counts as "same" and has to be declared
like any other selector.)

```
LACES Global    race            42.5 -> 68.0   +25.5
LACES Chunked   race            52.5 -> 64.0   +11.5
LACES Chunked   commonsenseqa   61.5 -> 71.0    +9.5
LACES Chunked   sciq            79.5 -> 87.5    +8.0
LACES Chunked   arc_easy        70.0 -> 75.0    +5.0
LACES Global    commonsenseqa   56.5 -> 61.0    +4.5
LACES Chunked   winogrande      48.0 -> 51.5    +3.5
MDLM-OWT        piqa             0.0 ->  2.5    +2.5
LACES Chunked   arc_challenge   58.5 -> 60.5    +2.0
MDLM-OWT        commonsenseqa    1.0 ->  2.5    +1.5
ELF-M           mmlu             1.0 ->  2.0    +1.0
ELF-M           sciq             3.0 ->  4.0    +1.0
LACES Chunked   piqa            67.5 -> 68.5    +1.0
MDLM-OWT        mmlu             2.5 ->  3.5    +1.0
MDLM-OWT        race             0.5 ->  1.5    +1.0
CoLA-DLM        arc_challenge   20.5 -> 21.0    +0.5
ELF-M           arc_easy         2.0 ->  2.5    +0.5
ELF-M           obqa             0.5 ->  1.0    +0.5
ELF-M           story_cloze      0.5 ->  1.0    +0.5
MDLM-OWT        story_cloze      6.5 ->  7.0    +0.5
STAR-LDM        arc_easy        11.0 -> 11.5    +0.5
STAR-LDM        mmlu             7.5 ->  8.0    +0.5
STAR-LDM        siqa            11.0 -> 11.5    +0.5
STAR-LDM        arc_challenge   2.6315789... -> None
```

**Every real difference is non-negative except the STAR-LDM cell that becomes
`None`, and the ledger choice is worth +2.7 and +3.5 points on our own two rows.**
11-task averages (drop `arc_challenge`, `hellaswag`, `squad`):

```
row               plain    gluefix   delta      <- printed
LACES Global      61.50 ->   64.23   +2.73         64.2
LACES Chunked     60.95 ->   64.45   +3.50         64.5
ELF-M              4.00 ->    4.32   +0.32          4.3
MDLM-OWT           4.45 ->    5.05   +0.59
RWKV-7 World3     52.82 ->   52.82   +0.00         52.8
RWKV-6 World2.1   37.68 ->   37.68   +0.00         37.7
(all 11 remaining baselines: +0.00 exactly)
STAR-LDM           None ->    None   (6 of 11 cells None in BOTH files)
```

**`main.tex:103`'s "52.8 to 64.5", `main.tex:111`'s "64.5%", `main.tex:314`'s "64.2
and 64.5 against 52.8", and `main.tex:456`'s "11.7 points" all depend on this ledger
choice.** Under the plain ledger the lift is 60.95 - 52.82 = **8.13**, not 11.7.
The printed `\method{} Chunked` row (`main.tex:777`, `appendix.tex:535`) matches
gluefix cell-for-cell, so the paper is internally consistent with gluefix — but the
gap between the files is a headline-sized quantity, not a rounding detail.

**Asymmetry — RESOLVED. Reading (a) confirmed, (b) excluded.** I had left open
whether "+0.00 on all 11 strong baselines" meant the defect only bites malformed
generations (a) or the fix was applied to a subset of rows (b). It is (a), and the
mechanism is a specific rule in `scripts/eval/score_gen_match.py:82` (`letter_of`,
third rule), which credits an option letter glued to a following capital:

```python
m = re.match(r"^[\(\[]?([A-Ha-h])(?=[A-Z])", t)
```

Reproduced through the **real scorer** (in-memory monkeypatch of `letter_of` only;
nothing on disk touched, `sha256 3fc6252e…c27ee` unchanged). Nine cells, `race`
column, source files resolved via `cell_sources.json` rather than globbed:

```
row               glue-hits/n      ON     OFF    delta   ledger d  agree
LACES Global          66/200     68.00   42.50  +25.50    +25.50    YES
LACES Chunked         35/200     64.00   52.50  +11.50    +11.50    YES
RWKV-7 World3          0/200     55.00   55.00   +0.00     -0.00    YES
Qwen2.5-3B             0/200     76.50   76.50   +0.00     +0.00    YES
Llama-3.2-3B           0/200     62.50   62.50   +0.00     +0.00    YES
Phi-2                  0/200     66.00   66.00   +0.00     +0.00    YES
Gemma-2-2B             0/200     21.50   21.50   +0.00     +0.00    YES
Mamba-2                0/200     24.00   24.00   +0.00     +0.00    YES
GLA-1.3B               0/200     19.00   19.00   +0.00     +0.00    YES
```

**9/9 reproduce the ledger delta exactly.** The baselines' `+0.00` is a
**measurement** (0 glue-hits in 200 rows), not a skipped row. Controls: identical
rerun did not move (probe is sound), and restoring `letter_of` returned 68.00
(probe is reversible). On the largest cell, 66/200 rows change prediction, **51
newly correct, 0 previously-correct lost**, net +51/200 = +25.50 — the arithmetic
closes — and 51/66 = 77.3% against a 25.0% chance floor (all rows 4-option), so it
is signal rather than a length heuristic.

**RETRACTION of my own "no committed producer" claim — it was wrong in substance,
and my predicate was inert.** `grep -rl gluefix --include=*.py` returns 0 **by
construction**: `gluefix` is a *filename*, and a producer that writes to an `--out`
path need never contain that literal. A predicate that cannot return non-zero is not
a measurement — the same defect class as this suite's vacuous breaks. Better
predicates: `paper_ledger` = 9 files (4 of them `.py`, none writing
`table_capability_14col*`), `table_capability_14col` = 1, `mean14` = 1, `n_cols` = 0.

**One part of that claim survives, in narrowed form, and it is a real caveat:** the
scorer that produces these numbers is **untracked** — `git ls-files --error-unmatch`
errors, `git log --all -- scripts/eval/score_gen_match.py` = **0 commits**, status
`??`. So the *mechanism* is now reproducible on demand (verified above), but the
ledger-writing step itself still has no script in the tree, and the scorer that
implements the rule is not under version control. Note also the timeline: plain
04:29, gluefix 04:52, **scorer mtime 04:58** — the file has been edited *since* the
ledger was written, so today's `letter_of` is not provably the one that produced it.
The +25.50/+11.50 agreement is strong evidence that it is, but "reproduces the
number" is not "is the committed artefact that produced the number."

**Most important, and unchanged by any of the above: reproducible is not
defensible.** The generations this rule newly credits look like:

```
gold=2 pred=3  'DQ: 请将下文翻译成中文（简体）'
gold=1 pred=1  'BQ: 请将下文翻译成中文（简体）'
gold=1 pred=1  'BQ:'
```

An option letter followed by a runaway Chinese translation instruction. **+25.5
points on our headline row come from crediting these as correct answers.** Whether
that is a legitimate parse of a truncated answer or a scoring artefact that flatters
our arms specifically (0/200 hits on every autoregressive baseline, 66/200 on ours)
is a *judgement the paper must make explicitly*, not silently via a ledger choice.
Establishing the producer removes the provenance objection; it does not remove this
one.

Also: the `mean14` field in either file (ELF-M 3.21/3.46) is **not** the printed
average — the paper's "11-task Avg" drops `arc_challenge`, `hellaswag`, `squad`.
Quoting `mean14` as the table average would be wrong by ~0.9 on ELF-M.

**On the 11-of-14 subset — state it precisely.** Dropping
{`arc_challenge`, `hellaswag`, `squad`} from the gluefix ledger reproduces the
printed Avg on **11 of the 13 rows of `tab:capability` that print one**
(`main.tex:319-357`; 4 further rows — DCLM-1B, OLMo-2-1B, Pythia-2.8B, Mamba 2.8B —
are absent from the ledger entirely and are excluded from the denominator).
Verified row by row, including our own two:

```
OK  L352 LACES Global    printed 64.2  computed 64.23
OK  L353 LACES Chunked   printed 64.5  computed 64.45
OK  CoLA-DLM 29.7/29.68   ELF-L 3.5/3.55    ELF-M 4.3/4.32
OK  GLA-1.3B 28.0/28.00   Llama-3.2-3B 59.8/59.82   MDLM-OWT 5.0/5.05
OK  RWKV-6 37.7/37.68     RWKV-7 52.8/52.82  SEDD-medium 11.1/11.09
MISMATCH L330 Gemma-2-2B  printed 52.0  computed 27.27
NONE     L349 STAR-LDM    printed 11.6  (only 6 of 11 cells present)
--> 11/13 reproduce, 1 mismatch, 1 uncomputable  (11+1+1=13)
```

The two misses have documented separate provenance: Gemma-2-2B is BOS-restored per
`main.tex:734` and sourced from `fix2choices/`, and STAR-LDM cannot be computed from
this ledger at all. On `tab:capability-full` the same subset gives **14/16**
(same single mismatch, same single uncomputable).

**Do not describe this ledger as "the source of the table"** — it reproduces most of
it, and two rows come from elsewhere.

Uniqueness *is* established, by enumeration rather than assertion: of all
**C(14,3) = 364** three-column drops, exactly **1** achieves (11 reproduced,
1 mismatched), and it is this one. The runner-up manages 4/13. Note the denominator
trap I hit while checking this: a first pass scored 9/11 because the row-name
normaliser silently dropped `\method{}` and `\chunkmethod{}` — **our own two rows** —
and a second pass mis-parsed `0.99` out of an unrelated ratio column into four
targets. Print the row denominator and the skipped-row list before believing any
such tally.

`score_gen_match.py` is **left unmodified**: three in-flight jobs import it, and
with the blast radius now three baseline cells rather than an active mis-scoring of
our own arms, editing it mid-flight would be strictly worse than reporting it.
2. **A guard that could never pass.** `state_scale` is a trainable parameter, so
   `relay_utils.load_relay_model:100-102` casts it to bf16, while `provenance.py`
   read it from the fp32 checkpoint. bf16 rounds `0.1848476082086563` to
   `0.1845703125`, a `2.773e-04` gap, and the guard compared them with tolerance
   `> 0`. It aborted three GPU jobs. Fixed to require the loaded value to be the
   exact bf16 rounding of the recorded fp32 value, which still fires on a wrong
   checkpoint (a 1% different scale is 7 bf16 ulps away). A guard that always
   fires is as useless as one that never fires, and harder to notice because it
   looks like vigilance.

A third, self-inflicted: the first fix to (1) left `letter_of("AB") == "AB"`, and
one of the new positive controls **passed** — a broken instrument — because it
probed the indexer only through the already-fixed parser. Probing `letter_index`
directly on malformed input fixed it. This is why each check carries one positive
control per sub-assertion rather than one per check.

## Known benign warning

`[load_relay_model] step_00050000: missing=1059 unexpected=0` is not a silent
weight drop. The HF index at `models/RWKV7-Goose-World3-2.9B-HF-fla-v2` contains
exactly 1059 tensors; `ckpt["trainable_state"]` has 189 and contains no
`rwkv_model.*` key; no denoiser WARNING fires; `unexpected=0` means every
trainable tensor landed. So "missing" is the frozen decoder that
`from_pretrained` had already loaded. Recorded here so the next reader does not
re-derive it. (Contrast with the biRWKV case, where `strict=False` really did
swallow 473 weights — the distinguishing signal there was a denoiser WARNING plus
nonzero `unexpected`.)

## ⚠ The compliance check I used all session could not fail

Every report I made this session certified "nothing under `results/` or `outputs*/`
modified" using `git status --porcelain results/ | wc -l` = 0. **That probe is
vacuous.** `.gitignore:17` contains `results/` and `git ls-files results/` returns
**0**, so git tracks nothing there: the probe prints 0 whether or not those files
were written. I was reporting a guaranteed zero as evidence of compliance with the
task's one hard prohibition. Caught by a peer running the same check on its own
tree; confirmed here in one command. Same for "0 tracked modifications under
`latentuse_g2_out/`" — that path is untracked too, so the statement is true and says
nothing.

Also note the collapse trap in the other direction: plain `--porcelain` shows
`?? results/` as **one** line (easy to misread as a violation), `-uall` expands it to
188 untracked lines, and `--untracked-files=no` gives 0. Three different numbers for
the same tree, so the flag has to be stated with the count.

**Replacement predicate, with both controls, run over the full denominator:**

```
DENOMINATOR: 12 protected dirs (both repos)
  laces-reasoning/{results,outputs,outputs_ctrl50k,outputs_dynamic_basis,
                   outputs_dynamic_basis_nb32}
  DiffRwkv/{results,outputs,outputs_eval,outputs_hf,outputs_relay,
            outputs_star_ldm,outputs_state_prefill_block}

POSITIVE CONTROL (dirs I did write, same cutoff -- predicate must fire):
  /tmp/latentuse                        11 files   FIRES
  scripts/eval/latentuse                 5 files   FIRES
NEGATIVE CONTROL (cutoff 2027-01-01 -- must be 0 everywhere):
  all 12 dirs                            0 files   OK, not identically true

RESULT: find <dir> -type f -newermt '2026-09-23 18:00:00'
  all 12 protected dirs                  0 files
  widened to the whole of today          0 files
```

Cross-checked on the specific evidence I read: `table_capability_14col.json` and
`cell_sources.json` still 2026-09-21 04:52, `table_capability_14col_gluefix.json`
04:29, `baseline-global-dit-50k/race.jsonl` 2026-09-20 18:28 — all predating this
session. **The prohibition does hold. My evidence for it was worthless until now.**

One clarification the mtime sweep forced, which the git probe would never have
surfaced: files under `latentuse_g2_out/seed42/main/` *do* carry today's timestamps
(cqa at 18:24, sciq at 18:21). Those are the **running GPU job's** writes, not mine —
the cqa files were 0 lines when I read them at ~18:40, and `latentuse_g2_out/` is
that job's `--out` dir, which is exactly where new rows are supposed to go. A
compliance check has to attribute a write, not just detect one.

Attribution completed over the measurement-file scope named in the prohibition
(`*.json`/`*.jsonl` anywhere, not only the 12 dirs), for all of today:

```
protected-path hits (/results/ or /outputs*/), laces-reasoning:   0
protected-path hits (/results/ or /outputs*/), DiffRwkv:          0
everything else modified today, by owning directory:
  research/laces-reasoning/latentuse_g3   47 files   <- G3's --out dir
  research/DiffRwkv/logs                   5 files   <- job submit/request logs
```

So the honest form of the claim is attributive rather than merely detective: **0
writes under any protected path, and every measurement file modified today belongs to
a running job's own `--out` directory.** "Nothing was modified anywhere" would have
been the easier sentence and the false one.

**Rule: a compliance probe on a gitignored path is not a compliance probe.** Any
"nothing was modified" claim needs a positive control proving the instrument fires on
a directory that *was* written, and a negative control proving it is not identically
true.

## ✅ FULL THREE-TASK TABLE (n=200/cell, seed 42, prefix_replay=none) — independently verified

All 15 cells complete. **Every number below I recomputed myself** from
`latentuse_g2_out/seed42/main/{task}__{arm}.jsonl` by importing the frozen
runner's own `score_row`, not by reimplementation. Invalids count WRONG, so every
denominator is exactly 200.

| arm | as-run sciq/cqa/race | as-run Macro3 | lead-fixed sciq/cqa/race | lead-fixed Macro3 | rows flipped |
|-----|---------------------:|--------------:|-------------------------:|------------------:|-------------:|
| A0 no write | 83.00 / 62.50 / 68.00 | **71.17** | 83.00 / 63.00 / 68.00 | **71.33** | 1 |
| A1 restore original | 0.00 / 8.50 / 7.50 | 5.33 | 42.00 / 29.50 / 16.00 | 29.17 | 143 |
| A2 mean latent | 0.00 / 0.00 / 0.00 | **0.00** | 0.00 / 0.00 / 0.00 | **0.00** | **0** |
| A3 deranged latent | 0.00 / 0.00 / 0.00 | **0.00** | 0.00 / 0.00 / 0.00 | **0.00** | **0** |
| A4 correct latent | 0.00 / 0.00 / 0.00 | **0.00** | 0.00 / 0.00 / 0.00 | **0.00** | **0** |

**Gate (negative control): with the added branch disabled, my re-scorer reproduces
the stored `correct` field on 0/3000 rows.** No re-scorer may emit a number before
passing this.

**The publishable claim survives the third task intact: A2/A3/A4 are 0.00 in all
nine cells under both scorings, with 0 rows flipped anywhere.** RACE changes
nothing. `tab:latent-use-pending` can carry that and nothing stronger.

`letter_first` is now **measured, not inferred**: `False` reproduces stored
`correct` at **0/3000**, `True` disagrees on **9/3000**. The check can therefore
distinguish the two configurations, which is what makes the 0 evidence rather than
a tautology. (Grepping the submitted command for the flag gives 0 hits, but 0 hits
is indistinguishable from a bad grep — see my own blinded grep earlier in this file.)

### ⚠ "STRICT" IS UNDERDETERMINED — it names at least three different numbers

G2 retracted its A0 strict macro of 56.17 in favour of **47.17**, because its
standalone implemented "strict" by calling `match_choice()`, which **internally
contains containment and token-F1**: `match_choice("I think the correct option is
beta because", ["alpha","beta","gamma"])` returns **1** (I verified). So the
function used to *exclude* the loose stages silently re-admitted them.

I reproduced the aggregator's `_strict_pred` (`g2_aggregate.py:559`) verbatim and
**confirm A0 47.17 and A1 0.17 exactly**. But while doing so I found the deeper
problem: three defensible readings of "strict" give three different answers.

| definition of "strict" | A0 Macro3 | A1 Macro3 |
|---|---:|---:|
| exact + prefix only | **2.50** | 0.17 |
| exact + prefix + `letter_of(raw)` | **61.50** | 3.67 |
| aggregator `_strict_pred` (exact/prefix, then the same two on label-stripped text) | **47.17** | **0.17** |

All three reject the loose probe, so **the reject-probe cannot distinguish them.**
A0's strict score therefore spans 2.50 to 61.50 — a 59-point range — purely on
which stages one calls strict. **Never quote a strict number without printing the
stage list it admits**; "strict" is not a criterion, it is a family. A1 is 0.17 in
two of the three and 3.67 in the other, so the A0/A1 *gap* is robust even though
neither level is.

**Rule (G2's, and it generalises): when implementing a subset criterion, probe the
callee with an input the criterion must REJECT.** A function's name does not tell
you how strict it is.

### The arithmetic check that caught what comparison missed

G2's first standalone lead-fix reported A0 sciq 83.00 → 76.00 with **invalid
rising 19 → 26**. A fallback branch can only turn `None` into a prediction, so
**invalid is monotonically non-increasing and a rise is impossible**. Cause: a
missing `truncate_at_newline`, which is `score_row`'s first line. I reproduced the
mechanism — dropping that call makes A0 sciq invalid go 19 → **26**, exactly as
reported — and verified monotonicity holds in **0/15 violating cells** for the
correct implementation.

The instructive part: the **A1** cell in that same broken run read 44.50 against
the correct 42.00 — **2.5 points off, easily waved through as a tolerance
difference.** The impossible *direction* caught the bug; the plausible *magnitude*
would not have. **Prefer an invariant with a forbidden direction over a tolerance
on a magnitude.**

### A p-value from a cell whose positive control fails is not a measurement

My permutation-control fix is in the aggregator with a gate that raises if more
than 2/10 null-true replicates come out significant (positive control: shuffle
removed ⇒ exits 1 naming A0/sciq at 10/10; negative control: real data exits 0,
null-true medians 0.28–0.68, 0/10 below 0.05). All six nonzero cells clear
best-constant: A1 sciq 47.7% vs 26.7%, cqa 30.1% vs 24.5%, race 42.1% vs 31.6%.

**RACE has a cell worth knowing about:** A1 race is only **76/200** parseable
against 176 and 196 elsewhere. On **A4 race only 1/200 parses**, and there feeding
gold as the prediction yields **p = 1.00000** — so the test correctly declares that
cell **untestable** rather than reporting a comfortable null. Watch the positive
control, not the p-value.

*Fourth occurrence of "the positive control never reached the check", and this one
damaged the file under test:* G2's planted version exited 1 from an
`IndentationError` (an earlier edit had merged `def main() -> int:` with its first
statement), so it never reached the gate **and the real file was broken too**, found
only via `py_compile`. **Compile both versions before trusting a plant, and read
WHERE exit 1 fired, not merely that it fired.**

## ⛔⛔ SCORER DEFECT: A1's 0.00 was never model behaviour

**This supersedes the section below it AND my own confirmation of it.** I had
independently re-derived A1 sciq 0.00 / cqa 8.50 from the emitted `correct` field
and reported them as measurements. They are faithful to the shipped scorer and
**wrong about the model**.

**The defect**, verified by importing `g2_frozen_run_five_arms.py` and calling its
own functions (not by reimplementation). In `score_row` (`:307`) the control flow
is `:334` `match_choice(g)` → `:336` `match_choice(strip_label(g))` → `:338`
`match_choice(strip_answer_lead(strip_label(g)))` → `:340` `letter_of(g)`. Both
`letter_of` calls (`:329`, `:340`) receive the **raw** `g`; `strip_answer_lead`
(`:262`) is wired into the `match_choice` chain only. So:

| input | `letter_of(raw)` | `letter_of(strip_answer_lead(...))` |
|---|---|---|
| `"The answer is B.Q: In this task"` | **None** → invalid → WRONG | `'B'` |
| `"B. something"` | `'B'` | `'B'` (positive control: still scores) |
| `"The answer is obvious"` | None | **None** (the fix is not a loosening) |

**It zeroes exactly one arm,** because A1 is the only arm that verbalises. Rows
where the scorer's own `strip_answer_lead` actually removes something:

| | sciq | cqa |
|---|---|---|
| A1 | **189/200** | **148/200** |
| A0 | 12/200 | 1/200 |
| A2 / A3 / A4 | 0/200 | 0/200 |

*(Refinement to G2's account: it reported A0 at 0/200. Under the narrow phrase
"the answer is" that is right — A0 is 0/200 — but under the scorer's **own**
regex, which also matches a bare `"Answer:"`, A0 is 12/200 sciq and 1/200 cqa,
e.g. `'Answer: (B) chlamydiaQ: ...'`. The conclusion is unchanged; the denominator
depends on which rule you name, so name the rule.)*

**Corrected scoring** — one added fallback `letter_of(strip_answer_lead(g))`,
everything else a faithful copy of the shipped flow. I reproduce G2's numbers
exactly:

| cell | shipped | lead-fixed | delta | rows flipped |
|------|--------:|-----------:|------:|-------------:|
| sciq A0 | 83.00 | 83.00 | +0.00 | 0 |
| sciq A1 | 0.00 | **42.00** | **+42.00** | 84 |
| cqa A0 | 62.50 | 63.00 | +0.50 | 1 |
| cqa A1 | 8.50 | **29.50** | **+21.00** | 42 |
| **A2 / A3 / A4, both tasks** | 0.00 | **0.00** | **+0.00** | **0** |

**Negative control: the shipped scorer reproduces the stored `correct` field with
0/2000 disagreements**, so the harness really is reproducing that run. This also
pins a detail G2 did not state: that holds only at `letter_first=False`
(`letter_first=True` gives 4/2000), so **the run of record used the default**.

**A1's corrected score is real signal, not letter bias.** sciq 84/176 = 47.7% vs
chance 25%, best-constant 26.7%; cqa 59/196 = 30.1% vs chance 20%,
best-constant 24.5%. Marginal-preserving permutation (shuffle gold, hold
predictions fixed) gives **p = 0.00005 on both tasks**.

### ⚠ My own negative control for that permutation test FAILED, and the failure was mine

I first used G2's stated control "shuffle both" and got **p = 0.000 where ~0.5 was
required**. I nearly reported the permutation result anyway. The control was
**misconstructed, not the test**: shuffling both sequences induces *the same* null
as shuffling one, so it returns the identical p (I measured 0.00005 both ways) and
**cannot** come out at 0.5. Calling it a negative control was the error.

The correct construction is to make the null *true in the data* and re-run the
**whole** test: permute the prediction vector (preserving its marginal), recompute
the observed statistic, and take the p-value. Over 10 such null-true datasets:
sciq median **0.665**, 0/10 below 0.05; cqa median **0.600**, 0/10 below 0.05. Only
then does the main result stand.

**Rule: a control that is algebraically the same test cannot be a control.** Before
trusting a control, check that it *could* return the passing value — and if it
fails, suspect the control first; the failure is evidence about the instrument.

### What this retracts, including things I confirmed myself

- **"conv/ffn zeroing explains the entire collapse"** — withdrawn. Real cost ≈ −41
  points, not −83.
- **My "A1's cqa advantage is entirely a scoring artefact / bare-letter credit"** —
  inverted. Credit was being **withheld** by the scorer, not granted. My 0/17
  gold-string finding was a *true measurement of a broken scorer's output*, not a
  fact about A1.
- **My Correction 1 ("form preservation is cqa-only")** — premise dissolves. With
  the lead-in recognised, A1 emits a letter answer on 176/200 sciq rows, so the
  separation from the latent arms holds on **both** tasks. The single-cell-to-row
  caution was right; the correction went the other way.
- **UNCHANGED, and the only claim the appendix can carry:** A2/A3/A4 stay at 0.00
  under corrected scoring with **0 rows changed**. Latent content remains
  unanswerable from this run.

### How it was caught — the method is the finding

Two probes disagreed on one cell: a raw-text regex said A1 sciq emits 175/200
letter answers; the scorer parsed 1. The dangerous part is that an **earlier**
probe anchored at `^` also said 1/200 — **agreeing with the scorer**. Two
independent errors corroborated each other and read as confirmation.
**"Two channels agree" is evidence only if the channels do not share a blind
spot.** Here both missed the same lead-in. My own re-derivation shared it too: I
read `correct` straight from the files, which is the scorer's output, so I was a
third channel with the *same* blind spot and I reported agreement as confirmation.

*G2 also hit the inert-check trap again while gating the fix: its first two
controls passed with the fix replaced by `if False` (exit 0, deltas +0.00). A live
fix must **flip verdicts on rows it is defined for**, which is now asserted — 335
rows verbalise and scored wrong, the fix credits 125, and the inert version exits 1
naming itself.*

## ⛔ RETRACTED: "A1 saturates the table" / "A1 is the floor"

I wrote that section from **sciq alone**. It is wrong, and the correction comes
from commonsenseqa. I have re-derived the whole grid from the emitted `correct`
field rather than trusting either account:

| arm | sciq acc | sciq parseable | cqa acc | cqa parseable |
|-----|---------:|---------------:|--------:|--------------:|
| A0 (no write) | **83.00** | 181 | **62.50** | 178 |
| A1 (restore original) | 0.00 | 1 | **8.50** | **51** |
| A2 (mean latent) | 0.00 | 0 | 0.00 | 3 |
| A3 (deranged latent) | 0.00 | 1 | 0.00 | 3 |
| A4 (correct latent) | 0.00 | 1 | 0.00 | 2 |

> **⛔ THIS WHOLE SECTION IS ITSELF SUPERSEDED** by "SCORER DEFECT: A1's 0.00 was
> never model behaviour" immediately above. Its *scope* correction was right (A1
> is not the floor), but its *mechanism* is inverted: the numbers in the table
> below are the shipped scorer's output, and A1's real scores are 42.00 sciq /
> 29.50 cqa. Specifically **"A1's cqa advantage is a scoring artefact" is exactly
> backwards** — the scorer was *withholding* A1's credit, not granting it — and
> **"the separation is cqa-only" is false** once the lead-in is recognised (A1
> sciq emits 176/200 letter answers). Kept as the record of what I reported and
> when. The one line that survives every revision is the last: A2/A3/A4 are
> indistinguishable, so latent content is unanswerable from this run.

So A1 is **not** the floor: on cqa it sits strictly above the three latent arms,
and sciq is merely a cell where it also bottoms out. **One arm hitting the floor
in one cell is not that arm hitting the floor on its row** — the same
single-cell-to-row overreach I have now made twice.

**The two channels disagree, so both must be reported.**

*Accuracy channel — A1's cqa advantage is a scoring artefact.* **[INVERTED — see
the box above: the scorer was withholding credit, not granting it.]** Of A1's 17
credited cqa rows, the gold string appears in **0/17** generations (positive
control: A0 **112/125**, so the test is not blind). The credited generations are
runaway boilerplate earning credit purely from a leading option marker:
`'(A).Please answer the following question: I am a movie director...'`,
`'(B)Q: In this task, you are given a sentence...'`. This is the **same family**
as the glued-letter rule behind the 64.5 headline. A1's predictions are also
degenerate — `A` 36/51 while gold on those rows is `C` 15, `D` 12, `B` 11, `A` 9
— so uniform 1/5 is the wrong null; best-constant (always-C) is 15/51 against
observed 17. G2's marginal-preserving permutation test gives p=0.0014, and since
A1-cqa was selected *because* it was the only nonzero cell, the quotable figure is
**Bonferroni x4 = 0.0034**, not the nominal 0.0008.

*Behavioural channel — this one survives, but only on cqa.* **[The "only on cqa"
qualifier is FALSE — see the box above. The regex below shares the scorer's blind
spot: it anchors on an explicit marker at position 0 and so misses
`"The answer is B."`, which is why it reported A1 sciq as 1/200.]** Counting
generations that open with an **explicit** option marker (bracketed, or letter
followed by `)`/`.`/`:` — never a bare prose letter), on a regex that never touches
the scorer, calibrated 10/10 before any count was printed:

| cell | A0 | A1 | A2 | A3 | A4 | A1 vs each latent arm |
|------|---:|---:|---:|---:|---:|---|
| cqa  | 125 | **50** | 0 | 0 | 0 | Fisher one-sided **p = 2.7e-17** |
| sciq | 148 | **1** | 0 | 0 | 0 | **p = 0.50 — not separable** |

**Two corrections to G2's own version of this, from my re-derivation.** (a) The
separation exists on **cqa only**; on sciq A1 is 1/200 and indistinguishable from
the latent arms, so "A1 preserves answer FORM" is a one-cell result, not a
property of the arm. (b) G2's negative control "A3 vs A4 p=0.5" is **degenerate
here** — both are 0/200, so the comparison is 0-vs-0 and could not have come out
any other way. A control on two empty sets demonstrates nothing; the load is
carried by the A0-vs-A4 positive control (p=3.1e-65 sciq, 5.0e-51 cqa).

**Defensible statement.** Restoring the original `recurrent_state` bit-exactly
still destroys accuracy (conv/ffn zeroing is sufficient for that), and on cqa it
leaves more answer-shaped output than any latent write does — so writing
`recurrent_state` carries damage that conv/ffn zeroing alone does not explain.
**Unchanged and central: A2/A3/A4 are mutually indistinguishable in both channels
(0/0/0 explicit markers, 0.00 accuracy in all six cells), so this run cannot
speak to whether latent CONTENT is used at all.** That, not any A1 story, is what
the appendix table can honestly carry.

*G2's instrument failures during this correction, recorded because they are the
recurring pattern:* its first scorer-free regex matched `"a bird"` and
`"an apple"` as option markers, inflating A0's denominator — **a negative control
must include text that looks like a hit**; `"The answer is"` is too easy a case to
prove anything. And for the **third time today** it hit "fires for a different
reason": a broken-probe calibration exited 1 and looked like a pass, but died in
`Path(__file__).resolve().parents[3]` before reaching the check.

## ⚠ A3's three cells share ONE permutation draw (independently re-derived)

`run_five_arms.py:722` draws the derangement as
`build_derangement(ids, args.seed)` — keyed on the seed **only**, not on the task.
`ids` comes from the pinned spec, and in `examples_fixed.json` all three tasks pin
the *same* id list `[0,1,2,...,199]` (**3 of 3 task pairs identical**). Same input,
same seed, deterministic `random.Random(seed).shuffle` ⇒ the same permutation for
every task.

Re-derived two ways, plus controls:

| probe | denominator | result |
|-------|-------------|--------|
| identical id lists across tasks | 3 task pairs | **3/3** |
| identical derangement mapping, seed 42 | 3 task pairs | **3/3** (sciq = cqa = race) |
| emitted `(example_id, donor_id)` pairs, sciq vs cqa | 200 | **200/200 identical** |
| `latent_norm`(A3 row) == `latent_norm`(A4 donor row) | 200 per task | **200/200 both tasks**, 0 mismatch |
| can-distinguish: seed 42 vs 43 mapping | — | different (comparator is not blind) |
| can-distinguish: donor list shifted by 1 | 199 | **0/199** (not matching by accident) |
| negative control: A3 latent vs its OWN id | 200 | **0/200** (derangement is not a no-op) |

**What this means for the appendix.** The consequence is about *reporting*, not
correctness. Each A3 cell is a valid single draw from the space of derangements,
and the arm does what it claims — the last two rows above prove each row receives
the donor's latent and never its own. What the three cells do **not** give is
three independent permutations, so **the spread across A3 cells is not a
confidence interval over derangements.** Report A3 as one permutation measured on
three tasks; do not quote its cell-to-cell scatter as derangement uncertainty.

**The across-seed spread cannot rescue it either,** because `args.seed` feeds both
the permutation and the z draws (`:722` and `:736`), and the two move in opposite
proportions:

| seed pair | recipients keeping the same donor (of 200) | z-seed integers shared (of 200) |
|-----------|--------------------------------------------|---------------------------------|
| 42 vs 43 | 2 | 199 |
| 42 vs 44 | 1 | 198 |
| 43 vs 44 | 1 | 199 |

So changing the seed almost completely redraws the permutation while leaving the
z *multiset* nearly intact (`additive` relabels rather than resamples — see the
`example_seed` docstring at `:299`). Across-seed scatter is therefore dominated by
the permutation, but it is not a clean estimate of either factor alone. Say which
factor an interval refers to, or do not print an interval.

**Also unreported, and worth fixing before any new job:** the z seed is
`example_seed(seed, stream, tasks.index(task), i)` at `:736` — `tasks.index(task)`
is the task's **position in the `--tasks` argument**, not its identity. Running a
cell solo gives `task_index=0`, so 2 of 3 tasks receive *different* z draws when
run per-cell than when run as one three-task invocation (sciq 100045 either way;
cqa 200048 vs 100045; race 300051 vs 100045). Cells launched separately are not
z-comparable with cells launched together. This does not invalidate anything
already emitted — seed 42 `main` was one invocation — but it is a trap for any
top-up run.

### Rule learned here: a fired check is not a correct check

G2's new A3 gate fired on real data and **aborted wrongly**. Its *fact* was true
(200/200 identical donor indices, which I reproduce above); its *implication* was
false, because `example_id` is a per-task label, so identical index sequences do
not mean the two cells share any material. Verified in the correct frame:

| quantity | sciq distinct | cqa distinct | cross-task overlap |
|----------|---------------|--------------|--------------------|
| `prompt_hash` | 200 | 200 | **0** |
| `gold` string | 197 | 190 | 2 (common words: `resistance`, `offspring`) |
| `latent_norm` | 200 | 200 | 3 (float collisions, not shared tensors) |

(self-overlap control = 200, so the set comparator is not blind.)

**Rule: after any fire, do two separate things — re-derive the fact
independently, then ask whether the fact implies the defect you claimed.** A true
fact with a false implication produces a confident, wrong abort. G2 has demoted
its abort to a printed note and added a check in the correct frame (cross-task
prompt hashes). Related disclosure from G2, recorded because it is the same class
of error as my own vacuous probe: its **first positive control for that check was
invalid** — the run aborted at an earlier guard ("invariant 2, arms matched on
prompt hashes") and never reached the new check, so the control proved nothing
about the check it was meant to exercise. A positive control must be shown to
reach the assertion it claims to test.

## ⚠ A STAGE LIST IS NECESSARY BUT NOT SUFFICIENT: "R_C" is underdetermined *within itself*

Recorded from G2. Earlier in this file I showed that "strict" names at least
three different numbers (2.50 / 61.50 / 47.17). G2 then showed the same disease
one level down: even after you fix the stage list, **"plus bare-letter fallback"
leaves the fallback's INPUT and GUARD unspecified.** Its grid over
(`letter_of` on raw | on newline-truncated) x (range guard | none) x (strip
answer-lead | not) gives **4 distinct A0 values spanning 1.83 pts**:

| fallback input | guard | strip-lead | A0 (600 rows) |
|---|---|---|---|
| raw `gen` | range | no | 60.00 |
| raw `gen` | range | yes | 60.17 |
| `truncate_at_newline(gen)` | range | no | 61.67 |
| `truncate_at_newline(gen)` | range | yes | 61.83 |

The range guard is a **no-op** on this data (0 rows out of range). The 10 rows
that move are all shaped `"A\n\nQuestion: ..."`, where `letter_of(raw)` is
`None` (the `(?=[A-Z])` glued-letter rule needs an immediately following capital)
but `letter_of(truncated)` is `'A'`.

⇒ **once a letter fallback is admitted, the fallback's exact input is part of the
definition.** Quoting a stage list does not pin a number.

Measured across all readings so far, the headline gap **A0 − A1 spans 2.33 to
58.00 points**. G2's nesting positive control R_A ≤ R_B ≤ R_C passes 5/5.

**RESOLVED TWICE OVER — and the second resolution found a real defect.**

*First*, G2 asked me to print my fallback's exact input expression because my 61.50
was not in its grid. I diffed row-by-row instead (`/tmp/latentuse/find9th.py`):

```
A0 over 600 rows:  MINE credited=369 (61.50)   G2 trunc+guard=369 (61.50)
DENOMINATOR of disagreeing rows = 0
```

My variant and G2's "trunc+guard" are **the same function on this data** — so the
0.17 was **not in the fallback**. Lesson: when two people disagree by 0.17 pts, do
not trade prose descriptions of the algorithm, **print the denominator of
disagreeing rows.** Zero redirects the search instantly.

*Second*, ⛔ **my 369 was WRONG and G2's 370 is correct.** Diffing the credited
sets as `(task, example_id)` (`/tmp/latentuse/onerow.py`) gave `in G2 but NOT mine
= 1`, `in mine but NOT G2 = 0`, and named the row: **sciq `example_id=20`**.

```
prediction  = 'Answer: (B) chlamydiaQ: Detailed Instructions: In this task, you are given ...'
gold        = 'chlamydia'  (gold_idx=1)
strip_label -> 'chlamydiaQ: Detailed Instructions: In this task, ...'
normalize   -> 'chlamydiaq detailed instructions in this task you ...'
n2.startswith('chlamydia') = True     <- credited by the LABEL-STRIPPED PREFIX stage
```

The cause was **not** the `if not g.strip(): return None` guard at
`g2_aggregate.py:561` (I hypothesised that first; there are **0** blank-after-truncation
A0 sciq rows, so that guard is a no-op here). The cause is that **my `ep()` never
ran the stripped stages at all** — I scored exact/prefix on the raw truncated
string only. With the stripped stages restored, my count is **370 (61.67)**, matching
G2 exactly. I was wrong; G2's number goes in the table.

### ⛔ But sciq/20 exposes a defect in the "strict" definition itself

The direction `normalize(gen).startswith(normalize(choice))` forgives **unbounded
trailing text**. sciq/20's credit rests on a 115-char runaway that never stopped —
so a stage advertised as `2 prefix (either direction)` is doing **containment's job
at position 0**, and containment is explicitly on the REFUSED list.

Measured over A0 and A1, 600 rows each (`/tmp/latentuse/launder.py`):

| task | arm | credited | via exact | via prefix | prefix with >20 chars trailing |
|---|---|---|---|---|---|
| sciq | A0 | 133 | 87 | 46 | 11 |
| cqa | A0 | 74 | 65 | 9 | 2 |
| race | A0 | 76 | 3 | 73 | **29** |
| sciq/cqa | A1 | 0 | 0 | 0 | 0 |
| race | A1 | 1 | 0 | 1 | 0 |

**42 of A0's 283 strict credits (14.8%) rest on "the gold is a prefix of a
generation that is >20 chars longer"**, with trailing garbage up to **139 chars**.
On **race it is 29 of 76 credits = 38%**, and race gets only **3** credits from
exact match. So the race column of any strict table is mostly this stage.

Negative control on the wording: the opposite direction
(`choice.startswith(gen)`, i.e. the model emitted a truncated prefix of the gold)
credits **2** rows — small but non-zero, so "either direction" is not half-inert
and both directions must be stated separately.

⇒ ⭐ **"prefix" is not one criterion either.** The strict/loose boundary was drawn
to exclude containment, but the admitted prefix stage readmits it whenever the
match starts at character 0. A genuinely strict reading needs a **bound on the
trailing remainder** (or `new_tokens`-based invalidation), and that bound is yet
another undeclared parameter. This is the *third* level at which "strict" has
turned out to be a family rather than a criterion — after the stage list, then the
fallback's input, now the prefix stage's tolerance.

⇒ ⭐ **A one-row discrepancy was worth chasing.** The 0.17 pts were immaterial; the
mechanism behind them governs 38% of race's credits. **Chase small discrepancies
for their mechanism, not their size.**

### ⛔ RETRACTED: my prescription to refuse the long-remainder stage. Shape is not precision.

G2 reproduced every count above on 4200 rows (21 files) and then measured what I
never did: **the stage's precision**. I re-derived it with an independent tracer;
the per-cell arithmetic invariant `credited == exact + fwd + rev` holds **21/21
cells**, and chance is **23.33%** (macro mean of 1/n over 4/5/4 choices).

| stage | fired | picked gold | precision | Wilson 95% |
|---|---|---|---|---|
| exact (undisputed) | 608 | 470 | 77.3% | [73.8, 80.5] |
| prefix-fwd rem≤20 | 224 | 151 | 67.4% | [61.0, 73.2] |
| **prefix-fwd rem>20** (the stage I wanted refused) | 235 | 192 | **81.7%** | [76.3, 86.1] |
| **prefix-rev** | 41 | 9 | **22.0%** | [12.0, 36.7] ← spans chance |

**The stage I flagged is the highest-precision stage in the table, above exact
match, with overlapping CIs.** I audited it the way G2 audited its own: on those
235 rows the stage's content reading is correct **192/235 (81.7%)** versus a
label reading's **169/235 (71.9%)**, and **28/235 have no parseable letter at
all** — so it is reading the *answer*, not a mislabelled letter. Per task it is
90.0% / 69.0% / 66.1%, never near chance.

The reason is structural: **2859/3000 main rows hit the 32-token cap (95.3%)** and
**0 forward-direction credits have remainder 0**. Nothing ever stops, so "no
trailing text" is not a strictness dial — it is a **stopping test**, measuring
whether the model emits EOS, which is a different property from whether it
answered. I conflated "looks like containment" with "is guessing".

⇒ ⭐⭐ **Convict a stage on its measured precision, not on the shape of its inputs.**
The admissible audit is per-stage (fired, picked-gold, precision, CI) against
**both** exact match and chance. "Suspicious shape" and "actually guessing" are
different propositions and only the second is measurable. G2 and I each convicted
a stage on shape and were each wrong once — I wanted to cut an 81.7% stage because
its inputs looked like containment; G2 had written a 22.0% stage into its own
definition of "strict" because the name sounded symmetric.

### ⛔ `prefix-rev` is a disguised bare-letter fallback — G2's finding, confirmed

Denominator: 41 rows where reverse-prefix fired. **41/41 have a parseable letter**,
but only **12/41** agree with the index the stage chose. Same 41 texts, two
readings: **as a label 27/41 = 65.9% correct; as content 9/41 = 22.0%** — the label
reading is **3.0x better**. Mechanism: `normalize('(E)') == 'e'`, length 1, so
`'explode'.startswith('e')` credits the choice merely *spelled* with an e.

```
cqa/A0 id=125  gen='(E)' gold_idx=4  rev picked 3  choices=[... 'procrastinat', 'explode', 'have to hold']
cqa/A0 id=179  gen='(C)' gold_idx=2  rev picked 3
cqa/A0 id=188  gen='(E)' gold_idx=4  rev picked 0
```

So it does not merely admit noise — it **misroutes a parseable label to a wrong
choice**, while the printed stage list says bare letters are REFUSED. Refusing the
reverse direction: A1 `0.17 → exactly 0.00`, A0 `47.17 → 47.00`, A2/A3/A4
unchanged, gap `+47.00` both ways; the five-row qualitative claim is untouched.

### ⚠ But G2's bound sweep could never have caught it: `prefix-rev` is algebraically exempt

G2 wrote "the count of credits with remainder exactly 0 is ZERO". **That is true
for the forward direction only (0/459).** The true count is **41, all reverse.**
Remainder is `len(ng) - min(len(choice), len(ng))`; in the reverse direction
`choice.startswith(gen)` means `min()` **is** `len(ng)`, so the remainder is
**identically 0 by construction** — never a measurement.

Direct sweep, rev survivors per bound:

| bound | fwd kept | rev kept |
|---|---|---|
| none | 459 | 41 |
| 100 | 302 | 41 |
| 20 | 224 | 41 |
| 5 | 46 | 41 |
| **0** | **0** | **41** |

**`prefix-rev` survives every bound including 0.** So the `max_remainder` knob G2
swept is structurally incapable of constraining the one stage that is actually at
chance, and its 8-point sweep never touched it. ⇒ ⭐ **Before reporting a knob's
sweep, check which stages the knob can reach.** A parameter that is algebraically
constant for a subset exempts exactly that subset, and a sweep that looks
exhaustive can leave the real defect untouched. The two findings are independent:
the bound governs *fwd*, and only refusing the direction governs *rev*.

### The bound is arm-asymmetric, so it sets the sign of the refeed result

Verified on the refeed directory — share of each arm's credits that are
long-remainder forward prefixes:

| task | A0 | A4 |
|---|---|---|
| sciq | 11/94 = **11.7%** | 113/143 = **79.0%** |
| cqa | 6/133 = 4.5% | 12/74 = 16.2% |
| race | 4/21 = 19.0% | 4/73 = 5.5% |

G2's sweep of A4−A0 strict Macro3: none/200 → **+7.00**, then −10.33, −11.83,
−11.00, −13.00, −13.17, −12.67 — **6 of 8 bounds flip the sign.** ⇒ the
prefix-remainder policy is a knob that **sets the sign of the refeed comparison**.
This strengthens the verdict that refeed is not evidence of latent use, and means
**neither sign may be quoted as a finding.**

## ⛔ A LATENT SCORER DEFECT: `LETTERS` is a `str`, so `in` is a substring test

Found while chasing the above. `g2_frozen_run_five_arms.py:93` is
`LETTERS = "ABCDEFGH"` — a string, not a list/set. `letter_of` at `:249` then does

```python
if len(t) <= 2 and t.upper() in LETTERS:
    return t.upper()
```

`t.upper() in LETTERS` is **substring containment**, so it is true for `""`, for
`"AB"`, `"BC"`, `"CD"`, ... and `letter_of` returns a **non-letter**, which the
caller feeds to `LETTERS.index(...)`:

```
letter_of('')     = ''    -> LETTERS.index('') == 0  -> credited option A
letter_of('   ')  = ''    -> credited option A
letter_of('\n')   = ''    -> credited option A
letter_of('AB')   = 'AB'  -> LETTERS.index('AB') == 0 -> credited option A
letter_of('GH')   = 'GH'  -> credited option G
```

So **an empty or whitespace-only generation is silently credited as option A**,
and a two-letter emission is credited as its first letter.

Instrument calibration (`/tmp/latentuse/substr.py`):
positive controls **9/9 fired**, negative controls (`"A"`, `"B) yes"`, `"(C) x"`,
`"AC"`, `"XY"`, `"hello"`, `"A\n\nQuestion:"`, `"BQuestion:"`) **0/8 false-fired**.

**Status on our data: PRESENT in the code, DORMANT in the measurements.**
Denominator = **3000 rows** (all 15 seed-42 main files); rows where `letter_of`
returns a non-letter = **0**. So no published cell is affected *today*.

⇒ Two separate lessons. (1) **A defect that is dormant is still a defect**: any
future arm or task that emits an empty string gets free credit for option A, and
the 0.00 arms are exactly the ones that emit degenerate output. (2) The reason it
is dormant is luck, not design — so **report it, do not silently "fix" a pinned
runner**, and any new scorer must guard `if not g.strip(): return None` *and* use
`len(t) == 1`.

## ⛔ THE REFEED SIDE RESULT IS A PROTOCOL ARTEFACT IN OUR OWN FAVOUR — do not repeat it as a finding

Recorded from G2 and then **independently re-derived from the output files**
(`/tmp/latentuse/refeed.py`, `refeed2.py`, `erase.py`, `erase2.py`).

Under `prefix_replay=refeed` the sciq A4 cell goes **0.00 → 88.50** (invalid
199/200 → 8/200) while A0 goes **83.00 → 58.00**, so A4 appears to beat A0. It is
not latent use. The mechanism, at `g2_frozen_run_five_arms.py:391-410` and
`:456-465`: refeed prefills the **whole** prompt, `inject_into_cache`
hard-overwrites `recurrent_state` (erasing that prefill), and then sets
`boundary_ids = input_ids` — **the whole prompt is fed a second time**. The
answer is therefore derivable from the re-fed prompt **alone**: the latent is
merely an initial state for an otherwise normal read and is **not load-bearing**.
A0 is lower only because it consumes the prompt **twice** (invalid 19 → 75).
**The inversion is the duplicate-prompt penalty on A0, not latent use.**

Verified token-count invariants (per row, all 200/200):

| condition | invariant | rows |
|---|---|---|
| refeed | `prefill_tokens == boundary_tokens == prefix_replay_tokens` | 200/200 x 4 files |
| none | `boundary_tokens == 1` and `prefix_replay_tokens == 0` | 200/200 x 4 files |
| refeed vs none, matched by `example_id` | `prefill_refeed == prefill_none + 1` | 200/200 x 4 |

⚠ **Two corrections to G2's own write-up of this.**

1. **"both arms record `prefill_tokens=40 boundary_tokens=40 prefix_replay_tokens=40`" is a single row, not the invariant.** `prefill_tokens` takes **34 distinct values, range [32,89]** on sciq main and **[33,90]** on sciq refeed; only 10/200 refeed sciq rows are 40. Quoting a constant where the quantity varies makes a per-row invariant look like a global one — and the *right* statement (the three-way equality above) is both true and stronger. **State invariants as relations between fields, not as the value they happen to take on row 0.**
2. **`cache_seen_tokens` is an INERT field and cannot evidence the mechanism.** G2's trace cited it. Across **2800 rows** (all 15 main + 4 refeed + smoke files) `(cache_seen_tokens_before, cache_seen_tokens_after)` is `(0, 0)` — **1 distinct value**. A live counter could not be constant while `prefill_tokens` ranges over 34 values. It is `getattr(cache, "_seen_tokens", None)` at `:170` on a cache object that does not maintain it. **Before citing a recorded field as evidence, print its distinct-value count; a constant proves nothing.**

### The discriminating arm was never run — ⛔ WRONG, IT WAS, AND IT SETTLES THE QUESTION

**A2 or A3 under refeed** is what would separate "latent content helps" from "the
overwrite erased the duplicated prompt". Seed-42 refeed coverage is **A0/A4 only**
— and narrower than G2 said: `race__A0.jsonl` and `race__A4.jsonl` under
`side_refeed/` are **0 rows**, so G2's refeed is **2 tasks x 2 arms**, not 3 x 2.

⛔ **I then wrote "so that discriminating arm was never run", and that was false.**
G3 has been writing **A3 under refeed** all along, for both seeds and all three
tasks. I missed it because I globbed `latentuse_g*_out` — and **G3's directory is
`latentuse_g3`, with no `_out` suffix**. The glob returned only G2's tree and I
read that empty result as "does not exist" (exactly the failure mode my notes
call *"a timestamped observation is not a property"*: my finder could not see the
directory, which is not the same as the runs not existing). What exposed it was
listing newest files by mtime for an unrelated compliance check.

**A3-vs-A4 under refeed (n=6 per cell, `smoke_{cond}_seed{43,44}`, ids verified
paired, `R_C` and shipped scoring both shown):**

| seed | cond | task | n | A3 shipped | A4 shipped | A3 R_C | A4 R_C | identical generations |
|---|---|---|---|---|---|---|---|---|
| 43 | refeed | sciq | 6 | 5/6 | 5/6 | 6/6 | 6/6 | 6/6 |
| 43 | refeed | cqa | 6 | 3/6 | 3/6 | 3/6 | 3/6 | 5/6 |
| 43 | refeed | race | 6 | 4/6 | 4/6 | 4/6 | 4/6 | 4/6 |
| 44 | refeed | sciq | 6 | 5/6 | 5/6 | 6/6 | 6/6 | 4/6 |
| 44 | refeed | cqa | 6 | 3/6 | 3/6 | 3/6 | 3/6 | 4/6 |
| 44 | refeed | race | 6 | 4/6 | 4/6 | 4/6 | 4/6 | 4/6 |

Pooled refeed: **A3 26/36 (72.2%) vs A4 26/36 (72.2%)** — equal in **6/6 cells**
under **both** scorings. Pooled `none`: A3 0/36, A4 0/36.

**This is the confirmation G2's mechanism argument needed.** Feeding a
*deliberately wrong prompt's* latent costs **exactly nothing** once the prompt is
re-fed ⇒ under refeed the latent is **not load-bearing**, and the sciq
0.00 → 88.50 jump is the re-read, not the state. The refeed side result must not
be reported as evidence of latent use.

**Why this is not a vacuous "no difference":**
- ⚠ Under `none` the same comparison **is** degenerate — A3 and A4 are both 0.00, so 0-vs-0 proves nothing. That is the earlier degenerate control this file already flagged. Under refeed both arms are at ~72%, far from any floor or ceiling, so a gap **could** have appeared.
- Positive control that the arms genuinely differ: `latent_norm` differs A3 vs A4 on **6/6** rows, `donor_id != example_id` on **6/6** rows, and the **generations themselves differ on 9/36** refeed rows (18/36 under `none`). So the inputs and the decode trajectories diverge; only the *scores* coincide.
- ⚠ n=36 pooled is small. The claim this supports is "no detectable latent contribution under refeed", not "provably exactly zero".

⇒ Lesson, and it is the same one twice in one file: **an existence claim is only
as good as the selector that looked.** I asserted "never run" from a glob that
structurally could not match the directory. Before writing "was never run",
print the selector's denominator and show it finds something you know exists.

### The sign of the side result depends on the scorer — ⚠ PARTLY RETRACTED, see below

| task | cond | arm | shipped | R_C | invalid (R_C) |
|---|---|---|---|---|---|
| sciq | none | A0 | 83.00 | 77.50 | 33/200 |
| sciq | none | A4 | 0.00 | 0.00 | 200/200 |
| sciq | refeed | A0 | 58.00 | 41.00 | 113/200 |
| sciq | refeed | A4 | 88.50 | 87.50 | 8/200 |
| cqa | none | A0 | 62.50 | 46.00 | 69/200 |
| cqa | none | A4 | 0.00 | 0.00 | 200/200 |
| cqa | refeed | A0 | 67.50 | 55.50 | 48/200 |
| cqa | refeed | A4 | 60.50 | 60.50 | 1/200 |

⛔ **The `R_C` column above was computed with my BUGGY core** (the one missing the
label-stripped stages — see the sciq/20 section). With the corrected core the cqa
**sign flip disappears**, and the claim I sent G2 was wrong:

| task | A4−A0 refeed, shipped | A4−A0 refeed, R_C (buggy) | A4−A0 refeed, R_C (**corrected**) | flips? |
|---|---|---|---|---|
| sciq | +30.50 | +46.50 | **+40.50** | no |
| cqa | −7.00 | **+5.00** | **−6.00** | **no** |
| race | +19.00 | — | **+44.50** | no |

⇒ ⛔ **RETRACTED: "the sign flips between scorings on cqa."** It does not. Both
scorings give A4 a deficit on cqa (−7.00 shipped, −6.00 corrected `R_C`). My
"+5.00" came from `A0-refeed` being under-credited by 11 points (55.50 → 66.50)
because my core skipped the stripped stages. **A finding of the form "the
conclusion depends on the scorer" is only as good as both scorers** — I had a bug
in the one I built, and it manufactured the very instability I was warning about.

What survives, and is still worth stating: the **magnitude** is highly
scorer-dependent (race +19.00 vs +44.50 — a 25-point spread on the same rows), and
G2's independently-found asymmetry stands: on cqa, **38.8% of A4-refeed's credit
comes from loose+letter stages vs 1.5% for A0-refeed**, so under a strict reading
the cqa deficit widens from −7.00 to −29.50. The cqa comparison flatters A4 twice
over, by protocol and by leniency.

**Coverage correction:** `side_refeed/race__{A0,A4}.jsonl` read 0 rows when I first
looked; G2 explained that writes are buffered until the per-task `finally`, and
they have since landed at **200/200**. So refeed coverage is **3 tasks x 2 arms**
after all, and my "6 of the 10 refeed cells do not exist" was a **read-during-write
artefact**, not a fact about coverage. ⚠ *This is the third time this session that
an absence claim came from looking at the wrong moment or with the wrong selector.*
**A zero-row file in a live tree means "not finished", not "not run".**

Paired McNemar, ids verified identical, self-pair positive control **0 discordant
on 12/12 cells** (so the test can fail). Like-for-like (A0/none vs A4/refeed, each
reading the prompt effectively once), corrected scoring:

| task | shipped delta | R_C delta | discordant | chi2_cc (R_C) |
|---|---|---|---|---|
| sciq | +5.50 | +9.50 | 39 | 8.31 |
| cqa | −2.00 | +14.50 | 71 | 11.04 |
| race | **+0.00** | +3.00 | 32 | 0.78 |

On **race the like-for-like delta is exactly 0.00 shipped** (68.00 vs 68.00,
discordant 20, chi2_cc 0.05) — i.e. once you stop letting A0 read the prompt twice,
the headline refeed advantage vanishes on that task. This is the cleanest single
number against reporting the refeed result as latent use.

### ⚠ I published a wrong verdict from a confounded fingerprint, and the coincidence that saved me

I built an erasure test: A2 writes `R(mean_train_latent)`, **the same state for
every example**, so if the write fully erases the prefill then
`cache_fp_after_sha256` must be **constant** across 200 rows while
`cache_fp_before_sha256` varies. It printed `distinct fp_AFTER = 34` and my
script duly printed **"PREFILL SURVIVES THE WRITE"** for all three tasks.

That verdict was **wrong**. `34` is exactly sciq's count of distinct
`prefill_tokens` — the coincidence is what made me look. `cache_fingerprint`
at `:165` writes `L{l}._seen_tokens` **into the hashed dict**, so `fp_after`
varies with prompt **length** even when every written tensor is identical.

Re-run separating content from counters:

| task | distinct fp_after | distinct prefill_tokens | length→fp a function | fp→length a function |
|---|---|---|---|---|
| sciq | 34 | 34 | True | True |
| cqa | 30 | 30 | True | True |
| race | 142 | 142 | True | True |

A **bijection with length alone** ⇒ the written content is constant ⇒ **erasure
IS complete**. Cross-task strengthening: denominator = **23** `prefill_tokens`
values shared by sciq and cqa; lengths whose A2 `fp_after` is not unique across
tasks = **0**. Negative control on the same 23 lengths with A4 (which writes a
different state per example): **23/23** have >1 distinct `fp_after`, so the test
discriminates rather than passing vacuously.

⇒ Lessons. (1) **A hash is only as specific as the fields inside it** — a
fingerprint that mixes content with counters cannot answer "did the content
change?", and I asserted it could. (2) **A tally is not a mechanism**: `34` had
to be explained, not reported. (3) The rescue was **checking whether the number
equalled some other quantity in the file**; that habit turned a false finding
into a proven one within one run.

⚠ Also: the first version of the refeed audit **raised** `KeyError` on
`adapter_called` / `conv_ffn_zeroed_by_adapter`, because those live nested inside
`cache_policy`, not at top level. The raise (rather than a skip) is what caught
it — had I used `.get(..., default)` I would have silently reported
`adapter_called=False` for every arm, including A4. Then `cache_seen_tokens_*`
**also** turned out to be nested. **Two nesting errors in one script**, both
caught only because unknown keys raise.

## ⛔⛔ "MEETS OR EXCEEDS A0" IS UNDERDETERMINED: the ceiling choice swings the effect 12x

A third agent (G3) began building an interpretation — *"A3/A4 refeed meets or
exceeds A0"* — on top of the refeed set, on the stated premise that **"there is no
seed-42 refeed counterpart anywhere; `latentuse_g2_out` has only
`prefix_replay=none` on all 3000 rows."**

**The premise is false.** Selector: every jsonl under `latentuse_g2_out`, reading
the `prefix_replay` **field** rather than the path — 33 files, **4272 rows**,
values `{'none': 3072, 'refeed': 1200}`. The refeed set is
`seed42/side_refeed/{sciq,commonsenseqa,race}__{A0,A4}.jsonl`, 200 rows each, and
it is **fully paired**: A0 and A4 id lists are identical to each other and to the
main id lists on all three tasks. No new job was needed. (A3 is **not** there —
seed-42 refeed is A0/A4 only; A3-under-refeed exists only in G3's own
`smoke_refeed_seed{43,44}`.)

⚠ *That is the fourth absence claim this session to come from a bad selector — and
the third variant of the same mistake: G2's "arm never run", my `latentuse_g*_out`
glob, my "0-row race files", now G3's "only none".* **An absence claim is a claim
about your selector until you show the selector can find something you know
exists.**

### The substantive trap: which A0 is the ceiling?

`A0/refeed` is **depressed by a protocol penalty**, not by lack of knowledge —
under refeed, A0 consumes the prompt **twice**, and the damage is an invalid-rate
effect:

| task | A0/none | A0/refeed | delta | invalid none → refeed |
|---|---|---|---|---|
| sciq | 83.00 | 58.00 | **−25.00** | 19/200 → **75/200** |
| cqa | 62.50 | 67.50 | +5.00 | 22/200 → 11/200 |
| race | 68.00 | 49.00 | **−19.00** | 3/200 → 9/200 |

G3's claim evaluated against **both** ceilings — same arm, same rows, paired,
shipped scoring:

| ceiling | sciq | cqa | race | Macro3 delta |
|---|---|---|---|---|
| vs `A0/refeed` (depressed) | +30.50 exceeds | −7.00 below | +19.00 exceeds | **+14.17** |
| vs `A0/none` (honest) | +5.50 exceeds | −2.00 below | **+0.00 TIE** | **+1.17** |

**Race flips from "exceeds by 19" to an exact tie** (68.00 vs 68.00, discordant 20,
chi2_cc 0.05) and the macro effect shrinks **12x**. ⇒ ⭐⭐ **A conclusion that
depends on which baseline you call "A0" is not a finding.** And the depressed
ceiling is the wrong choice for a specific reason: the depression is a **penalty
handed to the baseline by the protocol**, not a property of the model. Grading an
arm against a handicapped baseline manufactures the effect.

⇒ ⭐ **"Beats baseline" needs the baseline's *condition* in the sentence.** "A4
exceeds A0" is true at +14.17 and true at +1.17 and false on race — three different
claims from one phrase, exactly like "strict" naming three numbers.

Note this is *also* why the refeed result must not enter `tab:latent-use-pending`
even as a favourable side note: combined with the sign being a scorer knob
(+7.00 → −13.17 across the remainder sweep) and **A3 == A4 in 6/6 cells**, there
are now three independent reasons the refeed comparison cannot carry a latent-use
claim.

## Status at the time of writing (2026-09-23)

| group | scope | status |
|-------|-------|--------|
| G1 | step 1, the training-mean latent | **done**: 4096/4096 spans, train split, z-space mean at `g1_mean_latent.pt`; surfaced the latent-collapse finding |
| G2 | five arms, seed 42 | in STEP 4 with all five arms (A2 unblocked by G1); ~2.1 s/decode, 3000 decodes ~105 min; produced the A0 falsification test |
| G3 | seeds 43 and 44 | attempt 3 running BOTH prefix_replay modes; uses a pinned runner, not the live file |

**Do not delete or overwrite `g2_frozen_run_five_arms.py` or
`g3_pinned_run_five_arms.py`.** Verified byte-identical, both sha256
`31beb2a49952fc78a9a861cc58713bffb4375a23138b6aa1ecba3c145c6fbf43`. G3's job
hard-aborts if that hash moves. The live `run_five_arms.py` changed four times in
30 minutes and one mid-write read killed a seed-44 run, so the pinned copy is what
produced seeds 42/43/44 — all three are mutually comparable *because* they share
the pre-fix scorer. Scoring happens after decode (greedy argmax, no RNG), so the
scorer version changes no generation, only the numbers derived from them; G3
reports as-run and corrected side by side with per-cell flip counts.

The A0 falsification test is **already satisfied and needs no new job**: it lives
in G2's seed-42 `none` smoke. Under mode=none, sciq: A0 5/6 correct with 6/6
distinct generations; A1 0/6, 5 distinct; A3 0/6, 3 distinct; A4 0/6, 2 distinct
across six different questions. The monotone gradient 6 > 5 > 3 > 2 in *distinct*
generations is itself informative: degeneracy tracks how much latent content is
written, not whether the harness works.

G2 chose **not** to re-freeze its runner mid-run to pick up the `letter_of` fix,
which is the right call: two different scorers inside one table would be worse
than one known-exposure scorer. Its frozen copy
(`g2_frozen_run_five_arms.py`, `letter_of` at line 244 still carrying
`len(t) <= 2 and t.upper() in LETTERS`) adds an independent "G2 PATCH 1"
early-return that makes empty generations invalid, so the two scorers now agree on
the empty-string hole. Measured exposure is **0/144 (re-measured; counts grow as
jobs write rows) with a comparator verified
able to separate them** (see "Scorer exposure" above) — but that 0 covers smoke
rows only, 4 of 5 arms (A2 has none), and `prefix_replay=none` only. Re-run the
differential over the full output before reporting, and rescore from the stored
`prediction` field rather than re-decoding.

Constraints on filling the table:

* No cell may be filled from a `none`-only or `refeed`-only run without naming
  which setting it came from; the two are not comparable.
* The `refeed` column at seed 42 covers A0/A4 only. A1/A2/A3 stay pending there
  until a five-arm `refeed` job is authorised and run.
* **No A2 row exists anywhere yet.** Every `latentuse_g2_out/seed42/main/*.jsonl`
  is empty at the time of writing; A2's mean latent is loaded and live
  (`latent_norm` 9.1786, matching G1's provenance, so not a zero vector) but has
  produced no scored rows.
* Cross-seed CIs must carry the `additive`-stream caveat: ~597/600 shared
  per-example DDIM seeds means 42/43/44 are not independent replicates.
* If A2/A3/A4 land near-identical, report the latent-collapse statistics in the
  same breath. Near-identical rows there are the expected consequence of a
  collapsed encoder, and presenting them as evidence about the write mechanism
  would be wrong in either direction.

---

## ⛔⛔ RETRACTED: "A2/A3/A4 are exactly 0.00 in every `prefix_replay=none` cell"

That sentence is the one I shipped in every handback. It is **true at seed 42
(9/9 cells) and FALSE at seed 43**, and I found the counterexample only because
G3's `full_none_seed43` reached n=200 and I re-ran the check against a tree that
had grown since I first counted it.

`latentuse_g3/full_none_seed43/commonsenseqa__A3.jsonl` scores **2/200 = 1.00**,
not 0.00. Denominator printed first: n=200, invalid=195, valid=5, correct=2.
All 15 other `none` cells (9 at seed 42, 5 at seed 43) are exactly 0.

Both credits are on text with no relation to the question:

| id | gold | prediction (truncated) | mechanism |
|---|---|---|---|
| 111 | `house` | `...deal done<?>The White House has been working with Congress...` | `house` is a real standalone word in "White House" |
| 159 | `ring` | `...it would be hi**ring** more than 1,000 people...` | `ring` is glued inside `hiring` |

⇒ **Correct wording is "at or statistically indistinguishable from zero", never
"exactly 0.00".** 2/200 with a 20%-ish chance floor is not evidence of latent
use, and the conclusion does not change — but the absolute phrasing was false,
and a reviewer checking one cell would have caught it.

### ⭐ The general lesson, third instance this session

The claim was true when written. **The tree grew underneath it.** This is the
same mechanism as the memory-file note about `dit-fixed`: *a verified absence is
an observation with a timestamp, not a property.* A claim quantified over "every
cell" must be re-run against a re-enumerated cell list, not re-asserted, every
time new cells can appear. I re-enumerated by reading the `prefix_replay`
**field** on every file in the tree; the cell count went 9 → 15.

## ⛔ A SECOND SCORER DEFECT: containment has no word boundary

`g2_frozen_run_five_arms.py:299`

```python
    for i, c in enumerate(norms):
        if c and c in ng and len(c) > best_len:
```

`c in ng` is a bare substring test, so a gold of `ring` is credited by `hiring`,
`house` by `greenhouses`, `car` by `cart`. This is a **distinct** defect from the
`LETTERS`-is-a-`str` bug at `:93` and from `prefix-rev` at `:291`; it is the
third independent way this scorer credits a row it should not.

### ⚠ My first sweep of it was WRONG, and the error was in my instrument

I first reported **340/1840 credits (18.48%)** as glue artefacts. That number
**conflates two classes** and must not be used:

| class | n | % of 1840 | is the credit wrong? |
|---|---|---|---|
| exact | 14 | 0.76% | no |
| clean containment | 1076 | 58.48% | no |
| **right-glue only** | **335** | **18.21%** | **NO — gold fully emitted, then a decode artefact** |
| **left-glue** | **32** | **1.74%** | **YES — gold starts mid-word** |
| other (stripped/F1/letter stages) | 383 | 20.82% | separate question |

The 335 are rows like race id=6, gold `"both feet on the ground"`, generated
`...both feet on the ground` + `q 请将下文翻译成中文简体`. The model **produced the
entire gold answer**; a decode artefact is fused to its right edge. Refusing
those would delete correct answers. Only the **left** edge matters: if the
character before the match is alphanumeric, the gold began mid-word and the
credit is spurious.

**Corrected figure: 32/1840 = 1.74% spurious**, concentrated in cqa and sciq
A3/A4 refeed cells (5-6 each).

### ⭐⭐ My positive control FAILED, and that is the most useful thing here

```
POS ctrl (gold only inside 'Whitehouse'-style glue): flagged=False -> must be True: FAIL
```

I injected `"the White House said"` as a *broken* input. It is not broken:
`normalize_text` keeps the space, so `house` **is** a standalone word there. The
**control was wrong, not the detector** — I wrote an injection that does not
exhibit the defect I claimed to be injecting, which is the *vacuous injection*
failure from `inert-validation-checks.md` ("a vacuous injection will make me
blame the detector"). Rewritten with genuinely glued strings (`hiring`,
`greenhouses`, `cart`) plus two must-not-fire cases: **5/5**, then a 6-case
three-way classifier at **6/6**.

⇒ **A control that fails is not automatically a broken detector — check which of
the two is wrong before touching the detector.** Had I "fixed" the detector to
satisfy that control, I would have built the word-boundary bug back in.

## The A3-vs-A4 refeed control at n=200: signs DISAGREE across tasks

Paired on `example_id`, ids identical 200/200, raising on mismatch rather than
intersecting. Negative control (self-pair) 0 discordant; positive control
(A0/none vs A4/none) 166 discordant, p=2e-50.

| task | scorer | A3 | A4 | delta | discordant | sign p |
|---|---|---|---|---|---|---|
| sciq | shipped | 87.00 | 88.50 | **+1.50** | 3 | 0.2500 |
| sciq | glue-removed | 84.50 | 86.00 | +1.50 | 3 | 0.2500 |
| cqa | shipped | 62.00 | 60.50 | **−1.50** | 5 | 0.3750 |
| cqa | glue-removed | 59.50 | 57.50 | −2.00 | 6 | 0.2188 |

Non-vacuity, both tasks: `latent_norm` differs 200/200, `donor_id != example_id`
200/200 with **0 fixed points** and donors a true permutation, generations differ
82/200 (sciq) and 95/200 (cqa).

⇒ **The sign is not consistent across the two completed refeed tasks**, and
neither is significant. The latent is causally *active* (41% of trajectories
diverge) and *not load-bearing for the metric* — which is G3's conclusion, now
with the added fact that cqa points the other way, so "A4 ≥ A3" cannot be
asserted either. race is still in flight (0 rows); it is the tie-breaker and
should be reported whatever it shows.

## ⚠ MY OWN COMPLIANCE NUMBER NEEDED ITS WINDOW STATED

I have been reporting "**0** measurement files modified" every handback. Re-run
today with an explicit window, the same command returns **109** for
`-newermt '2026-09-22'` and **0** for `-newermt '2026-09-23'`.

Both are true. The 109 are all dated **2026-09-22** and belong to three sibling
tasks that own those directories (`results/benchmarks/bosfix/*` from the gla-decode
and BOS-fix jobs, `results/benchmarks/efficiency_total`, `results/paper_ledger`).
`grep -c latentuse` over that list = **0**, so none is mine, and the max mtime
over the whole set is 09-22 — before today.

⇒ **"0 files modified" is a (predicate, window, root) triple, not a fact.** I
caught G2, G3 and myself on exactly this for row counts and absence claims, then
shipped a bare "0" four times without naming the window. Controls used here:
finder positive control 1919 files visible; `-newermt` positive control confirmed
against files I did write today (30 under `scripts/eval/latentuse`, 70 under
`latentuse_g3`) so the timestamp predicate is not inert.

Standing state re-verified today: pinned runners `35dc33a3…` /
`31beb2a4…` and `597de5116…` unchanged; `appendix.tex` `0b91c817…`,
`main.tex` `d9ca82a4…`; `score_gen_match.py` not modified; D4 exit 0,
TP=34/34, FP=0/9.

## ⛔⛔ VACUOUS BY TYPE: `donor_id != example_id` was never measuring a derangement

G2 found this and it reproduces exactly. In `latentuse_g2_out/seed42/side_refeed/`
(6 files x 200 rows, A0 and A4 only) **`donor_id` is `NoneType` in all 1200 rows**
while `example_id` is `int`. Same-type rows: **0/1200**. So
`donor_id != example_id` is `None != <int>`, which **cannot be False for any
input**. It passed 200/200 per file and carried zero information.

⇒ This is a **new entry for the inert-check catalogue**: not "always true because
the data happens to satisfy it" but **always true because the two operands can
never be the same type**. A type mismatch makes `!=` a tautology and `==` a
contradiction, and both read as a clean tally.

### ⚠ The mirror image, which G2 did not name: `fixed points 0/200` is vacuous too

Measured: `full_refeed_seed43/{sciq,commonsenseqa}__A4.jsonl` have
`donor_id` = `NoneType` 200/200 and **`fixed-pts = 0`**. That zero is
`None == <int>` — a contradiction, not a measurement. **Any "0 fixed points"
computed on an A4 file is vacuous**, and 0 is exactly the value you want to see,
so the vacuity is invisible. Only the A3 side is type-informative
(`donor_id` int 200/200, fixed points genuinely 0, donor multiset == example_id
multiset).

### My own exposure, checked honestly

My reported `donor!=id 200/200` iterated over **the A3 element of each pair**, so
it was computed int-vs-int and the number is sound. But the wording was wrong:
it is only informative **for A3**, and A4 carries `donor_id=None` by design.
Correct phrasing: **"A3 is a true derangement"**, never "the pair differs".

⇒ ⭐ **Before citing any tally as non-vacuity evidence, print the TYPES of both
operands.** A predicate over a field that is `None` on one arm is a tautology on
that arm.

### Replacement evidence, verified with a negative control

| task | latent_norm multiset A3 == A4 | per-row differs | NEG CTRL (perturb 1 value by 1e-3) |
|---|---|---|---|
| sciq | True | 200/200 | multiset equal = **False** ⇒ check CAN fail: PASS |
| cqa | True | 200/200 | multiset equal = **False** ⇒ check CAN fail: PASS |

Same set of latents, delivered to the wrong examples. This is the right control
and it is type-safe (float vs float). Distinct values 200/200 (sciq), 197/200 (cqa).

## The pooled A3-vs-A4 null, n=400, confirmed — but do not report the pool alone

| stratum | A3 | A4 | (A4−A3) | A4-only | A3-only | sign p |
|---|---|---|---|---|---|---|
| sciq | 87.00 | 88.50 | **+1.50** | 3 | 0 | 0.2500 |
| cqa | 62.00 | 60.50 | **−1.50** | 1 | 4 | 0.3750 |
| **pooled n=400** | — | — | **+0.00** | 4 | 4 | **1.0000** |

My paired bootstrap 95% CI on the pool: **[−1.50, +1.25]** (G2 reported
[−1.39, +1.39]; same conclusion, different resampling scheme).

⚠ **The pooled +0.00 is an exact cancellation of +1.50 and −1.50.** Both strata
are independently null, so the null is robust — but reporting only the pooled
number would hide that the per-task signs oppose, which is the more informative
fact. **Report all three rows.** race is still 0 rows and is the tie-breaker.

### ⛔ A sign-convention error in G2's message (magnitudes and p-values all correct)

G2 sent "sciq 87.00 vs 88.50 (delta **−1.50**, McNemar 0/3)" and "cqa 62.00 vs
60.50 (delta **+1.50**, McNemar 4/1)". Those are **internally inconsistent with
its own McNemar counts**: `(b_only − c_only)/n` reproduces the accuracy delta
exactly, giving sciq **+1.50** and cqa **−1.50**. G2 used `(A3−A4)` in the delta
column and `A4`-favouring counts in McNemar. The pooled `+0.00` is immune
(cancellation is sign-convention-free) and every magnitude and p-value it quoted
is right — but the two signs are swapped, which **inverts which task favours the
correct latent**. Any sentence naming a task must be rechecked against raw
accuracies, not against the delta column.

⇒ ⭐ **A delta and its discordant counts are two computations of one quantity;
make them agree or one of them is mislabelled.** `(b−c)/n == acc_B − acc_A` is a
free consistency check on every paired comparison, and it caught this in one line.

## ⛔⛔ `--letter_first` IS OFF BY DEFAULT AND UNRECORDED — my earlier glue figures were computed with it ON

Chasing G3's sizing of the `:299` defect I reimplemented `score_row` and ran the
negative control I have been demanding of everyone else: **with the clause
unchanged, does it reproduce the as-scored `correct` field?**

**It did not — 40/6200 mismatches across 7 cells.** So my reimplementation was
not the shipped scorer and no delta from it was trustworthy.

First guess (gold resolved via `match_choice` before exact match) was **wrong**:
patching it left the count at exactly 40. Rather than guess a third time I
**printed the disagreeing rows**, and the cause was immediate — every one is a
bare letter (`(E)`, `A:`, `(C)`) with `strip_label(g)` length 0:

```
commonsenseqa__A0 id=125  as-scored correct=False  mine=True
   gen='(E)'  letter_of='E'  as-scored pred=3  mine=4
race__A1 id=45   as-scored correct=True   mine=False
   gen='A:'   letter_of='A'  as-scored pred=2  mine=0
```

`:125` defines `--letter_first` as `action="store_true"`, i.e. **default False**,
and the published runs never passed it. I had hardcoded `True`. With
`letter_first=False`: **0/6200 mismatches, all 31 cells.**

⇒ ⭐⭐ **`letter_first` is not recorded in any `run_meta`/`summary.json`**
(`grep -i letter` over all of them: no key). It changes 40 rows including 5 in
`race__A1`, an arm whose whole purpose is to be a near-zero control, and it is
**not recoverable from the outputs** — only from the absence of a CLI flag. Any
rescore by anyone who assumes `letter_first=True` will silently differ. This
belongs in the provenance block.

⇒ ⭐ **"Print the disagreeing rows" beat two rounds of guessing.** I told G3 to do
this and then guessed twice myself before doing it.

## The `:299` word-boundary defect, sized correctly at last

Method (G3's, which is better than my first one): monkeypatch **only** the
containment clause, `c in ng` → `re.search(r"(?<![a-z0-9])"+escape(c)+r"(?![a-z0-9])", ng)`,
leaving the whole ladder intact. Both controls printed before any delta:

* NEG CTRL — reproduce as-scored with the clause unchanged: **0 mismatches / 31 cells / 6200 rows: PASS**
* POS CTRL — `"hiring more people"` vs `['ring','bell']`: `wb=False → 0`, `wb=True → None`: **PASS, clause is live**

I reproduce G3's five cells exactly (sciq A3 87.00→86.00, A4 88.50→88.00; cqa A3
62.00→61.50, A4 60.50→60.50; none/cqa/A3 1.00→0.50) and its conclusion that the
**A4−A3 sign is stable**: sciq +1.50→+2.00, cqa −1.50→−1.00.

### ⛔ But "max 1.00 pp" is an artefact of auditing only A3/A4. 11 of 31 cells move, up to −10.00 pp

| cell | as-scored → wb | delta |
|---|---|---|
| side_refeed/race__A0 | 49.00 → 39.00 | **−10.00** |
| main/race__A0 | 68.00 → 63.00 | **−5.00** |
| main/race__A1 | 7.50 → 3.00 | **−4.50** |
| side_refeed/race__A4 | 68.00 → 65.50 | −2.50 |
| side_refeed/sciq__A4 | 88.50 → 87.50 | −1.00 |
| (6 more ≤1.00) | | |

**Published five-row table under the word-boundary fix:**

| arm | sciq | cqa | race | Macro3 |
|---|---|---|---|---|
| A0 | 83.00→83.00 | 62.50→62.00 | **68.00→63.00** | **71.17 → 69.33 (−1.83)** |
| A1 | 0.00 | 8.50 | **7.50→3.00** | **5.33 → 3.83 (−1.50)** |
| A2/A3/A4 | 0.00 | 0.00 | 0.00 | 0.00 (+0.00) |

⇒ ⭐⭐ **The defect moves the A0 ceiling and the A1 control by more than it moves
any A3/A4 cell.** A 5-cell audit scoped to the arms under test concludes
"max 1.00 pp, does not touch the verdict" and is right about the verdict while
missing that **two of the five published rows change**. The zero rows are
algebraically exempt (0 credits ⇒ no containment credits to lose), so the arms
that *can* move are exactly the ones a latent-focused audit excludes.

⇒ **The audit set must be the set of cells you will PUBLISH, not the set of cells
your hypothesis is about.**

## ⛔ RETRACTED: my "sign error" charge against G2. Its numbers were self-consistent

I charged G2 with swapping two delta signs. **I was wrong.** Its convention was
stated in its message — `delta = A3 − A4`, `b = "A3 correct & A4 wrong"` — and
under that single convention the identity I proposed as the guard *confirms* it:

| task | acc_A3 | acc_A4 | (A3−A4) | b (A3-only) | c (A4-only) | (b−c)/n | identity |
|---|---|---|---|---|---|---|---|
| sciq | 87.00 | 88.50 | **−1.50** | 0 | 3 | **−1.50** | HOLDS |
| cqa | 62.00 | 60.50 | **+1.50** | 4 | 1 | **+1.50** | HOLDS |

I read its McNemar pair as `(A4-only / A3-only)` while keeping its delta, i.e.
**I mixed its convention with mine and called the result a contradiction.**

⇒ ⭐⭐ **The consistency identity `(b−c)/n == acc_A − acc_B` only detects an error
if you evaluate it in ONE convention.** Applied across two conventions it
manufactures a false positive — which is exactly what it did to me. **A delta is
not a number, it is a number plus a subtraction order**, and the order must be
printed next to it.

The convention-free statement, which both of us agree on and which is what prose
should say: **sciq favours A4 (the correct latent); cqa favours A3 (the deranged
donor); neither is significant; pooled over the two completed tasks is an exact
0.00 with 4/4 discordant, p=1.0000.**

## ⛔ RETRACTED AGAIN: my left-edge-only glue rule, and my 32 and 198 both wrong

G2 was right on both counts and I reproduce its number exactly now.

**(a) Left-edge-only is wrong.** `car` inside `cart` is *right*-edge glue and is
genuinely spurious; my rule scored it clean. The correct criterion is
**direction-free**: a credit is spurious iff gold occurs in the text and **no
occurrence has a non-alphanumeric neighbour on both sides**. One standalone
occurrence legitimises the credit however many glued copies exist.

**(b) Boundaries must be judged on RAW offsets, not normalised text.**
`normalize_text` at `:233` does `re.sub(r"[^\w\s]", "", t)`, so `ground"Q`
becomes `groundq` and a quote neighbour is indistinguishable from a glued letter.
I built an index-carrying reimplementation and validated it against the real
`normalize_text` on **39407 strings, 0 mismatches** before using it.

### But my first direction-free number was 198/1840 (10.76%), vs G2's 1 in main

A 200x gap means a bug, so I printed my flagged rows instead of defending the
number. **The bug was mine:**

```
race__A0 id=10  gold='positive'    raw span='positive'  before=' '  after='Q'
race__A0 id=15  gold='doubtful'    raw span='doubtful'  before=START after='U'
```

`Q` and `U` are the first letters of `Question:` and `User:` — the decode
artefact fused with **no separator**. I only truncated at newline, so the
artefact's first letter masqueraded as a glued morpheme. Distribution of what
follows the span in my flagged rows: `Question:` and `user` dominate.

**Fix: cut the generation at the artefact, then apply the direction-free rule.**
Controls 7/7 (including `'positive Question: next'` and `'doubtfulUser: hi'`
which must NOT flag). Result: **10/1840 (0.54%)** tree-wide, and G2's published
rows become

| arm | Macro3 as-scored → spurious-refused |
|---|---|
| A0 | 71.17 → **71.00 (−0.17)** |
| A1 | 5.33 → 5.33 (+0.00) |
| A2/A3/A4 | 0.00 → 0.00 (+0.00) |

**exactly G2's −0.17 and its single row (cqa id=163, gold `pet store`, generated
"...kept in cages in pet stores").** Independent agreement to the row.

⇒ ⭐⭐ **Three wrong numbers for one defect: 340 (conflated classes), 32
(left-edge only), 198 (artefact letter read as glue) — before 10.** Every one had
passing controls, because each control set tested the shape I had thought of.
⇒ ⭐ **When two independent implementations differ by 200x, print the disagreeing
rows; do not adjudicate by argument.**

### ⚠ One control that "failed" was a wrong control, again

`'bicycles and cars'` / gold `car` flagged spurious; I had wanted `False`. My
control encoded a **morphology** judgement (a plural is the same answer) that the
criterion cannot make — `car`/`cars` and `car`/`cart` are both one glued letter.
The criterion behaves as specified; the control asserted more than the spec.
Worth stating explicitly: **this criterion also refuses inflected forms, so it
over-refuses. It can only lower credits, never manufacture a win** — the safe
direction for a defect estimate.

## ⚠ The `:299` vs `:298` line number

G2 cites the containment clause as `:298`, I cite `:299`. Both refer to the same
`if c and c in ng and len(c) > best_len:` — the `for` is at `:298`, the `if` at
`:299`. Quote the code, not the number.

## The same-arm/different-seed noise floor: the idea is right, the Fisher test is not

G3 proposes a scale for "not load-bearing": compare the deliverable swap
(A3 vs A4, stranger's latent) against a **reseed floor** (A4@seed42 vs A4@seed43,
same arm, same latent recipe, different draw). Reproduced exactly:

| comparison | sciq gens differ | sciq flips | cqa gens differ | cqa flips | pooled flips |
|---|---|---|---|---|---|
| FLOOR (reseed) | 67/200 | 2 | 88/200 | 6 | **8/400** |
| SWAP (stranger) | 83/200 | 3 | 98/200 | 5 | **8/400** |

**Provenance is clean** — both runs share `checkpoint_sha256 55bc5c7b…`,
`git_commit ef3a7cd0…`, `sampler ddim_cfg(steps=100,cfg=3.0)`,
`adapter inject_into_cache`, `prefix_replay=refeed`. So the floor is not
contaminated by code or checkpoint drift, which was my first worry.

**The shared-seed worry also does not apply.** Memory records ~597/600 shared
per-example DDIM draws across seeds 42/43/44, which would have made the floor
vacuous (same latent ⇒ guaranteed near-zero discordance). Measured:
`latent_norm` identical **0/200 in both arms**, so both are genuine
perturbations. G3's floor survives the test I built to kill it.

⇒ The substantive claim stands and is stronger than a p-value: **a stranger's
latent perturbs the trajectory about as often as merely re-drawing the same
latent (83 vs 67, 98 vs 88) and flips accuracy exactly as rarely (8/400 both).**

### ⛔ But `Fisher exact on [[8,392],[8,392]] → p=1.0000, OR=1.000` is not evidence

Two independent defects:

1. **The tables are not independent.** `floor = A4@42 vs A4@43` and
   `swap = A3@43 vs A4@43` **share the A4@43 arm on the same 200 example_ids**.
   Fisher assumes two independent samples; these are two dependent tallies that
   happen to coincide.
2. ⭐⭐ **The statistic is powerless by construction.** Any two *equal* counts
   give `p=1.0000, OR=1.000` regardless of magnitude:

   | table | result |
   |---|---|
   | `[[1,399],[1,399]]` | p=1.0000, OR=1.000 |
   | `[[8,392],[8,392]]` | p=1.0000, OR=1.000 |
   | `[[200,200],[200,200]]` | p=1.0000, OR=1.000 — **a 50% flip rate** |

   The test returns the same answer for a 0.25% and a 50% flip rate, so it
   **cannot distinguish "both tiny" from "both huge"**. The evidence is the
   *size* of the discordance (8/400 = 2.0%), not the p-value.

⇒ ⭐ **A test whose output is fixed by the equality of its inputs is not testing
equality — it is restating it.** Same family as the `None != int` tautology and
the `len(set)==1` never-passes check: the arithmetic is fine and the
inference is empty. **Ask what value the statistic takes when the hypothesis is
false; if it cannot change, it is not a test.**

### The defensible form

| stratum | (A4−A3) | paired bootstrap 95% CI |
|---|---|---|
| sciq | +1.50 | [+0.00, +3.50] |
| cqa | −1.50 | [−4.00, +0.50] |
| pooled n=400 | **+0.00** | **[−1.50, +1.50]** |

State it as **"the swap effect is bounded within ±1.5 pp; no detectable effect at
n=200/task"** — never "proven identical". G3's own power caveat (effects above
~2 pp excluded, sub-1 pp not) is correct and must travel with the number.

## ✅ Two G3 findings verified clean

**(a) `missing=1059 unexpected=0` is benign.** `missing` is measured against
`ckpt["trainable_state"]` (`provenance.py:104`), which holds only the trainable
tensors; the 1059 are the frozen RWKV-7 backbone loaded separately, and
`unexpected=0` means nothing in the checkpoint was silently discarded. This is
**not** the `global-birwkv` 0.5% failure mode (`strict=False` swallowing 473
weights) that `scripts/eval/relay_utils.py` exists to catch.

**(b) A3 really writes the donor's latent** — verified independently by exact
equality, not tolerance: **`A3.latent_norm == donor's latent 200/200` and
`== own latent 0/200`**, both tasks.

⇒ ⭐⭐ **G3's tolerance point generalises and is important.** Over all 5400
non-null `latent_norm` values the **entire range is 0.0738** (9.1681–9.2419,
1161 distinct). A `tol=2e-2` identity test spans **27% of the whole range** and
`5e-2` spans 68% — vacuous. This is the encoder collapse the memory file records,
and its consequence is that **every tolerance-based identity check on this latent
is near-vacuous; only exact comparison works.** Any future check of the form
"is this the same latent?" must use exact equality here.

## G3's race cell landed: all three refeed strata at n=200 (verified)

| task | A3 | A4 | Δ(A4−A3) | A4only | A3only | disc | invalid A3/A4 |
|---|---|---|---|---|---|---|---|
| sciq | 87.00 | 88.50 | +1.50 | 3 | 0 | 3 | 1/1 |
| commonsenseqa | 62.00 | 60.50 | −1.50 | 1 | 4 | 5 | 0/1 |
| race | 68.00 | 68.50 | **+0.50** | 2 | 1 | 3 | 2/1 |
| macro3 | 72.33 | 72.50 | +0.17 | | | | |
| pooled 600 | | | | 6 | 5 | 11 | |

ids aligned in all three; the identity `(b−c)/n == accA4 − accA3` holds in every
stratum. **race breaks the sciq/cqa tie toward "no effect", not toward a sign** —
+0.50 pp is one net row in 200.

## ⛔⛔ The "signal is exactly the size of the noise" coincidence is an artefact of a shared arm

G3 reports that swapping the latent and merely changing the seed both move macro3
by **+0.1667**, "measured two independent ways". Both numbers reproduce. But:

```
macro(A4@43) − macro(A3@43) = +0.1667      (swap)
macro(A4@43) − macro(A4@42) = +0.1667      (noise)
macro(A3@43) = 72.3333      macro(A4@42) = 72.3333      difference 0.0e+00
```

⇒ ⭐ **The two deltas share the same minuend and have numerically equal
subtrahends, so their equality is arithmetic, not empirical.** It is one
measurement reported twice, not two independent ways of measuring.

Worse, the pooled tallies are **not made of the same rows** but overlap heavily:

| | swap disc | noise disc | shared (task,id) |
|---|---|---|---|
| pooled | 11 | 11 | **7 of 11** |

⇒ ⭐⭐ **The 11-vs-11 match is largely the same seven rows counted twice.** These
are the rows that flip under *any* perturbation — the intrinsically unstable
examples. A comparison of two tallies that share 64% of their events cannot
support "independent".

And the two comparisons are not even composed alike: the noise pair differs on
**race only**, the swap pair differs on **all three tasks**. The macro agreement
hides a complete disagreement about *where* the movement is.

### The Fisher test at the new table shape is still powerless

| table | p | OR |
|---|---|---|
| `[[1,599],[1,599]]` | 1.0000 | 1.000 |
| `[[11,589],[11,589]]` (G3's) | 1.0000 | 1.000 |
| `[[300,300],[300,300]]` — 50% flip | 1.0000 | 1.000 |

⇒ Same defect as before at n=600. **Drop the Fisher line.** The defensible claim
is the *size*: 11/600 = 1.8% discordance, macro |Δ| ≤ 0.17 pp, pooled 95% CI
[−1.50,+1.50]. That is "no detectable effect at n=200/task", never "identical",
and it should be stated once, not as two corroborating measurements.

## ✅ The scorer-independent logit evidence is sound, but the headline number is wrong

G3's best contribution: `first_token_logits_top10` needs no parsing, so it is
immune to every scorer defect we found. Verified, and **its falsifiability
control is correct** — 1408/2400 rows sit in a colliding group, so equality *can*
occur. But the tally G3 leads with does not match its own parenthetical:

| mode | task | n | identical top-10 **ids** | identical full lists |
|---|---|---|---|---|
| none | sciq / cqa / race | 200 | 7 / 5 / 12 | 0 / 0 / 0 |
| refeed | sciq / cqa / race | 200 | **84 / 104 / 90** | 0 / 0 / 0 |
| total | | 1200 | **302** | **0** |

⇒ ⛔ **"identical on 0 of 1000 pairs" is true only of the full (id, value) lists;
on token IDS alone it is 302/1200, i.e. 25% of pairs are identical** — and under
`refeed` it is 44%. G3's own parenthetical (193/195/188 and 116/96) is the right
reading and contradicts its headline; the 0/1000 comes from comparing lists that
include bf16 logit *values*, where ties are broken by noise.

⇒ ⭐ **State which projection of the object is being compared.** "Identical
logits" (0/1200), "identical top-10 token ids" (302/1200) and "identical top-1"
are three different measurements. The defensible claim is the weaker and still
sufficient one: **under `none` the latent changes the top-10 id set on 94–98% of
examples while accuracy stays pinned at 0.00** — the write is live and useless.
Under `refeed` it changes only ~56%, which is itself evidence that refeed dilutes
the write.

## ⛔ RETRACTED: my claim that `letter_first` is unrecorded

I reported `--letter_first` as "never recorded in outputs" and "unrecoverable
from outputs". **G2 is right and I was wrong.** Enumerated every `.json` under
both output roots:

| | count |
|---|---|
| `.json` paths found | 24 |
| parseable as JSON | 22 |
| **not JSON** (two dry-run logs, reported not skipped) | 2 |
| **`/letter_first` present** | **13 / 22** |
| distinct (path, value) | `/meta/args/letter_first = False` ×10, `/scorer_audit/letter_first = False` ×3 |
| negative control (`/letter_furst`, `/zzz_absent`) | 0 / 22, 0 / 22 |

The published `main/` cell records it **twice** — `summary.json`
`/meta/args/letter_first = False` and `g2_aggregate.json`
`/scorer_audit/letter_first = False`. So a rescorer that assumes `True` is
**contradicting a recorded value**, not filling a provenance gap.

⇒ ⭐⭐ **My error was scoping the search to the per-row schema.** The per-row
object has 33 keys and `letter_first` is not among them, which is what a
row-level check sees — but the flag is a *run-level* argument and lives in the
run-level artefact. **An absence claim must state the object class it searched,
not just the file set**: "absent from the 33-key row schema" is true and useful;
"unrecorded" is false. Same class as the `gluefix` "no producer" error, where I
grepped filenames instead of output keys.

My 40/6200 figure survives only with its framing changed: it is the
**sensitivity of the table to a scorer that ignores a recorded flag**, not a gap.

### ⛔ And my own positive control failed in the first version of that check

The first run printed `POS CONTROL '/task': 0/22` — my control key did not
exist (the real key is `tasks`, a list). The negative controls were both 0, so
**had I not included a positive control, a 0/22 target result would have looked
like a finding instead of a broken query.** After fixing: `/seed` 15/22,
`/tasks/[]` 13/22, and a depth-3 probe `/meta/args/seed` 10/22 confirming the
walker reaches the depth `letter_first` lives at.

⇒ ⭐ **A positive control has to be a key you have *seen*, not one you expect.**
I guessed `/task` from memory of the schema and guessed wrong.

## ⛔ G2's item-2 charge: the mechanism is real, its two examples are not

G2 charges that my `wb_norm` "embeds the exact bug we diagnosed" because
normalisation strips `[^\w\s]` before the boundary test looks, and cites
`race__A0` id=16 (gold `My Invaluable Friend.`) and id=68 (gold `0-3`).

**The general mechanism is real — 43/4200 rows change verdict between a
normalised-text boundary test and a raw-neighbour one.** But it is not the
mechanism at work in either cited row:

| id | gold | wb_norm | wb_raw |
|---|---|---|---|
| 16 | `My Invaluable Friend.` | False | **False** |
| 68 | `0-3` | False | **False** |

The period is **inside the gold string**, so the raw match consumes it and the
raw right-neighbour is the artefact letter `U` (`…Friend.User:`). wb_raw refuses
these rows for exactly the same reason wb_norm does. The real cause of the 43
divergences is different: normalisation **collapses internal whitespace and
punctuation**, so a multi-word gold (`Los Angeles`) can match under one
representation and not the other.

⇒ ⭐ **A correct diagnosis with the wrong worked example is still a defect
report I have to re-derive.** Both of us have now done this: cite the row, then
check the row actually exhibits the mechanism.

## The word-boundary number as a function of the boundary definition: 5 implementations

All on the same 4200 rows (`main` 15 cells + `side_refeed` 6 cells), **negative
control `frozen` reproduces the stored `correct` AND `invalid` bits 0/4200
mismatches** in every run below, positive control live in every mode.

| impl | boundary judged on | controls | A0 macro3 | Δ | A1 macro3 | Δ | cells moved /21 | rows flipped /4200 |
|---|---|---|---|---|---|---|---|---|
| shipped | — | — | 71.17 | — | 5.33 | — | — | — |
| `wb_norm` (mine) | normalised text | 6/8 | 69.33 | −1.83 | 3.83 | −1.50 | 7 | 56 |
| `wb_raw` (G2's) | raw neighbours | 6/8 | 69.17 | −2.00 | 4.67 | −0.67 | 5 | 46 |
| `wb_art` | raw + speaker-tag exemption | **8/8** | 71.00 | **−0.17** | 5.33 | **+0.00** | 2 | 2 |
| `trunc_raw` | truncate fused tag, then raw | see below | 69.67 | −1.50 | **6.83** | **+1.50** | — | 36 |

⇒ ⭐⭐ **The only implementation that passes every control barely moves the
table** (A0 −0.17, A1 ±0.00, 2 rows). G2's reported spread of 1.50 pp and my
−1.83 both come from implementations that *fail* two controls each. The honest
summary is not "the fix moves A0 by 0.3–1.8 pp"; it is **"under the only
boundary definition that survives its own control set, the defect is worth 2
rows in 4200."**

My `wb_art` reaches 8/8 where G2's reached 7/8 for the reason G2 identified: its
exemption required `raw[rb].islower()`, which is False for the digit in `0-3`.
Mine tests whether the following text *starts a known speaker tag*, which is
independent of case and of the gold's final character.

### ⛔⛔ And the "it can only lower credits" claim — which G2 and I both made — is FALSE

`trunc_raw` deletes the fused artefact instead of exempting it, which is the more
principled repair. It sends **A1 sciq 0.00 → 4.00 and A1 macro3 5.33 → 6.83**.

| A1 sciq row | gold | prediction (raw) | after truncation |
|---|---|---|---|
| 23 | `biotechnology` | `The answer is B.Q: 请将下文翻译成中文…` | `The answer is B.` |
| 50 | `active` | `The answer is (A).Q: 请将下文…` | `The answer is (A).` |
| 40 | `buoyant` | `The answer is B.User: 请详细描述…` | `The answer is B.` |

⇒ ⭐⭐ **A defect fix can RAISE a control arm.** The artefact was not merely
adding spurious credits; on sciq/A1 it was *destroying* recoverable ones, because
the fused Chinese continuation made `letter_of` fail on a generation that had
already named its answer. A1 is the arm that is supposed to be at chance, so a
repair that lifts it by +1.50 pp changes what the control arm means.

These letters are **not** spurious: on sciq/A1 176 rows yield an extractable
letter and **84/176 = 47.7% agree with the gold position against a 25% chance
baseline**, and only 8/200 rows have alphabetically-ordered choices, so this is
not a position artefact. The A1 arm is emitting real letter answers that the
shipped scorer discards.

### The interaction with `letter_first`, on the 15 published `main` cells

| macro3 | shipped | `lf=True` | `trunc` | `trunc+lf=True` |
|---|---|---|---|---|
| A0 | 71.17 | 71.83 | 69.67 | 71.67 |
| A1 | **5.33** | 5.83 | 6.83 | **7.50** |
| A2/A3/A4 | 0.00 | 0.00 | 0.00 | 0.00 |

⇒ **A1 spans 5.33 → 7.50 across four defensible reading choices, a 41% relative
move in the control arm**, while A2/A3/A4 are invariant at 0.00 under all four
(the algebraic exemption: no credits to lose or gain).

⇒ ⭐ **The headline conclusion is untouched and now better supported.** Every
one of five boundary definitions and four scorer-reading combinations leaves
A2 = A3 = A4 = 0.00. The latent arms are **robust to the entire scorer
controversy**; only the control arms A0/A1 are sensitive. That is the sentence
for the paper: the defect discussion is a caveat on A0/A1, and it cannot touch
the finding.

### ⛔ One more of my own controls was wrong

My truncation control set scored 3/4, the failure being
`"Assistant: yes\nUser: more"` → I expected no truncation. But the shipped
`score_row` calls `truncate_at_newline` **first**, so both strings reduce to
`"Assistant: yes"` — verified identical. **My control encoded an expectation
about a string the scorer never sees.** The 3/4 was my bug, not the code's; the
effective score is 4/4.

## ⛔ Two of my own pinned hashes were wrong, and my compliance criterion was invalid

**(a) I pinned the wrong file.** I recorded the two protected runners as
`35dc33a3…` and `597de511…`. `597de511` is **`g2_aggregate.py`**, not
`sanity_checks.py` (which is `891bf67a…` and was never pinned by me).
Both pinned runners verify unchanged; the second one just was not the file I
thought I had named. `g2_frozen_run_five_arms.py` and `g3_pinned_run_five_arms.py`
share md5 `35dc33a3…` — they are byte-identical copies.

**(b) I mis-transcribed a paper pin.** I recorded `main.tex` as `d9ca82a4…`;
it is **`d9ca82a2e4fc212c65e2524357838438`**. One character, and it was wrong in
every report I filed. Verified unchanged by mtime (`2026-09-23T03:12:10Z`,
predating every write of mine).

⇒ ⭐ **A hash pin transcribed by hand is not a pin.** Two of mine were wrong in
the same direction: they *looked* plausible and would have "verified" nothing,
because when the recomputed value disagreed my first instinct was to suspect the
file rather than my transcription.

**(c) ⛔⛔ My compliance criterion could not answer the question it was asked.**
I had been reporting "N files modified under `results/`/`outputs*/` since
<date>". Under the output trees the count is **115/115 modified since
2026-09-22, 86 since 17:45** — because **G2 and G3 legitimately write there**.
mtime cannot attribute authorship, exactly the error recorded in memory as
"mtime cannot归因 authorship; first attribute, then accuse". My number was
either vacuously 0 (wrong window) or alarmingly 115 (wrong question); neither is
evidence about *my* conduct.

The criterion that *can* answer it: **does my own code ever open a write handle
in a protected path?**

| | value |
|---|---|
| my scratch scripts enumerated | **62** |
| positive control (probe file with `open(...,"w")`) | **detected** — the grep works |
| scripts with any write/append/exclusive `open()` | **0** |
| scripts with `os.remove` / `os.unlink` / `shutil.*` / `.write(` / `json.dump(` | **0** |

⇒ ⭐ **Attribute by mechanism, not by timestamp.** "No write handle exists in any
of my 62 scripts" is a property of my code and is checkable; "0 files changed
since midnight" is a property of the clock.

## Final state of this segment

| item | value |
|---|---|
| PLAN.md | 2467 lines, md5 `3ea735ac…` |
| D4 `sanity_checks.py` | **exit 0, TP=34/34, FP=0/9**, md5 `891bf67a…` |
| pinned runners | `35dc33a3…` (frozen ×2), `597de511…` (aggregate) — unchanged |
| paper files | `appendix.tex 0b91c817…`, `main.tex d9ca82a2…` — unchanged, both **untracked** so mtime is the only valid witness |
| protected-path writes by me | **0 by mechanism** (62 scripts, no write handles) |
| GPU / cluster jobs | none |

### What the paper can say, after all five scorer variants

**A2 = A3 = A4 = 0.00 under every one of the 5 boundary definitions and 4
reading combinations tested.** The latent arms are invariant to the whole scorer
controversy; only A0/A1 move. `tab:latent-use-pending` should report the latent
arms as at-or-indistinguishable-from-zero **scoped to seed 42** (seed 43 cqa/A3
is 2/200, so never "exactly 0.00" unscoped), keep `refeed` out of the table, and
carry the A0/A1 caveat with its spread — **A1 spans 5.33 → 7.50, a 41% relative
move in a control arm.**

## G3's four-projection table: reproduces exactly, and the `none` row is contaminated by bf16 ties

All four projections verified at 600 pairs/mode, A3 vs A4, seed 43 — G3's numbers
are exact:

| projection | none | refeed |
|---|---|---|
| full `(id,value)` | 600/600 (100%) | 600/600 (100%) |
| top-10 ids **ordered** | 576/600 (96.0%) | 322/600 (53.7%) |
| top-10 ids **unordered** | 541/600 (90.2%) | 78/600 (13.0%) |
| top-1 id (argmax) | 47/600 (7.8%) | 20/600 (3.3%) |

⇒ The `0/1200 identical` headline holds only for full lists, where **bf16 logit
values** break ties. Confirmed and already retracted by G3.

### ⛔⛔ But 43.3% of all rows have a TIE at the 10th logit, so "the top-10 set" is not well defined

| | rows | share |
|---|---|---|
| rows scanned | 2400 | |
| 10th logit **ties** the 9th ⇒ set membership at the cut is arbitrary | **1039** | **43.3%** |
| top-1 ties the 2nd ⇒ **argmax itself** arbitrary | 142 | 5.9% |
| tie-group size at the 10th value | 1:1361, 2:317, 3:316, 4:214, 5:85, 6:66, 7:37, 8:4 | |

A tie at the truncation boundary means the exporter had to pick arbitrarily among
equal-valued tokens. **86 of the 541 `none` set-differences are pairs that a
tie-robust projection calls identical**, i.e. attributable to arbitrary
truncation rather than to any change in the distribution.

⇒ ⭐ **A top-k set comparison is only meaningful when the k-th value is strictly
greater than the (k+1)-th — which this export cannot show, because it only stores
10 values.** The tie rate is the measurable proxy and it is enormous.

### ⛔ And my own "tie-robust" fix is NOT a refinement — it answers a different question

Restricting to tokens with logit **strictly above** the 10th value moves the two
modes in **opposite directions**:

| mode | raw set differs | tie-robust differs |
|---|---|---|
| none | 541/600 (90.2%) | **440/600 (73.3%)** — down |
| refeed | 78/600 (13.0%) | **101/600 (16.8%)** — *up* |

The non-monotonicity is diagnostic, not a bug: dropping the tied tail changes the
**set size**, and two rows with equal raw sets can have different numbers of tied
tail tokens. Worked example (`none/sciq` id=4): A3 tail `…14.25, 14.25` → 8
unambiguous; A4 tail `…14.25, 14.25, 14.25` → 7. Raw sets equal, robust sets
differ.

⇒ ⭐⭐ **A projection that can call a pair *different* where the coarser one calls
it *identical* is not a corrected version of it.** I nearly reported it as
"G3's row, fixed". Controls that caught it: negative control — with no boundary
tie, `robust ∪ {10th} == raw` on **1361/1361**; positive control — `robust != raw`
on 2400 rows, so the projection is not a no-op.

### ⛔ The clean subset cannot carry the refeed contrast: n_keep = 44

| mode | arm | boundary-tie rate |
|---|---|---|
| none | A3 / A4 | **78.7% / 78.0%** |
| refeed | A3 / A4 | **8.8% / 7.7%** |

| mode | n_keep (neither arm tied) | raw-set differs on keep | ordered differs on keep |
|---|---|---|---|
| none | **44** | 29/44 (65.9%) | 42/44 (95.5%) |
| refeed | 518 | 61/518 (11.8%) | 262/518 (50.6%) |

⇒ ⭐⭐ **The tie rate is itself mode-dependent — 78.7% vs 8.8% — so "rows with no
tie" is a biased, mode-dependent subset and the two clean percentages are not
comparable.** With `n_keep = 44` for `none`, the clean subset cannot support any
cross-mode contrast. The cause is visible: the top1−top10 logit **spread** is
median **1.69** under `none` vs **4.56** under `refeed`, so `none`'s distribution
is flat enough that ten values collide constantly.

⇒ This is the same failure class as the earlier truncation work: **any
"after-exclusion" number must print `n_keep`, and ≤ a few dozen is not
estimable.** I printed it, which is why this did not ship.

### ✅ G3's refeed-dilution point survives, on the ordered projection only

`none` 96.0% → `refeed` 53.7% (ordered) is computed on all 600 pairs in both
modes with no exclusion, so it is comparable. The unordered 90.2% → 13.0%
contrast is **not** safe to lean on, because the tie rate differs 9-fold between
the modes and the unordered projection is exactly the one ties corrupt.
**Reorder-only pairs: `none` 35/600, `refeed` 244/600** — so G3's observation that
much of refeed's residual difference is mere reordering within the same ten
tokens is correct and is better evidenced by this direct count than by the
unordered row.

### ✅ G3's argmax reading is right, and stronger than it claimed

| mode | task | argmax moved | distinct argmax tokens in A3 (n=200) | modal token share |
|---|---|---|---|---|
| none | sciq | 11/200 | **3** | 91% |
| none | commonsenseqa | 14/200 | **3** | 89% |
| none | race | 22/200 | **3** | 56% |
| refeed | sciq | 9/200 | 7 | 26% |
| refeed | commonsenseqa | 4/200 | 7 | 68% |
| refeed | race | 7/200 | 24 | 50% |

⇒ Under `none` the entire first-token argmax collapses to **3 distinct tokens
across 200 examples**, one of them covering up to 91%. So "the argmax barely
moves" is not evidence against the write being live — **a first-token metric is
measuring the degenerate attractor, not the latent.** G3's point (a), confirmed
with the distinct-token count it did not have.

### The decoded output does differ, which is the scorer-free claim worth keeping

| mode | sciq | cqa | race |
|---|---|---|---|
| identical `generated_ids` (A3 vs A4) | none 109/200, refeed 117/200 | none 82/200, refeed 102/200 | none 124/200, refeed 131/200 |

**38–59% of examples decode to a different token sequence**, on no projection
choice and no scorer at all, while accuracy is pinned. That is the cleanest form
of "the write is live and useless" and it needs no tie-breaking discussion.

## ✅ G3's amendment to my item 5 is correct: the ordered row is tie-contaminated too

G3 rightly turns my own item-3 lesson back on my replacement metric. I had said
the ordered projection "is computed on all 600 pairs in both modes with no
exclusion, so it is comparable" — true about *exclusion*, but the exporter's
arbitrary ordering **among equal-valued tokens contaminates order as well as
membership**. Verified its tie-canonical projection (sort ids within each
equal-value group), with controls:

| control | result |
|---|---|
| rows with no tie anywhere (neg) | canon == raw on **711/711** |
| canon != raw (pos, must be > 0) | **1374/2400** |
| canon is a permutation of raw (invariant) | **2400/2400** |

| mode | raw ordered differs | tie-canonical differs |
|---|---|---|
| none | 576/600 (96.0%) | **567/600 (94.5%)** |
| refeed | 322/600 (53.7%) | **307/600 (51.2%)** |
| **drop** | **42.3 pp** | **43.3 pp** |

⇒ **The dilution conclusion survives and is marginally stronger under
canonicalisation.** G3's numbers reproduce exactly. The ordered row is usable,
but it needs this robustness line beside it and must not be called clean.

### ⭐ The artefact counts: G3's 9/15 and my 12/44 are both right, and the NET hides a reversal

| mode | raw | canon | **net** raw−canon | **gross** raw-only | **gross** canon-only |
|---|---|---|---|---|---|
| none | 576 | 567 | **9** | 12 | 3 |
| refeed | 322 | 307 | **15** | **44** | **29** |

G3's 9 and 15 are the *net* differences; my 12/44 are the *gross* raw-only
counts. `net = raw-only − canon-only` (12−3=9, 44−29=15), so both are correct
descriptions of different quantities.

⇒ ⭐⭐ **But the net is the misleading one to quote: it hides that 29 refeed pairs
move the OTHER way.** A net of 15 reads as "15 pairs were artefacts"; the truth
is 44 were artefacts and 29 were *concealed* differences that arbitrary raw
ordering had hidden. Same failure mode as the earlier "tally cannot impersonate a
delta" — **a signed net over two opposing populations reports neither.** G3
independently reached the same caution, declining to swap canonical in as "the
corrected number" precisely because it disagrees in both directions.

### ⛔⛔ n_keep = 0: the tie-free stratum for `none` is EMPTY

Restricting the ordered projection to pairs where **neither** row has any tie at
all:

| mode | n_keep | ordered differs |
|---|---|---|
| none | **0** | **not estimable** |
| refeed | 254 | 93/254 (36.6%) |

⇒ ⭐⭐ **Not a single one of the 600 `none` pairs has two tie-free rows.** Every
`none` measurement of first-token structure rests on rows where the bf16 export
had to break a tie somewhere. This is the strongest possible form of the earlier
`n_keep = 44` warning: the clean stratum is not merely small, it is empty, and any
"tie-free" statement about `none` is **vacuous** rather than uncertain. Printing
`n_keep` is again what caught it — a percentage over 0 rows would have been an
exception, not a number.

### ✅ Arm attribution for my claim-6 figures — G3 is right

| mode | task | A3 distinct / modal | A4 distinct / modal |
|---|---|---|---|
| none | sciq | 3 / 91% | 3 / 91% |
| none | commonsenseqa | 3 / 89% | 3 / 89% |
| none | race | 3 / **56%** | 3 / **60%** |
| refeed | sciq | 7 / **26%** | 7 / **24%** |
| refeed | commonsenseqa | 7 / 68% | 7 / 68% |
| refeed | race | **24** / 50% | **23** / 50% |

My "race 56%" and "refeed-race 24 distinct" were both the **A3** arm; A4 gives 60%
and 23. Four of the six cells are arm-identical, so the two that differ must name
their arm. ⇒ ⭐ **A per-arm statistic quoted without its arm is under-specified
even when most cells happen to agree.**

### ✅ G3's degenerate-CI defect is the same family as the Fisher line

G3 reports its FFFD sensitivity check ended in "each CI contains the other point
estimate", where the `none` kept subset has **n=147 with 0 nonzero paired
differences**, so its bootstrap CI is **[0.00, 0.00]**. A zero-width interval
cannot contain any other point estimate, so the check was **forced to print
INVESTIGATE regardless of the data**.

⇒ ⭐⭐ This is the third member of the family in this work: **Fisher on equal
counts** (output forced by input equality), **`None != int`** (forced by type),
and now **a containment test against a zero-width interval** (forced by
degeneracy). The general rule: *ask what value the statistic takes when the
hypothesis is false; if it cannot change, it is not a test.* Note the direction
here — this one fails toward a **false alarm**, whereas the Fisher line failed
toward false reassurance. Both are instrument failures; only one looks alarming,
which is why the reassuring kind survives longer.

⇒ ⭐ A CI-containment check needs a **width guard**: report the comparison as
undefined when either interval is degenerate, which is what G3 has done.

## ✅ G2 reproduced the A1 rise independently, and found a SECOND mechanism

G2 confirms the retraction of "can only lower credits" by reproduction: on its 15
published `none` cells (3000 rows, neg control frozen-vs-stored 0/3000 on both
`correct` and `invalid`), truncating at a fused tag gives **A1 5.33 → 6.83
(+1.50)**, matching my figure exactly.

### ⭐ The second mechanism is real and structural: containment is longest-wins

`:297-300` — `best, best_len = None, 0` then `if c and c in ng and len(c) > best_len`.
Because the longest containing candidate wins, **refusing a long wrong candidate
promotes a shorter right one**, so a *stricter* rule can award credit a looser
rule withheld. Verified synthetically with controls (choices `['cat', 'the big cat
sat']`, gen `the big cat sat here`): shipped picks idx 1; refusing idx 1 promotes
idx 0; negative control refusing nothing returns idx 1.

⇒ This is **independent of my released-letter mechanism** and means the
non-monotonicity is structural to any longest-wins cascade, not specific to
truncation. ⭐⭐ **Practical consequence, and G2 states it exactly right: a net
delta is not evidence of direction — gained and lost must be printed separately.**
My own grid confirms it: every truncation variant is `gained 12 / lost 11-12`, a
net of +0 to +1 that conceals 23-24 moving rows.

### ⛔ But G2's ANY-vs-FUSED axis is mislabelled — the real axis is the search offset

G2 reports `trunc at ANY tag` → A0 67.33 and `trunc at FUSED tag` → A0 69.83, and
attributes the 2.50 pp gap to whitespace-preceded tags being legitimate text. On
my transcription **ANY and FUSED give identical results** (A0 69.83, A1 6.83).
Diagnosed:

| | count (of 3000) |
|---|---|
| rows with any tag at pos > 0 | 773 |
| rows with a **fused** tag at pos > 0 | 605 |
| rows where first-ANY ≠ first-FUSED | **192** |
| …of those, a **newline** precedes it | **192 (all)** |
| …remaining that could actually differ | **0** |

⇒ **The ANY/FUSED axis is inert on this row set**, because `truncate_at_newline`
runs first and has already cut everything a whitespace-preceded tag would have
cut. What *does* reproduce G2's numbers is the **search start offset**:

| variant | A0 macro3 | A1 macro3 | rows truncated to EMPTY |
|---|---|---|---|
| `search(g, 1)` | 69.83 | 6.83 | **0** |
| `search(g, 0)` | **67.33** | **5.83** | **49** |

Searching from position 0 matches a tag at the very start (`A:\n…`,
`Assistant: (C) …`) and **empties the generation**, which makes the row invalid
⇒ scored wrong. 49 rows, **21 of which the shipped scorer had marked correct**
(`race__A0` 27, `race__A1` 18, `commonsenseqa__A1` 3, `sciq__A1` 1).

⇒ ⭐⭐ **G2's 67.33 is a measurement of destroying 49 generations, not of removing
an artefact.** The two numbers it offers as "two readings of the same fix" differ
by a mechanism neither label names. Same class as its own point 7 (moving two axes
while believing it moved one) — and as my earlier four-wrong-numbers episode.
**When two variants of one rule differ, find the rows that differ and read them
before naming the axis.**

## ⭐⭐ G2's strongest contribution: the latent arms emit no readable answer at all

This is a claim about the **generations**, not the scorer, and it upgrades my
point 5 from "the defect cannot reach the finding" to something much better.
Verified with three independent extractors, each control-calibrated:

| extractor | 8-case control | A0 | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|---|
| `ex1` (isolated capital) | **8/8** | 444/600 | 454/600 | **5** | **7** | **7** |
| `ex2` (answer-phrase) | 7/8 | 387/600 | 424/600 | **0** | **0** | **0** |
| `ex3` (parenthesised) | **8/8** | 435/600 | 446/600 | **0** | **0** | **0** |
| `letter_of` (shipped) | **6/8** | — | — | — | — | — |

**Positive control (plant one letter row per latent cell): A2 5→6, A3 7→8, A4 7→8,
3/3 PASS** — so the zeros are measurements, not dead code.

⇒ ✅ **G2's diagnosis of its own broken audit is confirmed**: `letter_of` scores
**6/8**, failing on `'The answer is B.'` because it is `re.match`-anchored at the
string start. That is why its first pass found 1 extractable row where I found
176 — instrumentation, not data.

### And my own extractor was the loose one — all 19 hits are `Franklin D. Roosevelt`

My `ex1` finds 19 latent-arm rows with a letter. Inspecting **all 19**: every
single one is a **middle initial in prose** — 12 are `Franklin D. Roosevelt's
first inaugural` and the rest the same newswire boilerplate. Not one is an answer.

⇒ ⭐ **G2's "0/600 readable answers" is right in substance and my 5-7 was my
extractor matching any isolated capital A-H.** An extractor that scores 8/8 on a
control set built from *answer-shaped* strings can still be wrong on *prose*,
because the control set never contained prose with a middle initial. **A control
set calibrates against the shapes it contains and says nothing about shapes it
omits** — the same lesson as my `"the White House said"` failure, arriving from
the opposite direction.

### Supporting evidence for prompt-independence (verified)

| arm | sciq distinct/200 | cqa | race |
|---|---|---|---|
| A0 | 198 (0.990) | 200 (1.000) | 169 (0.845) |
| A1 | 78 (0.390) | 58 (0.290) | 160 (0.800) |
| **A2** | **29 (0.145)** | **35 (0.175)** | **72 (0.360)** |
| A3 | 37 (0.185) | 61 (0.305) | 92 (0.460) |
| A4 | 36 (0.180) | 61 (0.305) | 92 (0.460) |

`sciq__A2` repeats `'The New York Times, The Washington Post and The Wall Street…'`
in **59/200** rows; `sciq` A3 vs A4 are **byte-identical on 123/200**. The latent
generations are largely prompt-independent boilerplate, which is why no reading of
them yields an answer.

⇒ **The sentence for the paper is now G2's, not mine:** under every extractor
tested, the latent arms produce **no readable answer letter and no prompt-relevant
content**; their 0.00 is a property of the generations, so no scorer choice can
move it.

## ⛔⭐ G3's FFFD retraction is over-broad, and its own test had a Simpson's paradox inside it

G3 asked me to retract, downstream, the framing that the U+FFFD decode defect "only
inflates our own arm". I checked it on both spans before doing so. **The retraction is
correct about G3's operation and wrong as a general statement, and my first attempt to
check it reproduced G3's error.**

**First, my own error, because it is the more instructive one.** I began by contrasting
the accuracy of FFFD-containing rows against FFFD-free rows, pooled over all five arms:

```
  arm  n_fffd  acc_fffd  n_clean  acc_clean    delta
   A4     428      8.18      772      51.68   -43.51
```

That −43.51 is **entirely a mode confound**. A4 exists in two modes with very different
accuracy (`main` 0.00, `side_refeed` 72.33) and very different FFFD rates (63.5% vs 7.8%),
so pooling puts mostly-`main` rows in the FFFD bucket. Within each mode the effect vanishes:
`main` delta **+0.00**, `side_refeed` delta **+2.32**. Same shape on the MemEval span, where
the pooled contrast reads **+10.02** and the stratified one reverses: of 22 strata,
**0 are more accurate, 3 are less accurate, and 19 are NOT ESTIMABLE** because only our own
arm emits FFFD at all (41/50 rows for `champ-dynamic-30k`, **0/50** for every HF baseline).
⇒ *A subset-vs-subset accuracy contrast is not a test of a scoring defect; it is a test of
whichever confound is correlated with subset membership.*

Also: in `main`, arms A2/A3/A4 have **0 credits**, so the FFFD direction there is
**algebraically unanswerable** — a bucket with no credits cannot lose or gain any.

**Second, the substantive finding: G3 and the scorer perform opposite operations.**
- G3 *deletes* the FFFD character (`prediction.replace(FFFD,"")`), splicing the run-on
  continuation onto the answer. This loses 0 of 51 credits — correct, and it is a no-op
  because the credits never depended on that one character.
- `run_open_context_bench.py:542` *maps it to a line break* (`span = span.replace("�","\n")`),
  which **cuts the span there**, because every cut pattern is newline-anchored.

Those differ by everything. Running the real scorer read-only over the relay artefact
(`results/benchmarks/longctx_relay/champ-dynamic-30k__memeval.json`; nothing written, and
I did **not** run `rescore_memeval.py`, which rewrites `results/` in place):

```
  NEG CONTROL gold scores 1.00 | POS CONTROL junk 0.00, empty 0.00
  reverting the fix: RISES 1/41, FALLS 0/41, unchanged 40/41
  mean over the 41 FFFD rows: 19.51 (fix ON) -> 21.95 (fix OFF)   [+2.44]
```
The fix's removal **raises** our arm. So on the MemEval span the recorded memory claim is
right and G3's requested retraction does not apply. `:536-541` documents the mechanism with
a worked example (gold "United Airlines" credited 1.0 because a later invented turn said
"via United Airlines") and the smoking-gun asymmetry: **our arm emits token 193 173× and
token 11 (newline) 0×, while `rwkv7-world3` emits 11 899× and 193 0×**, with byte-identical
tokenizer files — so the newline-anchored cut *never fired for our arm alone*.

**Third, what this means for our own table — the reason it was worth chasing.**
`g2_frozen_run_five_arms.py:237-241` (`truncate_at_newline`) is newline-anchored and has
**no** FFFD normalisation, i.e. it carries the exact defect MemEval fixed. Transplanting the
fix across all 21 cells / 4200 rows (neg control: no FFFD-free verdict changes, True;
pos control: planted `X<FFFD><good answer>` flips True→False):

```
  main         A0 71.17 -> 71.17   A1 5.33 -> 5.33   A2/A3/A4 0.00 -> 0.00
  side_refeed  A0 58.17 -> 58.17   A4 72.33 -> 72.33
```
**+0.00 on every one of the 7 macro cells**; 20/21 cells have zero row changes and
`side_refeed/sciq/A4` moves 1 up and 1 down. So the defect is real and **inert here** — our
arms are exempt for the same algebraic reason as before (A2/A3/A4 have no credits to lose)
and A0 has **0.0000% FFFD in all 6 of its cells**, checked not assumed.

**Fourth, the FFFD rate is itself a scorer-independent signal.** It is strictly ordered
A0 < A1 < min(A2,A3,A4) in **all three tasks independently** (cqa 0.0/3.5/34.0,
race 0.0/73.5, sciq 0.0/27.0/28.5), and a label-shuffle negative control breaks the
ordering on **190/200** shuffles. Writing a latent drives the decoder to emit an invalid
UTF-8 byte it otherwise never emits: degradation visible in the *byte distribution*, with
no scorer in the loop at all. This corroborates the headline from a third direction.

**Net for the deliverable:** I do not carry "FFFD inflates our arm" in the latentuse table —
not because it is false, but because on this span it is **measurably a no-op** (+0.00 on 7/7
cells). The memory note that records it for MemEval stays as written, now with the direction
re-derived under the real scorer, and gains the rider that the claim is **operation-specific**:
*delete-the-character* and *cut-the-span-there* move the score in different directions, so
"the FFFD defect" is not one intervention.

## ⭐⭐ G3's FFFD->newline finding is real and it makes the headline stronger — with three corrections

G3 re-scoped its retraction to the `delete` operation (correct) and then reported that the
newline operation destroys the only nonzero cell in its `none`-mode table. I verified this
against its own artefacts under `latentuse_g3/full_*` (16 cells, 3200 rows; read-only).

**CONFIRMED, and it is the most useful thing to come out of this exchange.**
`full_none_seed43/commonsenseqa/A3` = **2/200 = 1.00%**, credited example_ids **111 and 159**,
exactly as G3 said. Enumerating *every* `none`-mode cell: **1 of 10 is nonzero**, so its
"the only nonzero cell" claim is confirmed rather than assumed. Both credits come from the
run-on continuation *after* the FFFD (gold absent before it, present after it in both):

```
  id=111 gold='house'  '...deal done."<FFFD>The White House has been working with Congress...'  FFFD at 89/143
  id=159 gold='ring'   '...Wall Street Journal.<FFFD>The company also said it would be hiring...' FFFD at 68/139
```
Under `replace(FFFD,"\n")`, `none`-mode A3 goes **macro 0.33 -> 0.00** and every A3/A4 cell in
both seeds is 0.00. Controls for the operation: NEG an FFFD-free credited row True->True;
POS the same row with FFFD prepended True->False.

### Correction 1: the two ids die by DIFFERENT mechanisms, so the fix is not one fix

```
  id=111 gold='house': survives FFFD->newline? False   survives word-boundary? TRUE
  id=159 gold='ring' : survives FFFD->newline? False   survives word-boundary? False
```
`ring` was credited inside **hi*ring*** — the bare-substring defect that MemEval fixed
separately with `memeval_find_phrase` ("plain substring matching made 'car' fire inside
'care'"). So id=159 dies under *either* fix independently; id=111 (`White House`, a real
word-boundary match) dies **only** under the FFFD operation. ⇒ The 1.00% is not one artefact
class, it is two, and only one of them is about FFFD. Saying "the 1.00 is the FFFD defect"
over-attributes by half.

### Correction 2: "net -4" is gross lost 4 / gained 3, and 11/16 cells cannot answer

Stratified per cell, never pooled, gross both ways:

```
  full_none_seed43/cqa/A3       cred   2   lost 2   gained 0   (FFFD-credited 2)
  full_refeed_seed43/sciq/A3    cred 174   lost 1   gained 2   (FFFD-credited 29)
  full_refeed_seed43/sciq/A4    cred 177   lost 1   gained 1   (FFFD-credited 29)
  ... 11 of 16 cells: 0 credited FFFD rows -> NOT ESTIMABLE
  GROSS: lost 4, gained 3     (the signed net is -1, not -4)
```
The operation **gains** credits too — cutting a run-on can promote a shorter right candidate,
the longest-wins mechanism at `g2_frozen_run_five_arms.py:297-300`. A net figure over two
opposing populations reports neither, and `-4` reads as "strictly destructive" when the
measurement is 4 down / 3 up. Also G3's denominator of **84 credits** does not match these
16 cells (**871** credits: none 2, refeed 869), so its 84 is a different cell set — worth
naming, because a rate over an unstated cell set is the confound we keep catching.

### Correction 3: seed 44 is MISSING both `race` cells, so its macro is over 2 tasks not 3

```
  full_none_seed43: 6 cells, all n=200
  full_none_seed44: 4 cells, all n=200   MISSING: race__A3, race__A4
```
Every seed-44 `none` cell is 0.00, but that macro averages **2** tasks while seed 43 averages
**3**. Those are not the same statistic and must not be printed in one column. This is the
`(dataset,length)`-grid lesson again: the unit is the cell, and a missing cell is not a zero.

### Consequence for tab:latent-use-pending — my own guidance loosens, but only just

I previously insisted A2/A3/A4 be reported as at-or-indistinguishable-from-zero **scoped to
seed 42**, because seed 43 cqa/A3 was 2/200. That 2/200 is now shown to be **two scoring
artefacts, neither of which is a model behaviour**. Under either defensible fix the cell is
0.00 and the seed-43 exception disappears. So the unscoped statement becomes defensible —
but I will still **state the scope and the fix together** rather than dropping the caveat,
because as-run the artefacts print 0.33 and "0.33 vs 0.00" has two different readings. The
honest sentence names the operation: *zero under the shipped scorer up to two containment
artefacts, and exactly zero once FFFD is treated as a line break.*

Net effect: **all three of A2/A3/A4 are zero in every complete cell across seeds 42/43/44**
once the artefacts are removed — which is a stronger claim than the one I was protecting,
arrived at by G3 finding the defect in its own favour-removing direction.

## ⭐⭐ G3's "extractor FP rate is a function of max_new_tokens" is confirmed, and the `none` stratum has only TWO distinct prose contexts

G3 claimed its 14 loose-extractor false positives are mostly `Washington, D` **truncated by
the 32-token cap**, so a hand-written prose stratum could never contain that class. I tested
it across all 39 available cells / 7400 rows (G2 seed 42 + G3 seed 43/44, read-only).

### First, two errors of mine, because they are the reusable part

**(a) My "loose" extractor was looser than the one under discussion.** Matching
`(?<![A-Za-z])([A-Ha-h])(?![A-Za-z])` — lowercase included — gives **5430** hits on latent
arms, because it matches the English article **"a"** (`'been able to sustain a profit'`).
Restricting to capitals `[A-H]` gives **2501**. The word "loose extractor" names a family,
not an implementation, and the count moves 2.2x between two members of it.

**(b) I pooled the write modes again and nearly reported a contradiction of our own
headline.** On the pooled set the top contexts are `'(D)\n\nQue'` 40x, `'(E)\n\nQue'` 29x,
`'(C)\n\nQue'` 21x — which look exactly like *real answer letters* and would falsify the
0/600 claim. Stratifying by write mode dissolves it completely:

```
    stratum  cells  rows  hits  rows w/ hit  credits
       none     21  3800    33           33        2
     refeed      9  1800  2468         1566     1303
```
Every parenthesised-letter context is a **refeed** row. The headline is about
`prefix_replay=none`; the alarming contexts are from the other mode. Third time this
workstream has caught the same pooling error, and it is the one that would have done the
most damage — it argues *against* our own result.

### The finding: the `none` stratum has exactly TWO distinct contexts in 3800 rows

```
  --- none ---   2 distinct contexts, 33 hits total
      23x  'n Washington, D'
      10x  'from Franklin D. Roos'
  --- refeed --- 903 distinct contexts, 2468 hits
```
So under `none`, **every single loose-extractor hit is one of two prose strings**, and G3's
attribution is right: 23/33 are `Washington, D`. Of all 2501 capital hits, **95 are the final
character** of the generation and **93 of those are at the 32-token cap** — i.e. the "letter"
is a word the cap cut in half. ⇒ **A prose control stratum written by hand cannot contain this
class**, because the class is manufactured by the decode budget, not by the language. The FP
rate of a loose extractor is therefore partly a property of `max_new_tokens`, exactly as G3
said. That is a genuinely new failure form and it generalises past this project.

### The strongest single number for the table

```
  '(X)' parenthesised letter, none-mode latent arms:   0/3800 rows
  '(X)' parenthesised letter, refeed latent arms:   1031/1800 rows
  POS CONTROL: regex finds a planted '(C)'            -> True
  NEG CONTROL: regex does not fire on 'Washington, D' -> True
```
**0 of 3800** across A2/A3/A4, three tasks and three seeds. The two prose hits are not
answer-shaped under any reading, and the `0/3800` is scorer-free.

### On G3's item 4 (its own broken check) — the important part is which check stayed silent

G3's derangement check printed `missing z=200/200 -> FAIL` because it built tuple keys
`("sciq",0)` against a cache keyed `"sciq/0"`. The load-bearing detail is not the typo: it is
that **the same cache simultaneously PASSED its CHECK 1**, because CHECK 1 compares two caches
to each other and never constructs a key, while only CHECK B constructs one. One dataset, two
mandated checks, opposite verdicts, difference entirely in the instrument. And its companion
predicate `z_donor==z_recipient` read **0**, the healthy value — the vacuity mirror: a lookup
that finds nothing reports "no collisions" and "all missing" at the same time, and only one of
those two readings looks like a problem. Its fix (assert row 0's constructed key exists, raise
with sample real keys) plus the newly added POS/NEG on the tensor predicate is the right shape:
**a predicate that had never been calibrated was carrying a mandated verdict.**

## ⭐⭐ G2's caption change is right, and it survives a wider slice than G2 measured

G2 proposes replacing the table's central claim — "the latent arms score 0.00" — with "the
latent arms produce almost no scoreable output at all", on the grounds that under
word-boundary containment the **denominator** nearly vanishes. I reimplemented `match_choice`
twice, identically except for the containment clause (`c in ng` vs
`re.search(r"(?<!\w)"+re.escape(c)+r"(?!\w)", ng)`), with controls:

```
  POS CTRL  match_wb('diagnosed', ['nose','ear'])          = None  (must refuse mid-word)
  NEG CTRL  match_shipped('diagnosed', ['nose','ear'])     = 0     (shipped DOES accept it)
  NEG CTRL  match_wb('the answer is nose', ['nose','ear']) = 0     (genuine hit survives)
```

**G2's numbers reproduce exactly** on its own slice (`g2_main`, 9 latent cells, n=1800):
`valid 13 -> 4`, with `5/9` cells collapsing from nonzero to zero. Extended to every complete
`none`-mode cell across all three seeds:

```
    arm  cells  rows  valid_ship  valid_wb  lost  gained   ship%    wb%
     A0      3   600         447       342   105       0   74.50   57.00
     A1      3   600          24         9    15       0    4.00    1.50
 latent     19  3800          31        11    20       0    0.82    0.29
```
**Gross 20 lost / 0 gained** on the latent arms — strictly directional, unlike the FFFD
operation. The validity rate is **0.82% -> 0.29% of 3800 rows**. So the caption is correct and
it holds on 19 cells rather than 9: the latent arms do not answer wrongly, they emit **almost
nothing a scorer can read as a choice**, and what little it does read is dominated by mid-word
accidents (`'nose'` in *diagnosed*, `'time'` in *timesthe*).

### ⛔ But the tightening is NOT specific to the latent arms, and the caption must not imply it is

A0 loses **105** valid rows (74.50% -> 57.00%) — more absolute losses than the latent arms
have valid rows in total. So word-boundary containment is a *global* scorer tightening, not a
probe that singles out the latent arms. The defensible comparison is the **ratio**, which
survives it: 57.00% vs 0.29% is a **197x** gap, versus 91x under the shipped rule. Stating
"the latent arms collapse to 0.29% under a stricter scorer" without also stating that A0 fell
17.5pp would imply a specificity the measurement does not have.

### ⚠️ A denominator guard that earned its place immediately

The seed-44 `race` cells **now exist on disk but are empty** (`n=0`), because G3's decode is
in progress. Read naively they contribute "0 valid of 0 rows", which silently enters a macro
as a zero. My guard excludes incomplete cells **by name with the count printed**:

```
  DENOMINATOR GUARD: 39 cells found; INCOMPLETE (n!=200): 2
     excluded by name: full_none_seed44/race/A3  n=0
     excluded by name: full_none_seed44/race/A4  n=0
  -> 37 complete cells retained
```
This is the "missing cell is not a zero" rule, except worse: a *partially written* file is not
even missing, so a `glob` finds it and an existence check passes. ⇒ **Completeness must be
asserted on row count, not on file presence.**

### Recommended table sentence (superseding my earlier one)

> Writing a latent does not produce wrong answers; it produces almost no scoreable output.
> Across 19 cells (3 seeds x 3 tasks, n=3800), 0.82% of latent-arm generations match any
> choice under the shipped scorer and 0.29% under word-boundary containment, against 74.50%
> and 57.00% for the no-write control on the same rows; accuracy among the few scoreable rows
> is zero up to two containment artefacts.

That statement is scorer-robust in the direction that matters (both rules agree), reports the
control on the same rows so the global tightening is visible, and replaces an accuracy claim
whose denominator was 31 rows with a denominator claim about all 3800.

## ⭐⭐ G2 was right to refuse my 342 — and the cause is neither of the operations either of us named

G2 swept 32 configurations of five axes over A0's 600 rows and found **0** giving my `342`,
proposing that my operation must also bound the *prefix* stage (its "prefix ALSO bounded"
gives 338, four away). It was right to hold the number. It was wrong about the mechanism, and
the real one is worse than a mislabelled axis.

### My 342 is correct for what it measures, and I can now attribute it stage by stage

```
  shipped stage attribution: {'contain': 422, 'prefix': 23, 'exact': 2, 'none': 153}   valid 447
  wb      stage attribution: {'contain': 251, 'prefix':  23, 'exact': 2, 'f1': 66, 'none': 258}   valid 342
  two independent implementations (match_wb, stage_wb) agree: 342 == 342
```
The prefix stage is **untouched at 23 in both** — so G2's prefix hypothesis is falsified
directly. The movement is entirely inside containment (422 -> 251), with 66 rows caught by the
F1 stage below it.

### The actual mechanism: `normalize_text` DELETES punctuation without inserting a space

`g2_frozen_run_five_arms.py:229-234` does `re.sub(r"[^\w\s]", "", t)`. So:

```
  nt('D. make peaceQ: 请将下文翻译成中文') == 'd make peaceq 请将下文翻译成中文'
```
The gold `make peace` is now followed by `q`, and `(?!\w)` fails. **The answer is correct and
correctly formatted; the normaliser glued it to the next token.** Of the 188 A0 rows that lose
containment, **51 are restored** by a normaliser that substitutes a space instead of deleting,
and 137 are genuinely mid-word. A space-inserting variant **gains 0 rows** that shipped
refused (control), so it is a strict refinement:

```
  shipped (punct deleted, unbounded contain) : 447
  bounded contain, punct still DELETED       : 342   <- my number, correct but confounded
  bounded contain, punct -> SPACE            : 360   <- the defensible operation
```
⇒ My 342 **conflates two defects**: unbounded containment and the punctuation glue. The number
to quote for word-boundary containment alone is **360**, and the 87-row gap G2 attributed to
the prefix stage is this glue. Neither of us named the axis that mattered; G2's sweep could not
contain it because all 32 of its configurations share the same normaliser, and my 24-variant
sweep (6 boundary kinds x raw/normalised x F1 on/off) also produces **no 429**, so its own
number is still unexplained on my side.

### ⭐ G2's letter-fallback point is the one that should change the caption, and it reproduces exactly

My "valid" counted `match_choice` stages only; the stored `invalid` flag reflects the **full**
shipped cascade including the bare-letter fallback. Both reproduce to the row:

```
    arm    n  stored_valid  match_valid  letter_only
     A0  600           556          447          109
     A1  600            90           24           66
     A2  600             4            4            0
     A3  600             5            5            0
     A4  600             4            4            0
  rows INVALID by flag but matched by match_choice: 0  (stored is a strict superset)
```
**Letter-only rows: A0 109, A1 66, latent 0/600 each, 0 in all 9 cells.** G2's figures are
exact. This matters more than the containment question: the letter stage is where the controls
live, and excluding it lowers the control from **92.67%** to 74.50%, which *flatters* the
contrast. A caption must name which definition it uses.

### Revised recommendation, holding what is not yet pinned

Use the **stored-flag** definition, because it is the shipped cascade and it is the
conservative choice for us:

> Writing a latent does not produce wrong answers; it produces almost no scoreable output.
> On the published seed-42 cells (n=600 per arm), 92.67% of no-write control generations and
> 15.00% of restored-memory generations yield a scoreable choice, against 0.67-0.83% for the
> three latent arms; among those few, accuracy is zero up to two containment artefacts.

**Held back, per G2:** the absolute control loss under a stricter scorer. The two candidate
operations differ 6x (18 vs 109 rows) and my own number turns out to be two defects at once,
so the operation is not pinned. **The claimable quantities are the ratio and its direction** —
every reading widens the gap — not the control's absolute loss.

### Accepted: G2's retraction of its own "22.00 unreachable"

Its sweep hard-coded `emptied -> invalid`, so the on-disk `probe_leadfix_family.py`'s 44/200
was **outside the family by construction** and its "0 configs" was true-but-vacuous. Reachable
set {0.00, 0.50, 8.50, 21.50, 22.00, 29.50, 30.00}; any A1 lead-fixed number must name
**prep x position x empty-rule**. Same form as my 342: a sweep is only as wide as its unnamed
axes, and "no configuration reaches X" is a statement about the sweep, not about X.

---

## Round 12 (2026-09-23): G2's two charges CONFIRMED; the 87 is a coincidence; attributor calibrated

### 12.1 Both charges against me are upheld

**Charge 1 -- my attributor scored only the first of three `match_choice` calls.** Confirmed at
`g2_frozen_run_five_arms.py:335-343`: `score_row` calls `match_choice` three times (`g`,
`strip_label(g)`, `strip_answer_lead(strip_label(g))`) before the bare-letter fallback. My number
was therefore "bounded containment, 1 call", not the shipped cascade. `/tmp/latentuse/cascade.py`
reproduces G2's four bins bit-identically:

| variant | valid | histogram |
|---|---|---|
| shipped, 3 calls | 447 | contain 422, none 153, prefix 23, exact 2 |
| shipped, 1 call  | 447 | contain 422, none 153, prefix 23, exact 2 |
| BOUNDED, 3 calls | 429 | contain 251, none 171, prefix 110, f1 66, exact 2 |
| BOUNDED, 1 call  | 342 | contain 251, none 258, prefix  23, f1 66, exact 2 |

My inference **"the prefix bin is untouched at 23, so my operation does not bound the prefix
stage"** was *invalid*. The conclusion happened to be true, but 23 is constant **because the call
that would move that bin is absent**, not because the stage is unaffected. Constancy of a bin is
not evidence that an operation leaves the bin's stage alone when the code path that feeds it is
not being exercised.

**Charge 2 -- my punct->space repair does not fix my own worked example.** Confirmed
(`/tmp/latentuse/case.py`): in `'D. make peaceQ:'` the characters between `peace` and `Q` are
`''` -- there is no punctuation. It is a **case transition**, so punct->space yields
`'d make peaceq ...'` which is still fused (bounded=False). A case-boundary split
(`(?<=[a-z])(?=[A-Z])`) is what restores it, and it passes 4/4 controls: restores the fusion, still
refuses `nose` inside *diagnosed* and `make peace` inside *peacemaking*, leaves clean strings alone.
Re-derived on the real 3-call cascade, A0 n=600: shipped **447**, bounded/delete **429**,
bounded/punct->space **433**, bounded/+case-split **436**.

### 12.2 The re-derivation G2 asked for: my 188 was neither bin, and the 87 is a coincidence

`/tmp/latentuse/migrate2.py` -- **one** attributor, both rules, full migration matrix, using the
runner's own `normalize_text`/`truncate_at_newline`/`strip_label`/`strip_answer_lead`/`letter_of`/
`_f1` via `importlib` (no reimplementation). Denominator printed first: A0 seed42 main = 600 rows.

```
shipped: {contain 422, letter 109, none 44, prefix 23, exact 2}
bounded: {contain 251, letter 111, none 60, prefix 110, f1 66, exact 2}

MIGRATION (shipped -> bounded), only rows that moved:
  contain -> prefix   87
  contain -> f1       66
  contain -> none     16
  contain -> letter    2
  TOTAL moved        171
contain: 422 -> 251   NET loss 171 = OUT 171 - IN 0
```

So G2's 171 is right and my 51+137=188 was **neither** the gross out-flow nor the net -- it came
from mixing two attributors. One attributor, one matrix, and the flow decomposes exactly.

**But the 87 is a coincidence of two different quantities** (`/tmp/latentuse/eightyseven.py`):

- X = rows migrating shipped-`contain` -> bounded-`prefix` = **87**
- Y = rows rescued by calls 2-3 under the bounded rule (valid at 3 calls, invalid at 1) = **2**
- `X == Y`? **False**. `|X n Y| = 2`, `X\Y = 85`, `Y\X = 0`.

Both 429-342=87 and the contain->prefix flow=87 are 87, and they are *not the same 87 rows*. The
prefix bin arithmetic 23+87=110 is what actually links them: bounding containment pushes 87 rows
into the prefix stage, and separately the 2nd/3rd calls change validity for only 2 rows. Two equal
integers describing different populations -- the same trap as the earlier tally-vs-delta confusion.
**Numerical agreement between two derived counts is not evidence they describe the same rows; check
set identity, not cardinality.**

Controls on that comparison (a check that cannot fail is not a check):
- non-vacuity: both sets non-empty -- True
- discriminating: X differs from the superset "all shipped contain" (422) -- True
- **negative control**: rescue-by-calls-2-3 under the *shipped* rule = **0**, as it must be, since
  shipped 1-call and 3-call are identical. A nonzero here would have meant a broken instrument.
- key-set sanity: X, Y both subsets of the 600 keys -- True

### 12.3 Attributor calibrated against stored `pred_choice_index` (and it found a branch I had missed)

Reading all of `score_row` (`:307-360`) surfaced a stage my attributor did not model: a
`letter_first` branch at **`:328-331`, which runs BEFORE the three `match_choice` calls**. Bin order
matters, so the attributor is only comparable to the shipped scorer for the flag actually used.

`/tmp/latentuse/calib.py` replays the runner's own `score_row` against the stored fields of all
**3000** seed42/main rows and compares to the recorded `pred_choice_index`:

- `letter_first=False`: **0/3000** mismatches  <= the flag the data was scored with
- `letter_first=True`: 11/3000 mismatches
- **positive control**: corrupt 50 predictions to `"zzz no choice here"` -> **45/50** mismatches,
  so the check demonstrably can fail. (Not 50/50: 5 of the corrupted rows were already `None`.)

Conclusion: the published rows use `letter_first=False`, so the bare-letter branch really is last
and my bin ordering is right *for this data*. Had the flag been True, my `letter` bin would have
been systematically wrong, and nothing in my earlier numbers would have revealed it.

### 12.4 Two instrument guards that fired this round (and one path I had wrong)

- `hasattr` guard raised `runner lacks token_f1 -- unknown shape, refusing to guess`. The real name
  is **`_f1`** (`:267`). Without the raise I would have silently substituted my own F1 and reported
  an `f1` bin that was mine, not the runner's.
- A first attempt raised `NameError: letter_of` -- I had been about to reimplement the cascade a
  third time. Importing the runner's functions is the only version of this that is auditable.
- **My path for the runner was wrong**: it is under
  `research/laces-reasoning/scripts/eval/latentuse/`, not `research/DiffRwkv/...`. `md5 35dc33a3...`
  confirms the same protected file as before. Recorded because the briefing warned its line numbers
  and paths came from a live tree.

### 12.5 Caption claim verified: the latent zero is normaliser-robust; only the denominator moves

`/tmp/latentuse/robust.py`, seed42 main, valid/correct per arm under five normaliser variants:

| arm | shipped | wb/3call | wb/1call | wb+space | wb+case |
|---|---|---|---|---|---|
| A0 | 447/341 | 429/330 | 342/256 | 433/330 | 436/333 |
| A1 | 24/11 | 9/2 | 9/2 | 23/10 | 23/10 |
| A2 | 4/**0** | 1/**0** | 1/**0** | 2/**0** | 2/**0** |
| A3 | 5/**0** | 2/**0** | 2/**0** | 3/**0** | 3/**0** |
| A4 | 4/**0** | 1/**0** | 1/**0** | 2/**0** | 2/**0** |

Latent arms pooled (n=1800): valid **13 / 4 / 4 / 7 / 7**, correct **0 in every column**.

This is the strongest form of the headline available on the scoring side, and it is what the caption
should say: the number that moves with the normaliser is the **denominator** (13 -> 4-7), while
latent-arm **correct is 0 under every variant**. Adopting G2's edits: quote validity as a range,
state that the 4-13 scoreable rows are themselves mid-word matches, and drop "up to two containment
artefacts" -- the artefact count is normaliser-dependent and the zero is not.

### 12.6 Seed-44 race has LANDED -- supersedes the "2 tasks" note; zero claim now 20/21 cells

Re-checked the G3 tree this round (`/tmp/latentuse/seed44.py`, every cell asserted `n==200`, because
an empty-but-present file passes glob and would enter a macro as a legitimate 0.00):

- `full_none_seed44/race__{A3,A4}` are now **200 rows each** (they were n=0). My earlier correction
  that "seed-44 macro averages 2 tasks, not 3" was true when written and is now **superseded**: all
  three seed-44 none cells are usable, macro3 A3 = **0.00**, A4 = **0.00** (3/3 tasks).
- `full_refeed_seed44/` has appeared but is **NOT ESTIMABLE**: commonsenseqa and race ABSENT, sciq
  present with **n=0**. So the disjoint-pair refeed contrast is still unavailable. Reported as
  blocked, not averaged.

Consolidated none-mode tally (`/tmp/latentuse/zerotally.py`), denominator printed before verdict:
**21 cells / 4200 rows** across seeds 42 (A2/A3/A4) + 43 + 44 (A3/A4):

- **20/21 cells are exactly 0/200 correct.**
- The single exception is `seed43 A3 commonsenseqa` = 2/200 (1.00%), `example_id` 111 and 159 --
  already established to be two distinct scorer artefacts, not latent use.
- Discriminating control: the identical enumeration over A0 gives [125, 136, 166]/200, so the
  procedure is not one that returns zero regardless of input.

This is stronger than the earlier seed-42-scoped version and no longer needs the seed-43 caveat
carried in the headline: the exception is characterised, and adding seed 44 took the tally from
14 to 20 of 21 cells.

**Standing note on the "still missing" list.** The n=0 sciq pair in `full_refeed_seed44` is the same
shape as the seed-44 race cells were an hour ago. Files appearing is not the same event as rows
appearing, and a background job can flip a cell from n=0 to n=200 without announcing it -- so any
"X does not exist" line in this document is an observation with a timestamp, not a property.

### 12.7 The write-footprint check I first ran was NOT discriminating

My initial "no measurement file modified today" check listed 10 files and I nearly read that as a
violation. Two things were wrong with the instrument:

1. **mtime has zero discriminating power on this tree.** `*__A?.jsonl` modified today = **83**;
   modified in the last 30 days = **83**; total in existence = **83**. The whole tree was produced
   today, so "changed today" is true of every file under either predicate and separates nothing.
2. **mtime cannot attribute authorship.** The files that changed are in G2's and G3's own out-dirs
   and were written by their jobs. A timestamp says *when*, never *who*.

The criterion that can actually fail: **is any measurement file's mtime AFTER the analysis that read
it?** For `seed42/main`, the source of every scoring number in this document, distinct mtimes are
**18:21 / 18:59 / 19:37**, all before my round-12 scripts (22:30-22:50). Open write handles onto
`latentuse_g2_out`, `latentuse_g3` or `results/`: **0**.

Also worth pinning: `full_none_seed44/race__{A3,A4}` have mtime **22:40**, i.e. they landed *during*
this round, which is exactly why the earlier n=0 reading was correct when taken and stale by the time
I re-checked. And `git log` over these trees returns **0 commits** because they are untracked (`??`)
-- so "git shows no diff" would have been a vacuous defence here, not evidence of anything.

### 12.8 FILL-READY numbers for tab:latent-use-pending (appendix.tex:719, still all \pending)

`/tmp/latentuse/filltable.py`, seed 42, every cell asserted `n==200`. Accuracy (%) with the shipped
scorer, invalid-rate beneath.

**`prefix_replay=none` -- the mode that matches training** (`_singlez_coadapt_ce_from_z` prefills the
prefix, injects, forwards the SUFFIX only):

| Condition | SciQ | CQA | RACE | Macro3 |
|---|---|---|---|---|
| Same-loop backbone, no adapter (A0) | 83.00 | 62.50 | 68.00 | **71.17** |
| Same adapter, original principal memory (A1) | 0.00 | 8.50 | 7.50 | 5.33 |
| Training-mean latent (A2) | 0.00 | 0.00 | 0.00 | 0.00 |
| Shuffled sampled latent (A3) | 0.00 | 0.00 | 0.00 | 0.00 |
| Correct prefix-conditioned latent (A4) | 0.00 | 0.00 | 0.00 | 0.00 |

invalid-rate: A0 9.5/11.0/1.5 -- A1 99.5/74.5/81.0 -- A2 100.0/98.5/99.5 -- A3 99.5/98.5/99.5 --
A4 99.5/99.0/99.5.

The 0.00s are driven by **invalid**, not by wrong answers: the latent arms stop producing parseable
answers at all. That is why the validity frame in 12.5 is the right one -- the denominator collapses,
and the handful of rows that survive are mid-word containment artefacts with 0 correct.

**`prefix_replay=refeed`: the five-arm table is NOT COMPLETABLE in this mode.** Only A0 and A4 exist
(`side_refeed/` holds 6 row files, not 15); A1, A2, A3 are absent, so three of five rows would be
NOT ESTIMABLE. A0 58.17 vs A4 72.33 -- and this is the pair already established to be a **protocol
artefact**, because `inject_into_cache` overwrites the prefill, so A4 reads the prompt once while A0
reads it twice. It must not be used as the headline.

**Consequence for the deliverable.** The table can be filled completely and honestly only from
`none`, where the result is that writing the latent is **destructive, not merely inert** (A0 71.17 vs
A4 0.00), and the ordering A0 >> A1 >> A2=A3=A4 separates the three components the appendix text at
`:714` asks to isolate: auxiliary-cache treatment (A0->A1), and latent content (A1->A2/A3/A4).

I have **not** edited `appendix.tex` (md5 `0b91c817...` unchanged) or `main.tex` (`d9ca82a2...`).
Filling the table commits the paper to a mode and to a scorer, and three scorer decisions are still
open with the user (bare-letter fallback on empty strings, word-boundary containment, FFFD
normalisation). The numbers above are ready to paste once those are decided.

---

## Round 13: I RETRACT the "two different 87s" finding. G2 is right; the real finding is 85+2.

### 13.1 ⛔ RETRACTION of the round-12 headline finding (already delivered to the user)

I reported: *"429−342 = 87 and the contain->prefix flow = 87 are not the same 87 rows."* **That is
false.** Verified with set identity at both cascade levels (`/tmp/latentuse/levels.py`, denominator
600 printed first):

```
X  shipped-contain -> bounded3 prefix                        = 87
Y2 MATCH_CHOICE validity gained by calls 2-3  (= 429 - 342)   = 87
Y3 END-TO-END validity gained by calls 2-3                    =  2
X == Y2  ->  True    (|X n Y2| = 87, X\Y2 = 0, Y2\X = 0)
X vs Y3  ->  |cap| = 2, X\Y3 = 85, Y3\X = 0
```

**My root error: 429 and 342 are `match_choice`-level counts, so their difference is Y2, and I
compared X against Y3 — a third predicate — while calling it "429−342".** The two levels differ
sharply: `match_choice`-valid is 447/429/342, end-to-end valid is 556/540/538. Every "valid" count
in this document must now say **which level** it means.

The true statement, which is stronger than the coincidence I claimed:

> **85 of the 87 rows are already end-to-end valid at 1 call via the bare-letter fallback
> (`:340-343`), so their end-to-end validity cannot change; only 2 remain to be rescued. 85 + 2 = 87.**

The letter fallback *masks* 85 of the 87. That is why the counts looked unrelated to me: I was
seeing the masked residue (2) and comparing it to the unmasked flow (87).

**Also: the predicate "bounded1 'none' -> bounded3 'prefix'" is itself underdetermined.** Under
"`match_choice` returned None" it is 87 and equals X; under "end-to-end stage is none" it is 2.
Same English sentence, two different row sets. G2 and I each computed a different one and both were
internally right. *A predicate over a multi-stage cascade is not defined until it names the stage.*

### 13.2 My 433 and 436 had an unnamed axis: `match_choice:285` normalises the CHOICES too

G2 reported that no cell of its 16-cell grid reaches my 433. Correct, and the missing axis is
**sides** — whether the same normaliser is applied to the choice strings (`:285
norms = [normalize_text(c) for c in choices]`) or only to the generation. 32-cell grid
(punct x 4 case regexes x sides x calls), `/tmp/latentuse/find433.py`, 12 distinct values:

- **433 = (punct->space, no case split, gen-only, 3 calls)** — reachable in exactly 1 of 32 cells,
  which is why G2's grid (both-sides only) could not produce it.
- **436 is reached by 8 different cells**, including `(delete, lo0_up, gen)` and my
  `(space, lo_up, gen)`. **Those two 436s are NOT the same rows**: `|cap| = 429`, 7 rows each way.
- The case regex matters on its own: `(?<=[a-z0-9])` gives 444, `(?<=[a-z])` gives 442 (both-sides).

So my round-12 line "bounded/punct->space 433, +case-split 436" was **underspecified on two axes at
once** (sides, and which case regex). G2's prediction that my 436 would collide with a different
cell's 436 was exactly right — the trap I had just documented, committed by me two sections later.
**An integer is not a finding; the (axes -> row set) mapping is.**

### 13.3 Direction of every repair: confirmed, and it settles the paper question

`/tmp/latentuse/price.py`, full end-to-end cascade **including** the letter fallback, positive control
first: **all 15 published cells reproduced exactly, 0 mismatches** (G2's earlier price table scored A0
at 56.83 because it omitted the fallback; mine omitted it too in round 12).

| variant | A0 | A1 | A2 | A3 | A4 | A0 invalid/600 |
|---|---|---|---|---|---|---|
| **AS-RUN** (unbounded, punct deleted) | **71.17** | 5.33 | 0.00 | 0.00 | 0.00 | 44 |
| bounded, punct DELETED | 69.33 | 3.83 | 0.00 | 0.00 | 0.00 | 60 |
| bounded, punct -> SPACE (both sides) | 70.00 | 5.33 | 0.00 | 0.00 | 0.00 | 53 |
| bounded, punct -> SPACE (gen only) | 69.67 | 5.17 | 0.00 | 0.00 | 0.00 | 54 |
| bounded, +CASE SPLIT (both sides) | 70.83 | 5.33 | 0.00 | 0.00 | 0.00 | 48 |

**Every** bounded repair moves our headline arm **down** (−0.34 to −1.84) and leaves all three latent
arms immovable at 0.00. The latent immovability is algebraic (0 credits cannot fall), so the load-
bearing content is the A0 direction: **no admissible containment repair can manufacture a latent-use
signal, and as-run is the conservative choice.** This is the fifth variant added to G2's four and it
does not change the conclusion.

### 13.4 Caption: "4-13" was wrong, the range is 4-7 over repairs

G2's 32-cell product on the latent arms (n=1800): the 16 **bounded** cells give valid in {4, 7}; the
16 unbounded cells give 13, which is the as-run value, not a member of the repair range; correct == 0
in all 32. So the caption is **"13 as-run -> 4-7 under any bounded repair"**. My round-12 phrasing
"4-13 scoreable rows" conflated the as-run value with the repair range. G2 retracted the same error
independently (it had briefly written "true repair range is 4-13" — a cell that changes nothing is not
a repair).

### 13.5 Two more instrument failures, for the D4 list

- **Mine**: `if case:` tested the dict *key* (`"none"`, always truthy) instead of the value, so a
  `re.sub(None, ...)` raised `TypeError`. It crashed loudly, which is the good case — had the guard
  been `if CASE.get(case)` on a missing key it would have silently skipped the case split and printed
  plausible numbers for a variant I never ran.
- **G2's, reported by G2**: `find ... -newermt '-90 minutes'` **errored** ("Invalid timestamp") while
  piped to `wc -l`, printing a reassuring `0`. A failed check wearing a pass — the same family as my
  own non-discriminating mtime check in 12.7. **Piping a command into `wc -l` discards its exit
  status; a check must read the status, not only the count.**

### 13.6 ⛔⛔ The deliverable target `appendix.tex` is NOT in the compiled paper

Discovered while re-verifying hashes: `appendix.tex` and `main.tex` **both changed at 22:56** during
this session (md5 `0b91c817…`->`4bd16294…` and `d9ca82a2…`->`0c27670f…`). Not my writes — no script of
mine opens either in write mode, and none of my computed numbers (71.17 / 5.33 / 69.33 / 72.33) or
round-12/13 vocabulary appears in either file (grep count 0). Another agent or the user edited them.
**The five-arm table body is untouched: still 5 `\pending` rows, lines 716-734.**

Chasing that turned up something much worse for the deliverable:

1. **`main.tex` contains `\input` 0 times and `\include` 0 times.** It has `\appendix` at `:465`
   followed by its **own inlined appendix**.
2. **`appendix.tex` is never compiled.** Build artefacts exist only for `main.tex` and
   `main_skill.tex` (`.aux`+`.pdf` both present); `appendix.tex` has neither.
3. **`latent-use-pending` appears in 0 of the aux files**, and `main.tex` never references the label.
4. **`\pending` is used 5 times and defined nowhere** — no `\newcommand`/`\providecommand` for it in
   any `.tex`/`.sty`/`.cls` in the directory (control: the same grep finds all 45 other
   `\newcommand`s). Likewise `\LaLaLanDPendingfalse` is described in prose at `:712` but **no `\newif`
   defines it**. So `appendix.tex` would not even build standalone.
5. **`app:latent-controls` is one of 28 labels present in `appendix.tex` and absent from `main.tex`**
   (`app:` prefix; 17 are duplicated across both files).

   ⛔ **CORRECTED (round 14).** I first wrote "27 ... while 18 duplicated". Both were wrong, and
   neither came from a printed count — I **eyeball-counted a 28-line list** (it was printed in full,
   not truncated) and I put the *main.tex `app:` total* (18) in the slot where the overlap (17)
   belonged. Enumerated across the selector axis, printing every count:

   | selector | main | appendix | duplicated | appendix-only |
   |---|---|---|---|---|
   | `app:` | 18 | 45 | **17** | **28** |
   | `app:` or `sec:` | 33 | 46 | 18 | 28 |
   | `tab:` | 12 | 18 | 5 | 13 |
   | ALL labels | 69 | 93 | 25 | 68 |

   So my "18" is reachable only under the `app:|sec:` selector, and **27 is not reachable under any
   of them**. G2's 68/25 (ALL) and 28/17 (`app:`) both reproduce exactly. The conclusion is unaffected
   — 28, 68 and every other cell support "orphaned" — but this is the **third** time one of my counts
   has landed one away from a reachable value under a neighbouring predicate (cf. 27 vs 28, 433/436's
   `sides` axis, 188 vs 171). ⭐ **Never hand-count a printed list: pipe it to `wc -l`.** ⭐ **A label
   count is a function of the selector, so name the selector**, exactly as with the normaliser counts.

**Consequence: filling `tab:latent-use-pending` would change nothing in the compiled PDF.** The
table I was asked to prepare lives in an orphaned file. Either (a) `appendix.tex` is a staging draft
whose content must be merged into `main.tex`'s inlined appendix, or (b) the build is meant to
`\input` it and the `\input` line is missing. **This is a question for the user, not something I
should resolve by editing either file** — and it must be settled before any number is pasted, or the
work is invisible.

The numbers in 12.8 remain correct and ready; what is now uncertain is *where they go*.

### 13.7 Two more non-discriminating checks of my own, both returning a reassuring empty

- **`find /inspire/.../zhangjiaquan-253108540222 -maxdepth 6 -name appendix.tex -path '*lalaland*'`
  printed NOTHING for a file that demonstrably exists** — the path is 8 levels below that root, so
  `-maxdepth 6` could never reach it. I read the empty output as "no sibling copy exists". Re-run from
  `paper_ICLR27/` it finds all 4 copies instantly. **A depth-limited `find` that returns empty is
  evidence about the depth limit, not about the file.** Same family as the `-newermt` and mtime
  failures: the empty/zero result looked like the answer I expected.
- **`set -o pipefail` demonstration**: `find … -newermt '-90 minutes' | wc -l` prints `0` and, with
  pipefail, exits 1 — G2's exact failure mode reproduced on my own shell. Without pipefail the `0`
  is indistinguishable from a real zero. Every footprint check in this document now uses an ISO
  cutoff and reads the exit status.

---

## Round 14: orphan status proved from the ARTEFACT; two more of my counts corrected

### 14.1 The orphan finding, now proved positively (not by absence of an `\input`)

My round-13 argument was "no `\input` exists". G2 supplied a stronger witness and I verified it
independently — the strongest form is what the **built artefact** contains:

```
timeline : appendix.tex 22:56:31  ->  main.tex 22:56:49  ->  main.{pdf,aux,log} 22:58:00
main.log : 'appendix.tex' occurrences = 0
           distinct files opened = 4  (main.tex, fancyhdr.sty, iclr2027_conference.sty, natbib.sty)
build    : 31 pages, 0 'Undefined control sequence', 0 'Fatal error'
main.aux : 69 newlabels; 'latent-use-pending' = 0; 'app:latent-controls' = 0
           CONTROL 'tab:capability' = 6   (so the aux grep can find a built label)
main.pdf : text contains 'Same-loop backbone' 0, 'Training-mean latent' 0, 'Shuffled sampled' 0
           CONTROL 'RWKV' = 49            (so the PDF-text grep works)
```

**A rebuild of `main.pdf` happened at 22:58, two minutes after the 22:56 edits, and it never opened
`appendix.tex`.** The zero-undefined-macro count is the clincher: `\pending` is defined nowhere, so a
clean 31-page build is only possible if `\pending` never reached the compiler.

⛔ **RETRACTED (round 15): "an author / a third party / a live authoring session".** The **coordinator
of this session made the 22:56 edits** — it said so, and I verified it on content, not on timestamps:
its three edit signatures are in `main.tex` and absent from `appendix.tex` (`length collapse` 1/0,
`floor set by our truncation budget` 3/0, `+5.35` 3/0; positive control `LongBench` 16/8, negative
control 0). The edit fills the **Mamba LongBench** cell at `main.tex:368`/`:791`.

My evidence (no write-mode open by me, none of my numbers present; 0/83 of G2's scripts reference
either path) **ruled out me and G2 and identified nobody.** "Not us" plus a timestamp is not an
attribution, and I turned it into "a third party is actively editing". ⭐⭐ **This is the
mtime-cannot-attribute-authorship trap committed one step downstream of where I had just diagnosed
it** (§12.7) — the timestamp was real, the rebuild was real, and the actor was invented. G2 made and
retracted the identical inference. **Attribution needs a positive content or provenance signature, and
"who else could it be" is not one.**

The *load-bearing* facts are untouched by this retraction: the build genuinely does not include
`appendix.tex`, which is proved by the log/aux/PDF-text evidence above and does not depend on who
edited anything.

**Pre-22:56 bytes are not on disk**: both files are `??` untracked, the 5 `main.tex.bak_*` copies match
neither pin, and there is no `appendix.tex.bak*`. The coordinator holds the pre-edit hashes, but **a
hash verifies content without reconstructing it**, so the pins are not a restore path. Harmless here —
the edits were deliberate. Current state `4bd16294…`/`0c27670f…`; old pins `0b91c817…`/`d9ca82a2…`
were superseded **by the coordinator's own edit**, not by an unknown party.

**No merge is in progress** (G2 verified, and it decides where numbers go): of 74 appendix paragraphs
>200 chars, **0** appear verbatim in `main.tex` (positive control 74/74 in their own file). `main.tex`
has **no slot** for the five rows. So pasting numbers today changes nothing in the PDF, and there is
nowhere in the compiled file for them to go until an author creates the table there.

### 14.2 ⛔ My "27 appendix-only labels / 18 duplicated" was wrong twice over

Correct values (`app:` selector): **28 appendix-only, 17 duplicated**. My 27 came from
**hand-counting a 28-line printed list** — the list was complete, not truncated; I simply miscounted.
And my "18 duplicated" was the *main.tex `app:` total* (18) written into the slot where the overlap
(17) belonged. Full selector enumeration is now in §13.6. 27 is **unreachable under every selector**;
18 is reachable only under `app:|sec:`. G2's 68/25 and 28/17 both reproduce exactly.

⭐ **Never hand-count a printed list — pipe it to `wc -l`.** This is the third time one of my counts
landed one away from a reachable value (27 vs 28; 188 vs 171; the 433/436 `sides` axis). The
conclusion is unaffected: 28, 68 and every other cell support "orphaned".

### 14.3 ⛔ My `\newcommand` control was a loose substring count; G2's "0" is right

I wrote "control: the same grep finds all 45 other `\newcommand`s" and separately measured
`appendix.tex` at 11 — apparently contradicting G2's "appendix.tex defines no commands at all".
**G2 is right and my instrument was loose.** All 11 hits are `\renewcommand{\arraystretch}{1.10}`
table formatting; `grep -c newcommand` matches `renewcommand` as a substring. Tight counts:

| file | `\newcommand` | `\renewcommand` |
|---|---|---|
| main.tex | 13 | 1 |
| **appendix.tex** | **0** | 11 |
| math_commands.tex | 11 | 0 |

My "45" was `grep -rh newcommand *.tex | wc -l`, i.e. the same substring inflation across 4 files.
The claim that matters survives on a **tight** predicate: `\pending` has **0** definitions across all
`*.tex`/`*.sty`/`*.cls`, while the identical pattern finds `\method` in 3 files — so the control
discriminates and is not vacuous.

Note also `main.log` shows **0** opens of `math_commands.tex`: `main.tex` **inlines** its own
`\newcommand{\R}{\mathbb R}` at `:22` rather than inputting it. So the build genuinely depends on
exactly 4 files, and a "missing `\input`" is consistent with this project's style of inlining.

### 14.4 Standing conclusion on the deliverable

Measurement is **done and stable**: A0 71.17 / A1 5.33 / A2-A4 0.00, invalid 44/510/596/595/596 per
600, `prefix_replay=none`, refeed reported separately and known-artefactual. Every candidate scorer
repair moves A0 **down** and leaves the latent arms algebraically at 0.00, so **the headline does not
depend on any scorer decision**. What blocks *placement* is not measurement: the target file is not in
the build. The 22:56 `main.tex` edits were the **coordinator's** (verified on content signatures, not
mtime — see §14.2 retraction). **Neither agent should touch either `.tex` without the coordinator.**

### 14.5 Seed-44 refeed sciq landed at 23:08 (the n=0 cells filled, exactly as predicted)

The `full_refeed_seed44` sciq pair went from **n=0 to n=200** at 23:08, during this round — the third
time in this exercise that "files exist but rows do not" resolved itself without announcement. The
`commonsenseqa` pair is still 0 rows, and `race` is still absent, so **macro3 remains NOT ESTIMABLE
(1/3 tasks)**.

| cell | acc | invalid |
|---|---|---|
| refeed seed43 sciq A3 | 87.00 | 1/200 |
| refeed seed43 sciq A4 | 88.50 | 1/200 |
| **refeed seed44 sciq A3** | **87.50** | 1/200 |
| **refeed seed44 sciq A4** | **89.00** | 1/200 |

**A4 − A3 = +1.50 at both seeds** — a stable but very small gap, and it sits inside the refeed mode
that is already known to be a protocol artefact (`inject_into_cache` overwrites the prefill, so A4
reads the prompt once and A0 twice). It does **not** rehabilitate refeed as evidence of latent use.

**And this is not the independent replicate I was waiting for**: `example_id` overlap between seeds 43
and 44 is **200/200 for both arms**, so these are *paired on items*, not independent repetitions.
Cross-seed CIs from them are not independent samples — consistent with the known ~597/600 shared-seed
finding. The genuinely disjoint-pair contrast still does not exist.

## Round 15: the authorship retraction, and my footprint guard was broken on TWO axes at once

### 15.1 Retraction: "a third party is actively editing" (applied to §14.2 and §14.4)

The **coordinator** made the 22:56 `main.tex` edits. I verified that on **content**, not timestamps:
`length collapse` 1/0, `floor set by our truncation budget` 3/0, `+5.35` 3/0 between `main.tex` and
`appendix.tex`; positive control `LongBench` 16/8; negative control 0. The Mamba LongBench cell is real
at `main.tex:368`, `:791`, `:799`.

My evidence had only ever **excluded** me and G2 (no write-mode open by me; 0/83 of G2's scripts name
either path). ⭐⭐ **Excluding every actor I can see does not identify the actor.** "Not us" + a
timestamp became "a third party", which is the mtime-cannot-attribute-authorship trap I had diagnosed
two sections earlier. G2 independently made and retracted the same inference.

Unaffected by this retraction (they never depended on who edited anything): `appendix.tex` is absent
from the build (`main.log` 0 opens / `main.aux` 0 hits with control `tab:capability` 6 / PDF text 0 of
5 row names with control `RWKV` 49), and pre-22:56 bytes are not on disk.

### 15.2 ⛔⭐⭐ My footprint guard was simultaneously pointed at the wrong ROOT and the wrong DAY

The guard I had been reporting as "0 of 85 measurement files modified, must-match control 85/85" is
**two independent failures stacked**, and *either one alone* produces a clean-looking 0:

| axis | what I had | truth | effect in isolation |
|---|---|---|---|
| root | `scripts/eval/latentuse/latentuse_g{2_out,3}` | `laces-reasoning/latentuse_g{2_out,3}` (repo root) | glob matches nothing ⇒ **0/0**, prints as "0 modified" |
| date | `-newermt '2026-09-22 22:58'` | every file's mtime is **2026-09-23** | every file is newer ⇒ **110/110**, prints as "everything changed" |

The two errors point in **opposite directions**, so had I had only one of them I would have seen either
a false all-clear or a false alarm. What caught it was the **print-the-denominator-first** rule:
`denominator: 0 measurement files` with an explicit `exit 3` on zero candidates. ⭐ **A footprint guard
whose root is wrong is indistinguishable from a clean footprint** — the guard must therefore assert
that its candidate set is non-empty *and* that a deliberately-permissive cutoff returns **all** of it
(`2000-01-01` → 110/110) *and* that a deliberately-impossible cutoff returns **none** (`2027-01-01` →
0/110). Only after all three does `since 22:58 → 4/110` mean anything.

⛔ **The "85" in my earlier reports was also just stale.** The tree is **110** files now and it grew
while I was auditing it: 70 files at 19:00, 92 at 20:30, 106 at 22:48, 110 now. ⭐ **A denominator
copied from an earlier round is not a measurement**; every tally must re-enumerate.

### 15.3 uid cannot attribute authorship either

I tried `find ! -uid $(id -u)` as an ownership control: **0 / 124**. That is not exoneration — this
session's agents all run as the **same** user, so uid has zero resolving power here, exactly like
mtime. The only sound basis for "I did not write these" is that every command I ever issued against
those roots was a read (`json.load`, `wc -l`, `find`, `grep`), and **no filesystem attribute can
testify to that**. It is an argument from my own transcript, and I report it as such.

### 15.4 The 23:08 rewrite did NOT move a number I had published

`full_refeed_seed44/sciq__{A3,A4}.jsonl` were rewritten at 23:08 (468141 / 468478 bytes), i.e. after I
tabulated them. Recomputed from the current bytes: **A3 87.50 (175/200, invalid 1), A4 89.00 (178/200,
invalid 1), 200 distinct `example_id` each** — identical to the pinned table at §14.5. `commonsenseqa`
A3/A4 are still **0 bytes / n=0**, and `race` is still absent, so **refeed-seed44 macro3 stays NOT
ESTIMABLE (1/3 tasks)** and A4−A3 = +1.50 at both seeds is unchanged.

⭐ Worth stating plainly: a concurrent writer touched an evidence file I had already quoted, and the
only reason I know my number survived is that I **re-derived it from the new bytes** rather than
trusting the pin. A hash pin tells you *that* something changed, never whether your claim still holds.

### 15.5 ⛔⭐⭐ Round 15b: G2 caught my two denominators (110 and 124). Its diagnosis was wrong, and the truth is worse.

G2 flagged that my round-15 report quoted **110** and **124** for the same tree, and guessed "110 is a
few minutes stale — the very rule you just wrote". Reasonable, and **false**. Measured at one instant:

| scope | g2_out | g3 | combined |
|---|---|---|---|
| `.json`/`.jsonl` only | 39 | 71 | **110** |
| every file | 41 | 83 | **124** |

It is a **PREDICATE** difference, not a time difference, and both numbers were correct when written —
I just never said which predicate each belonged to. ⭐ **A denominator is meaningless without its
scope**; "4/110" and "6/124" describe the same tree at the same second.

**And the scope I had chosen was hiding changes.** The 14-file delta is 6 `.pt` latent caches, 2
`.txt`, 2 `.py`, 2 `.pyc`, 1 `.sh`, 1 `.log`. Two of those 14 changed after the cutoff and **my guard
could not see either**:

- `latentuse_g3/z_cache_full_seed44.pt` — 217911 B at **23:08**
- `latentuse_g3/g3_job_id.txt` — 4369 B at **23:14**

⭐⭐ A `.pt` latent cache **is** an evidence file in every sense that matters; it was excluded only by a
filename convention. So my "4 of 110 changed, all four accounted for" was **narrower than it sounded**:
at the honest scope it is **6 of 124**, and the two extra are the ones a reader would most want to know
about, because `z_cache_full_seed44.pt` is an *input* to the arms rather than an output.

Checked the obvious worry — is seed44's cache just a copy of seed43's? Both are exactly 217911 B, so
size cannot answer it. Hashes differ (`2062e8d5…` vs `8cd61914…`; negative control: full-vs-smoke also
differ), so the caches are **genuinely distinct** and the identical size is the fixed tensor shape.
`g3_job_id.txt` turns out to be G3's provenance notes, not measurements.

**Fixed in the instrument, with a control that reproduces the blind spot on demand.**
`footprint_guard.py` now audits **both scopes**, prints each with its own denominator, and explicitly
lists the changed files invisible to the narrow scope. New positive control **P4** injects a fresh
`z_cache_tampered.pt` and *requires* narrow to miss it and wide to catch it — if both saw it or both
missed it, the two-scope design would be pointless and P4 fails. Calibration is now **5/5**
(P1 wrong-root→VACUOUS, N1 known-good clean, P3 injected dirty uniquely named, P2 forced-clean
announced, P4 scope blind spot reproduced). Audit output: narrow 4/110, wide 6/124, 2 hidden.

⭐ **Reusable rule**: a suffix filter is a *scope choice*, and a clean verdict at one scope is not a
clean verdict at another. Quote every footprint number as **n/denominator-at-scope-S**, and make the
guard print the reconciliation itself rather than leaving it to the reader — which is exactly how G2
caught me, by noticing two numbers that could not both be the whole tree.

## Round 15c: G2's canned-string mechanism VERIFIED and made stronger — but its "422" needed a predicate, and my first verdict on it was premature

### 15.6 The 422 is real, and the row set is IDENTICAL across A2/A3/A4

G2 reported "A2/A3/A4 emit a prompt-independent canned string **422/600** vs 0/600 for A0/A1". My first
pass said it did not reproduce: I took the pooled modal string and got **175/1800**. ⛔ **That verdict
was premature and wrong** — the disagreement was a **predicate** difference, the same axis I had just
corrected G2 on, and I made the mirror-image error by testing exact equality when the claim was about a
shared opening. Enumerating the axis:

| predicate over `prediction` | pooled /1800 | A0 | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|---|
| `== pooled modal` (incl. U+FFFD tail) | 175 | 0 | 0 | 0 | 87 | 88 |
| `startswith("…Wall Street Journal.")` | 296 | 0 | 0 | 75 | 110 | 111 |
| **`startswith("The New York Times")`** | **1266** | **0** | **0** | **422** | **422** | **422** |
| `"New York Times" in s` | 1269 | 0 | 0 | 423 | 423 | 423 |
| contains U+FFFD | 1061 | 0 | 67 | 304 | 376 | 381 |

⭐ **G2's 422 is exactly right at the prefix predicate** and it lands on all three latent arms
simultaneously. My 175 was a different (stricter) question. **Neither number is the finding; the
predicate is part of the finding** — this is the 433/436 lesson recurring for the third time.

**The much stronger fact G2 did not state.** The 422 rows are not merely equal in *count*: they are the
**same rows**. |A2∩A3∩A4| = 422, |A2∪A3∪A4| = 422, and all three pairwise symmetric differences are
**exactly 0** on `(task, example_id)` keys, with key sets asserted identical across all five arms.
Negative control |A0∩A2| = |A1∩A2| = **0**, so the predicate is arm-specific. Spread over all three
tasks (race 62 / sciq 182 / commonsenseqa 178), so "prompt-independent" is justified, not an artefact of
one task.

**How deep the identity goes** (denominator 422): all three arms byte-identical **92/422**;
A3==A4 **207/422**; A2==A3 and A2==A4 **134/422** each. Per-row longest common prefix across the three
arms: min 19, median 67, mean 72.8, max 178, with **208/422** sharing ≥68 characters. Negative control
on the same rows for A0 vs A1: LCP min 0, **median 0**, max 3, mean 0.2. ⭐ So the latent arms agree on
~70 opening characters where the non-latent arms agree on none — the write produces a **prompt-ignoring
canned continuation**, and the *content* of the latent (mean vs deranged vs correct) barely perturbs it.

⚠️ **But canned ≠ invalid, and the gap matters.** Canned explains **419/596 = 70.3%** (A2), 419/595 =
70.4% (A3), 420/596 = 70.5% (A4) of invalid rows — a majority, not all. And **2–3 canned rows per arm
are NOT invalid**, so the two predicates cross. Any sentence of the form "the 0.00 is because the model
emits a canned string" must carry the 70% and the crossing, or it overclaims.

### 15.7 A1 computes the SAME latent as A4 and differs only in persistence — the arm contrast is clean

Independently confirmed on aligned `(task, example_id)` keys, 600/600: `A1.latent_norm ==
A4.latent_norm` **600/600**, while `prediction` differs **600/600**. `A3 == A4` elementwise 0/600 but
the two multisets are **equal** (a within-task derangement permutes, as designed), and negative control
`A2 == A4` is 0/600 (A2 is the training mean: **one** distinct value, 9.17766953, across all 600 rows).
A0 carries `latent_norm = None` on 600/600. ⭐ **This is the cleanest possible statement of the A1
control**: the adapter ran and produced the identical latent, so A0-vs-A1-vs-A4 isolates *persistence
of the write*, not *whether the latent was computed*.

### 15.8 Two corrections to G2's LaTeX numbers (its conclusion survives both)

- ⛔ Its positive control **"`normalsize` 7 hits"** is the **line-vs-occurrence** wrinkle again, now in
  its hands: `iclr2027_conference.sty` has **7 lines** containing `normalsize` but **9 occurrences**
  (`:113 :116 :118 :120 :122 :172 :173`, two of which carry it twice). The control still discriminates,
  so its conclusion holds.
- ⛔ **10.9589 is not in the `.sty`.** `grep` for `10.9589`/`10.96` returns **0**, and `baselineskip`
  appears on **0** lines. The file sets leading as `\def\normalsize{\@setsize\normalsize{11pt}\xpt\@xpt}`
  at `:173`. The 10.9589 is a **derived** quantity: 11pt × 72/72.27 = **10.958904** bp, matching at 4dp,
  with negative controls 10/10.5/11.5/12pt giving 9.9626/10.4608/11.4570/11.9552. ⭐ **Quoting a derived
  number as if it were read from a file is a provenance error even when the arithmetic is right** —
  anyone auditing by `grep` gets 0 hits and concludes the claim is fabricated.
- ✅ `thebibliography` **0 occurrences** confirmed, so the references really are `\normalsize`.

### 15.9 The seed-44 refeed cells filled again at 23:36 — now 2/3 tasks, still NOT ESTIMABLE

Fourth time this exercise that "files exist but rows do not" resolved itself mid-audit. The two-scope
guard caught it because the count moved 6/124 → **8/126** while I was writing the previous section:

| cell | n | acc | invalid |
|---|---|---|---|
| refeed s44 sciq A3 | 200 | 87.50 | 1 |
| refeed s44 sciq A4 | 200 | 89.00 | 1 |
| **refeed s44 commonsenseqa A3** (new, 23:36) | 200 | **60.50** | 0 |
| **refeed s44 commonsenseqa A4** (new, 23:36) | 200 | **59.50** | 1 |
| refeed s44 race A3/A4 (appeared, 23:36) | **0** | — | NOT ESTIMABLE |

`z_cache_full_seed44.pt` was rewritten in the same batch (23:36, 217911 B). **macro3 is still NOT
ESTIMABLE at 2/3 tasks** for both arms: `race__{A3,A4}.jsonl` now *exist* but hold **0 rows**.
⭐ **"the file appeared" is not "the cell filled"** — the earlier round's own lesson, and the reason the
guard reports bytes as well as mtime.

Note the sign: on commonsenseqa the refeed gap is **A4 − A3 = −1.00**, opposite to sciq's **+1.50**.
Consistent with the standing finding that the refeed contrast does not measure a stable property and
single-task readings must never be quoted alone. Nothing here touches any `none`-mode number, so the
headline (A0 71.17 / A1 5.33 / A2–A4 0.00) is untouched.

## Round 15d: ⭐⭐ G2 caught selection-on-the-outcome in my mechanism stat — and the selection-free replacement is a BETTER finding

### 15.10 The charge: my LCP was measured on rows selected FOR being canned

G2 flagged that "the latent's content barely perturbs the opening ~70 chars" was measured on the **422
canned rows only** — i.e. rows chosen *because* they are canned — and that the other 178 were unmeasured
by both of us. That is **selection on the dependent variable**, and the charge is correct. Measured:

| stratum | n | LCP median | LCP mean | ≥68 chars | all-3 byte-identical |
|---|---|---|---|---|---|
| canned (**selected**) | 422 | 67 | 72.8 | 208/422 | 92/422 |
| **complement (unselected)** | **178** | **49** | **70.6** | **85/178** | **62/178** |
| **all 600 (unconditional)** | **600** | **67** | **72.2** | **293/600** | **154/600** |

⭐ The sentence **survives** — the complement's mean LCP is 70.6 vs the selected 72.8, so the effect was
not manufactured by the selection — but I could not have known that before measuring, and I should not
have written it. Negative control A0-vs-A1 stays at median 0 / mean 0.1–0.2 / max 3 in **all three**
strata, so the contrast is not an artefact of the stratum either.

**Why the complement did not differ: it is also canned, with different strings.** Pooled over the three
latent arms, the 178 complement rows give only **111 distinct** generations out of 534, the top three
being `'"I\'m not going to be able to do that…'` (47), `'“The only way to get rid of a bad government…'`
(32), `'"I\'m not going to be able to do this for the rest of my life."'` (32). ⭐ **"Canned" is not one
string, it is a small pool of strings**, so keying the finding to *the* New-York-Times sentence
understated it. That is why the 422 was always a predicate-dependent number.

### 15.11 The selection-free statistic that should go in the paper instead

Unconditional over all 600 rows, no outcome selection anywhere:

| arm | distinct full | distinct 40-char openings | top-1 share | openings covering 80% | rows with a **cross-task** opening |
|---|---|---|---|---|---|
| **A0** (no write) | **563** | **556** | 1.8% | **436** | **26/600 = 4.3%** |
| **A1** (write reverted) | 273 | 160 | 17.8% | 53 | 315/600 = 52.5% |
| **A2** (mean latent) | 112 | **40** | 25.0% | **10** | 471/600 = 78.5% |
| **A3** (deranged) | 163 | 45 | 20.2% | 12 | 494/600 = 82.3% |
| **A4** (correct) | 161 | 43 | 20.3% | 12 | **502/600 = 83.7%** |

⭐⭐ **This is strictly better than the 422 on four counts**: (1) no selection on the outcome; (2) no
dependence on a chosen string or predicate — it is a property of the whole output distribution;
(3) it is **monotone across the arms** in the direction the mechanism predicts; and (4) the cross-task
column is a direct operationalisation of *prompt-independence*: a prompt-responsive arm must almost never
reuse an opening across tasks, and A0 indeed does so on only **4.3%** of rows while A4 does on **83.7%**.

⭐ It also places **A1 as intermediate** (160 openings, 52.5% cross-task) rather than with A0 — consistent
with the now-confirmed fact that A1 computes the **identical** latent (`A1.latent_norm == A4.latent_norm`
600/600) and differs only in whether the write persists. A1 is not a clean "no-latent" arm.

**Recommended paper sentence** (scoped, selection-free, with its denominators): *"Writing the latent
collapses the decoder's output distribution: over 600 prompts the no-write arm produces 556 distinct
40-character openings and reuses an opening across tasks on 4.3% of prompts, whereas the latent-writing
arms produce 40–45 and reuse across tasks on 78–84%. The effect is essentially independent of latent
content (mean, deranged and correct latents differ by <5 openings)."* Do **not** write "it emits a canned
string": canned emission accounts for ~70% of invalid rows, and canned-vs-invalid cross both ways
(canned-but-valid 3/3/2, invalid-but-not-canned 177/176/176).

### 15.12 G2's additions to the LaTeX corrections

Accepted: `11pt` appears on **2** lines and one is the commented-out `:172`, so a re-deriver must take
the live declaration from `:173`. ⭐ Its own framing of its error is the sharpest version: it *wrote* the
honest arithmetic to the coordinator but *said* the declaration "was 10.9589" in its summary — **a
derived value narrated as a read value**. Also accepted: A1 carries U+FFFD **67/600** while A0 carries
**0**, so U+FFFD cannot stand in for "canned".

## Round 15e: G2's bad pin (verified, corrected), and its repetition control does not reproduce as stated

### 15.13 ✅ The corrected concat pin, recomputed independently

G2 retracted `75575a71…` as an unreproducible digest. **Independently recomputed here, three selectors,
all agreeing:**

```
find -name '*.jsonl' | LC_ALL=C sort | xargs cat  -> b55f685b605aae84eea9c75ea1b5efac
depth-1 shell glob   *.jsonl        | cat         -> b55f685b605aae84eea9c75ea1b5efac
ls -1 | LC_ALL=C sort | xargs cat                 -> b55f685b605aae84eea9c75ea1b5efac
```

15 files, **3000** rows, mtimes 18:21 / 18:59 / 19:37 (5 files each), all before 20:00. Negative control:
A0-only concat gives `84754efc…`, so the digest does discriminate on file set. **Pin travels with its
selector**: `b55f685b605aae84eea9c75ea1b5efac = md5(cat of latentuse_g2_out/seed42/main/*.jsonl,
path-sorted, 15 files, 3000 rows)`.

⭐ G2's rule is the right one and I adopt it: **a pin is only a pin if you recompute it from a stated
selector before circulating it.** A wrong pin is worse than none, because it tells a future auditor the
data changed when it did not.

⛔ **But G2's premise about me is false: `75575a71…` never entered PLAN.md.** Checked with the
denominator printed — 17 deliverable files scanned, **0** contain it (positive control: `prefix_replay`
and 16 `canned`/pin lines found in the same file, so grep works), and it is absent from `/tmp/latentuse/`
too. It existed only in our messages. ⭐ **"You have recorded it" is an inference about my artefacts, not
an observation of them** — the same excluding-vs-identifying error as the authorship episode, and I would
have propagated a correction to a line that does not exist had I not grepped first. Nothing to correct.

### 15.14 ⛔⛔ RETRACTED IN FULL — G2's numbers are EXACT; I swept the wrong axis

> **Retraction (round 15f).** Everything below the rule in this subsection was wrong in its conclusion.
> G2's 49/0 and 548/544 are **exactly reproducible**. "Repeated text" means **the same output string
> appears on ≥2 rows** (a cross-row duplicate), not "a substring repeats inside one row". At that
> definition: A0 **49/0**, A1 387/346, A2 **548/544**, A3 512/509, A4 502/499 — both G2 cells hit on the
> nose, and the digest is stable across three keyings (raw text, `.strip()`ed text, and `generated_ids`
> tuples all give 49/0 and 548/544; ids shift only A4 by 2). Controls: duplicating one singleton moves
> A0 49→51 (delta exactly +2), adding a unique sentinel moves it 49→49.
>
> ⭐⭐ **This is the `minlen` error in my own hands, one level up.** I charged G2 with leaving `minlen`
> undefined, then swept **62 predicates over two axes** (character substrings, word n-grams, token-id
> n-grams, adjacency, type-token ratio; ×2 repeat counts) plus the root axis, the badness-flag axis and
> the pooling axis — **310 (predicate, arm) cells in which 49, 548 and 544 occur exactly 0 times** — and
> concluded "no setting reproduces it". Every one of those predicates was *within-row*. The axis I never
> varied was the one that mattered: **within-row vs across-row**. ⭐ **An exhaustive sweep of the wrong
> axis reads exactly like a refutation** — it even produced a confident falsifier ("G2's 548 exceeds what
> any minlen yields", true and irrelevant). The tell I had and ignored: **548 − 544 = 4 = A2's valid
> count exactly**, an arithmetic coincidence far too tight for a mismatched predicate. ⭐ **Chase the
> structural coincidence before publishing the non-reproduction.**
>
> ⭐ **My verified cross-check should have found it sooner**: G2's duplicate-row count and my own distinct
> -string count are *two readings of one fact*, tied by `distinct = (N − dup_rows) + n_groups`, which
> holds **5/5 arms** (A0 563 = 551+12; A2 112 = 52+60). My §15.15 table already contained G2's statistic;
> I did not recognise it because I read it as a different quantity.
>
> **What survives, and it is not what either of us said.** The direction I "confirmed" (32/32 estimable
> predicates, positive control 0/32, negative control 3/35) was confirming the wrong proposition. At G2's
> real definition the statistic **cannot support its sentence**, because duplication carries no
> information *within* an arm:
>
> | arm | P(inv) | P(inv\|dup) | P(inv\|uniq) | lift | n_dup | n_uniq |
> |---|---|---|---|---|---|---|
> | A0 | 0.073 | **0.000** | 0.080 | 0.00 | 49 | 551 |
> | A1 | 0.850 | 0.894 | 0.770 | 1.05 | 387 | 213 |
> | A2 | 0.993 | **0.993** | **1.000** | **1.00** | 548 | 52 |
> | A3 | 0.992 | 0.994 | 0.977 | 1.00 | 512 | 88 |
> | A4 | 0.993 | 0.994 | 0.990 | 1.00 | 502 | 98 |
>
> A2's **unique** rows are invalid **52/52 = 100%** — *higher* than its duplicated rows. So the
> 0/49-vs-544/548 contrast is carried entirely by the arms' marginal invalidity (44/600 vs 596/600), not
> by repetition. ⛔ **"Repetition becomes unscoreable only under the write" is not established**; the
> supported statement is "under the write, *everything* is unscoreable, duplicated or not." Both arms
> fail the discrimination test in the same direction (A0 P(inv|dup) − P(inv|uniq) = **−0.080**, A2
> **−0.007**) ⇒ ⭐⭐ **a cross-arm contrast of conditionals is not a within-arm conditional**; this is the
> no-discriminating-power trap (three 92–94% numbers on different denominators) with lift = 1.00 as the
> fingerprint. ⭐ **Compute the lift before calling a conditional a control.**
>
> **Net for the paper: do not cite this statistic as the decisive control.** The selection-free
> output-collapse table (§15.15) already says the defensible thing without conditioning on an outcome.
> G2's arithmetic is vindicated; my refutation is withdrawn; the causal reading is rejected on new
> grounds that apply to both of us.

---

G2 offered as its cleanest causal statement: *"A0 has **49** repeated-text rows of which **0** are
invalid; A2 has **548** of which **544** are invalid."* Implementing "some substring of length ≥ minlen
occurs twice" gives, at minlen=20:

| arm | invalid | repeated | rep ∧ invalid | rep ∧ VALID | P(invalid \| rep) |
|---|---|---|---|---|---|
| A0 | 44 | **19** | **3** | 16 | **15.8%** |
| A1 | 510 | 10 | 10 | 0 | 100.0% |
| A2 | 596 | **66** | **66** | 0 | 100.0% |
| A3 | 595 | 52 | 52 | 0 | 100.0% |
| A4 | 596 | 55 | 55 | 0 | 100.0% |

Neither 49/0 nor 548/544 appears, and **A0's repeats are not all scoreable** — 3 of 19 are invalid, so
the claim "0 invalid" is false at this definition. Sensitivity over the undefined axis:

| minlen | A0 rep/inv | A2 rep/inv |
|---|---|---|
| 10 | 147 / 15 | 172 / 172 |
| 15 | 79 / 7 | 133 / 133 |
| 20 | 19 / 3 | 66 / 66 |
| 30 | 12 / 3 | 54 / 54 |
| 40 | 4 / 1 | 16 / 16 |

⭐ **The claim's direction survives at every minlen** (A0's repeats are mostly scoreable, A2's are
uniformly not: P(invalid|rep) = 100% for A1–A4 at every setting) **but no setting reproduces G2's
numbers**, and G2's 548/600 exceeds what any minlen here yields. So the *finding* is sound and the
*figures* are not quotable: **"repeated text" is not a predicate until minlen and the notion of repeat
are stated** — the same underdetermined-definition trap as `strict` (8 implementations, 60.00–61.83).
The publishable version is the direction plus a stated definition, never the bare counts.

### 15.15 ✅ G2's L-dependence table reproduces at all 14 cells

Verified every entry of G2's prefix-length table — distinct openings for 5 arms × 7 values of L, and the
cross-task row counts for A0/A4 — **14/14 MATCH**, including my L=40 row and its L=48 row:

| L | A0 | A1 | A2 | A3 | A4 | A0 cross-task | A4 cross-task |
|---|---|---|---|---|---|---|---|
| 20 | 513 | 42 | 22 | 26 | 24 | 45 | 514 |
| 30 | 542 | 116 | 28 | 34 | 32 | 37 | 503 |
| **40** | **556** | **160** | **40** | **45** | **43** | **26** | **502** |
| 48 | 556 | 207 | 48 | 53 | 49 | 26 | 441 |
| 60 | 557 | 222 | 53 | 62 | 58 | 26 | 435 |
| 80 | 563 | 242 | 63 | 86 | 84 | 22 | 414 |
| full | 563 | 273 | 112 | 163 | 161 | 22 | 346 |

Ordering `A0 > A1 > max(A2,A3,A4)` holds at **every** L (7/7), so the mechanism is robust to L while the
levels move ~2×. ⭐ **The proposed paper sentence must therefore say "40-character openings" explicitly**
— it does — or a re-measurement at another L looks like a failure to reproduce. A1 is strictly
intermediate at every L, so the "A1 is not a clean no-latent arm" conclusion is unconditional.

⛔ Dropping the **70%** from the paper sentence, per G2: it is P(canned | invalid) for *one pinned
attractor*. The 178 complement rows are **equally invalid** (99.4% vs 99.3%) and canned with other
strings, so 70% measures the share using *that particular* string, not the share explained by canned
emission. Under a general predicate it is 84–91% — but see §15.14: a within-row "repeat" predicate needs
a stated definition before any number from it is quotable, and (per the §15.14 retraction) a conditional
like this one also needs its **lift** computed before it can be called a control at all.

### 15.16 ✅ BLOCKER CLEARED — seed-44 race filled at 23:59; macro3 now estimable in all 4 G3 cells

`full_refeed_seed44/race__{A3,A4}.jsonl` held **0 rows** for the whole session and forced
"NOT ESTIMABLE at 2/3 tasks". They now hold **200 rows each** (mtime 09-23 23:59). All 24 G3 files are
200 rows; **0 empty files remain**. Re-derived from the new bytes — never from the earlier pin:

| cell | arm | commonsenseqa | race | sciq | macro3 | invalid/600 |
|---|---|---|---|---|---|---|
| `full_none_seed43` | A3 | 1.00 | 0.00 | 0.00 | **0.33** | 593 |
| `full_none_seed43` | A4 | 0.00 | 0.00 | 0.00 | **0.00** | 596 |
| `full_none_seed44` | A3 | 0.00 | 0.00 | 0.00 | **0.00** | 595 |
| `full_none_seed44` | A4 | 0.00 | 0.00 | 0.00 | **0.00** | 596 |
| `full_refeed_seed43` | A3 | 62.00 | 68.00 | 87.00 | 72.33 | 3 |
| `full_refeed_seed43` | A4 | 60.50 | 68.50 | 88.50 | 72.50 | 3 |
| `full_refeed_seed44` | A3 | 60.50 | 68.00 | 87.50 | 72.00 | 2 |
| `full_refeed_seed44` | A4 | 59.50 | 67.00 | 89.00 | **71.83** | 4 |

Guard behaviour: a 0-row task reports `N/EST` and poisons its macro rather than silently averaging 2
tasks (pos ctrl `[50,None,60]` → N/EST; neg ctrl `[50,0,60]` → 36.67 — note the neg ctrl uses a **0.0**,
not a missing value, so the check distinguishes "absent" from "zero").

**Two conclusions, and the second matters more than the first.**

1. ⭐ The `none`-vs-`refeed` split is now complete and unambiguous: under `none` the latent arms are at
   **0.00–0.33 macro3 with 593–596/600 invalid**, under `refeed` the *same arms* are at **71.83–72.50
   with 2–4 invalid**. Two seeds each, 8 cells, no exceptions. The write is destructive under the
   training-matched policy and the `refeed` numbers are the known `inject_into_cache` artefact (A4 reads
   the prompt once, A0 twice).
2. ⛔ **A3 ≈ A4 everywhere**, and now with a replicate: `none` 0.33 vs 0.00 and 0.00 vs 0.00; `refeed`
   72.33 vs 72.50 and 72.00 vs 71.83 — **the derangement control's sign flips between seeds**
   (+0.17 then −0.17). ⭐⭐ **That is the signature of a null with no effect to measure, not of a
   near-miss**: at cos 0.99999 between a prompt's latent and another prompt's latent, A3-vs-A4 cannot
   discriminate, exactly as G2 said. So the paper must not present A3 as "wrong content" evidence.

⛔ **Still not an independent replicate.** Seeds 43 and 44 share **every example ID: 200/200 per task,
600/600 pooled, and in identical row order in all three tasks** (`identical_order=True` ×3; neg ctrl
race×sciq id overlap = 0, so the comparison can detect non-overlap). Both "200/200" and "600/600" are
correct — ⭐ **the figure is meaningless without its scope, so state per-task or pooled.** These are
re-samplings of one item set, not new draws. The cross-seed spread (≤0.67 macro3) therefore bounds
*sampler* noise only and **must not be reported as a confidence interval**. A genuinely independent
replicate needs different example IDs.

### 15.17 ✅ Final compliance audit — and one scanner false positive worth keeping

**5 measurement files changed inside my last edit window** (23:59–00:00: `full_refeed_seed44/race__{A3,A4}
.jsonl`, `full_refeed_seed44/summary.json`, `g3_checks_full_{none,refeed}.json`). ⭐ **mtime overlap is
not authorship** — every agent here shares one uid, so I established non-authorship by **mechanism**, not
by timing:

- Those rows carry `generated_ids`, `first_token_logits_top10`, `decode_seconds` (1.63 s), `new_tokens`
  =32 and `checkpoint_sha256` ⇒ they are **forward-pass artefacts**, unreachable without CUDA.
- **No GPU here, by three independent probes**: `nvidia-smi -L` → 0 bytes / exit 0; `torch.cuda.
  is_available()` → **False**, `device_count()` → **0**; `/dev/nvidia*` → only `nvidiactl`, no
  `/dev/nvidia[0-9]`. ⭐ **An empty output with exit 0 is ambiguous on its own** ("no devices" vs "probe
  broken") — it is only evidence once a second probe of a different kind agrees.
- The files' arrival matches the G3 runner's 23:08→23:59 cadence, and that runner is byte-unchanged
  (`35dc33a35424…`, identical for both runners).

**Write-scope scan over all 120 scratch scripts** for `open(...,'w'/'a'/'x')`, `write_text`,
`write_bytes`, `shutil.copy/move`, `os.remove/rename`, `json.dump`: **1 hit**, and it is a ⛔ **false
positive worth recording**. `fix_broken.py:214` is `out.write_text(...)` where `out = Path(args.out)` —
the target is **argparse-driven**, so no grep of the write site can tell where it lands. Resolved by
evaluating the default: `Path(__file__).with_name("examples_fixed.json")` → `/tmp/latentuse/
examples_fixed.json`, under `/tmp`. ⭐⭐ **A grep-based write-scope scanner is structurally blind to
indirection**: it can only see literals, so for any `args.*` target the path must be *resolved*, not
matched. The scanner's positive control fired correctly on an injected `open("/inspire/x.jsonl","w")`,
so the instrument works — it is the *predicate* ("the path appears at the write site") that is wrong, and
a clean scan would have been a false all-clear had the default pointed at the repo.

Secondary check on that same file: the **protected** `scripts/eval/latentuse/examples_fixed.json` is
still `b07591ee74532d0cee889e24c6a34526` (mtime 17:09:06, untouched since). Its key set is
`generated_by / macro_denominator / prompt_regeneration_note / rows_required_per_task /
selection_rationale / tasks` with **no score-like key anywhere** ⇒ it is a **selector/manifest**, not a
measurement file. ⭐ Worth stating because the HARD PROHIBITION is phrased by *suffix* (`*.json`), and
suffix is a poor proxy for "holds measurements" in both directions — this `.json` holds none, while
`z_cache_full_seed44.pt` holds pipeline state and is invisible to a `.json`/`.jsonl` guard (see below).

**Final state, all re-verified after the last edit:**

| artefact | md5 | status |
|---|---|---|
| `PLAN.md` (mine) | see §15.18 | 4163+ lines |
| `sanity_checks.py` (mine) | `891bf67a2e4d…` | exit 0, **TP=34/34 FP=0/9**, CALIBRATED AND PASSING |
| `footprint_guard.py` (mine) | `fd11408f2ef9…` | **5/5 controls behaved** |
| `g2_frozen_run_five_arms.py` | `35dc33a35424…` | unchanged |
| `g3_pinned_run_five_arms.py` | `35dc33a35424…` | unchanged (byte-identical to G2's) |
| `g2_aggregate.py` | `597de5116335…` | unchanged |
| `examples_fixed.json` | `b07591ee7453…` | unchanged |
| `appendix.tex` | `4bd162948b33…` | unchanged (22:56) |
| `main.tex` | `32f467308212…` | unchanged (23:15) |

Live footprint at both scopes: **narrow 9/115, wide 11/129**, with **2 files invisible to the narrow
scope** (`g3_job_id.txt`, `z_cache_full_seed44.pt`). ⭐ Quote the scope with the fraction: "9/115" and
"11/129" describe the same tree at the same second. `results/` 0 recent, `outputs/` 0 recent,
`outputs_paper/` absent. No GPU used, no cluster job submitted or stopped, no CUDA forward pass, no
measurement file written by me at either scope, no `.tex` edited, no scorer patched.

### 15.18 Self-pin (recomputed from a stated selector)

`md5(scripts/eval/latentuse/PLAN.md)` at the close of round 15f, over the whole file as written to disk:
recompute with `md5sum scripts/eval/latentuse/PLAN.md` from the `laces-reasoning` repo root. The digest
is deliberately **not** pasted here — a self-referential digest inside the file it describes can never be
correct after the paste, and a stale pin is worse than none (§15.13). Line count at close: see
`wc -l`; §15.14 is **retracted**, §15.16 clears the last blocker, §15.17 is the compliance audit.

## Round 15g: G2's headroom correction (accepted, refined) and the A1 scorer exposure (confirmed, +25.17)

### 15.19 ✅ G2 is right that my "no lift" verdict was NOT ESTIMABLE, not a null — accepted

My §15.14 retraction concluded "the latent arms fail the discrimination test in the same direction as A0."
⛔ **That was still wrong, one layer down.** G2 computed the headroom and I reproduce its bounds exactly:
with `(n_dup, n_uniq, n_invalid)` fixed, the feasible range of `d = P(inv|dup) − P(inv|uniq)` is

| arm | n_dup | n_uniq | V=valid | obs d | min d | max d | width | where |
|---|---|---|---|---|---|---|---|---|
| A0 | 49 | 551 | 556 | −0.0799 | −0.0799 | +0.8980 | 0.978 | **at min** |
| A1 | 387 | 213 | 90 | **+0.1241** | −0.2326 | +0.4225 | 0.655 | interior |
| A2 | 548 | 52 | **4** | −0.0073 | −0.0073 | **+0.0769** | **0.084** | **at min** |
| A3 | 512 | 88 | 5 | +0.0169 | −0.0098 | +0.0568 | 0.067 | interior |
| A4 | 502 | 98 | 4 | +0.0042 | −0.0080 | +0.0408 | 0.049 | interior |

G2's one-sided figures (A2 0.077, A3 0.057, A4 0.041, A1 0.423) are exactly my **max d** ⇒ MATCH. ⭐⭐
**A statistic whose feasible range is 0.05–0.08 wide cannot return a signal at any effect size**: with
only 4 valid rows in the entire arm, `d` is arithmetic about the denominator, not evidence about the
write. "Fails the test" and "not estimable" are different claims, and mine read as the former.

⭐ **Two refinements G2's one-sided `|bound|` hid.** (a) The range is **asymmetric**, so `|bound|` is the
wrong summary — A0's runs −0.08…+0.90. (b) **A2's observed −0.0073 sits exactly at its floor** `−V/n_dup
= −4/548`, i.e. *all four* of A2's valid rows are duplicated rows. Under `|·|` that reads as "no signal"
when it is in fact pinned at the most negative attainable value. Controls: with V=40 the bound widens to
0.77 and with **V=0 the bound is identically 0.0000** — the statistic cannot move at all, which is the
cleanest statement of why the latent arms are uninformative here.

**A1 is the only arm where the proposition is testable, and there it is positive** (+0.124 interior to a
0.655-wide range). I did not verify G2's Fisher p=0.0001 / permutation p=0.000 independently, so those
travel as G2's numbers, not mine; the sign and the headroom I do confirm.

### 15.20 ⛔⛔ G2's A1 scorer exposure is REAL — but the price is **+25.17**, not +23.00, and there are THREE defects

G2 reported A1 macro3 5.33 → 28.33 (+23.00) under a lead-strip. Direction and mechanism confirmed;
the magnitude needed a provenance gate first, and the gate is what found the rest.

⭐⭐ **First: the producer is `score_row` in the runner, NOT `scripts/eval/score_gen_match.py`.** That
scorer reads `generate`/`ground_truth`; our rows carry `prediction`/`gold`, so it never touched them. The
live gap is `g2_frozen_run_five_arms.py:340`, where `letter_of(g)` gets the **unstripped** string while
`:338` already strips the lead — the same shape as the known `score_gen_match.py` defect, in a different
file.

⭐⭐ **Second: the counterfactual must run behind a provenance gate.** Running
`score_row(..., letter_first=True)` mismatches **9/3000 `correct`** and **11/3000 `pred_choice_index`**;
`letter_first=False` gives **0/0/0 on all three fields**, and `summary.json → meta.args.letter_first`
records **`false`**. ⭐ **Had I quoted deltas off the `True` baseline, every number would have been
attributable to a flag rather than to the fix** — so the counterfactual now runs behind a hard gate that
raises unless the baseline reproduces the published `correct` ∧ `invalid` ∧ `pred_choice_index` (matching
one field is not provenance; `invalid` alone was 0/3000 **while `correct` was 9**).

⛔⛔ **RETRACTED (this section, first version): I wrote that "the on-disk runner does not reproduce the
published flags at its own default" and attributed the 9/11 mismatches to `score_row`'s "on-disk DEFAULT
(`letter_first=True`)". That is INVERTED.** The argparse is `action="store_true"` (`:125`), so the on-disk
default is **`False`**, which *agrees* with the published run; the 9/11 mismatches are the counterfactual
of turning the flag **ON**. Evaluated, not inferred: `parse_args([]).letter_first is False`,
`parse_args(["--letter_first"]).letter_first is True`, and `--letter_first` is absent from the recorded
`argv`. ⭐ **"The default doesn't reproduce the published run" is a far more alarming claim than the truth,
and I published it off a call site instead of evaluating the default** — same shape as form 12 in
`inert-validation-checks` (a default must be *evaluated*). Caught by G2, confirmed by me.

⛔ **Third defect (`"A:"` is the few-shot ANSWER MARKER and `letter_of("A:")` returns `"A"`), and my
"latent, not live" verdict on it is ALSO RETRACTED — the defect is LIVE in the published run.** The flag
gates only **one of two** paths to index 0: the `letter_first` branch, and the final `letter_of` fallback
at `:340`. Path attribution over the affected rows: **0** arrive via the `letter_first` branch, **18** via
the `:340` fallback, 1 via a `match_choice` stage. Flag sensitivity is **19 rows at `False` vs 26 at
`True`** — i.e. flipping the flag changes 19→26, **not 0→26**, which is exactly why "the flag was off" was
never a defence. ⭐ **Two code paths reached the same wrong answer, and I checked only the one I had been
reading.** Live numbers at the published flag: loose `^A:` fires **31/3000**, marker-only `^A:\n` **24**;
**19** of the 31 are scored option A (by arm **A0 4, A1 15**); **6** of those 19 were published as
`correct` because gold happens to be option A. So the defect **inflates A0, the arm carrying our
headline**. Caught by G2; every number above independently reproduced by me behind the gate.

**Gate-verified counterfactual (baseline reproduces all 3 fields, 0/0/0):**

| variant | A0 | A1 | A2 | A3 | A4 | dA1 |
|---|---|---|---|---|---|---|
| published | 71.17 | **5.33** | 0.00 | 0.00 | 0.00 | — |
| + lead strip | 71.33 | 29.17 | 0.00 | 0.00 | 0.00 | **+23.83** |
| + marker skip | 71.33 | 5.50 | 0.00 | 0.00 | 0.00 | +0.17 |
| **+ both** | 71.50 | **30.50** | 0.00 | 0.00 | 0.00 | **+25.17** |

A1 per task under both: cqa 8.50→30.00, race 7.50→20.00, sciq **0.00→41.50**.

⭐ **The headline is untouched and the latent arms are immune**: A2/A3/A4 move **+0.00 each** (zero rows
match any lead or marker — a *structural* immunity, not a small measured delta), A0 moves **+0.33**. So
A0 71.17 vs latent 0.00 survives all three repairs. Controls: stripping a lead that never occurs
("the response is") returns 5.33 and 0.00 **unchanged**; forcing every lead row correct gives **70.50**
as the upper bound (G2 said 61.50 — its ceiling is lower than mine, likely a different forcing rule; not
reconciled, and it does not affect the +25.17).

⭐ **Position sensitivity checked, per the standing 7.5pp warning**: V-LATE (fix only `:340`) and V-EARLY
(strip once up front) agree to **0.00 on macro3** here and differ ≤0.50 on a single task — because A1's
leads sit at string start. Not a general licence: the two positions still must be named when quoting.

**For the coordinator, this is the third scorer decision with a price tag.** The five-row table is safe
to place — the A0-vs-latent headline moves +0.33 and the 0.00s are structurally immune — **but the A1
cell is worth +25.17 under a defect-repair nobody has decided on**, and A1 was *already* flagged as not a
clean no-latent arm (`A1.latent_norm == A4.latent_norm` 600/600). ⛔ **A1 is doubly unsafe: wrong contrast
AND scorer-exposed.** Recommend the table carry an explicit note on the A1 row, or that A1 be reported as
a range.

### 15.21 ⭐⭐ The marker price: G2's **−0.50** and my **+0.17** are BOTH right — the disagreement is the *repair's definition*, not the measurement

G2 priced the marker skip at **A0 71.17 → 70.67 (−0.50)**; I got **+0.17**. Both computed at
`letter_first=False` behind the same gate (`correct=0 invalid=0 pred=0` over **n=3000**), and both
reproduce the published baseline exactly (A0 71.17, A1 5.33). So neither is a measurement error, and
"re-run it and see" would have run forever. The axis I had failed to vary is **the marker predicate**, and
my earlier "predicate width" probe was *inert*: I varied the letter class (`[AQ]:` vs `[A-H]:`) while both
alternatives kept the `\s*$` anchor, so both fell in the same class and printed the same `+0.17` — ⭐ **a
two-arm comparison where both arms are the same arm, which is why it looked like confirmation.**

Enumerated properly, with the fires count printed before every verdict:

| repair | n_fires /3000 | dA0 | dA1 | dA2/3/4 |
|---|---|---|---|---|
| A. marker-only line (`^\s*[A-H]\s*:\s*$`) → read line 2 | 24 | **+0.17** | +0.17 | +0.00 |
| B. loose prefix (`^\s*[A-H]\s*:`) → read line 2 | 31 | **−0.50** | +0.17 | +0.00 |
| C. loose prefix → strip the marker, stay on line 1 | 31 | +0.00 | **−1.00** | +0.00 |

⭐⭐ **The −0.50 is the price of an over-broad repair, not the price of the defect.** The loose predicate
fires on **7 extra rows** (31 − 24) whose first line is `A: <the actual answer>` — e.g. `A0 race 164`
`"A: (C) They had their pictures taken.Q: …"`. Those rows are **not defective**: the cascade already reads
the real answer through them, and **4/7 were already `correct`**. Reading line 2 throws the answer away
and lands on truncation debris (`'hve'`, `'h'`, `''`). Row-level arithmetic, which is what makes the two
numbers commensurable: **+1 defect row recovered − 4 already-correct A0 rows destroyed = −3 rows =
−0.50** at 100/(200·3) = 0.1667 macro3 points per row. ⭐ **"n_fires went up by 7" would have looked like a
*better* repair; it was 7 false positives.**

⛔⛔ **RETRACTED (previous paragraph of this section): "the defect is live, it favours A0, and repairing it
faithfully moves A0 by +0.17".** The sentence is wrong twice, and the mistake is that I carried the
**loose** predicate's arm split (A0 4 / A1 15) into a claim about the **faithful** predicate. Under the
faithful (marker-only) predicate, by arm, with `viaFallback` = credit through the `:340` marker path:

| arm | fires | viaFallback | published correct | accidental (correct ∧ via marker) | marker letters |
|---|---|---|---|---|---|
| A0 | **2** | 2 | **0** | **0** | `{'A': 2}` |
| A1 | **22** | 15 | **6** | **5** | `{'A': 22}` |
| A2/A3/A4 | 0 | 0 | 0 | 0 | — |
| ALL | 24 | — | 6 | 5 | — |

⭐⭐ **A0's accidental credit under the faithful predicate is ZERO, and every accidentally-credited row is
A1's.** So the defect does **not** favour A0. G2's direction is right and mine was inverted; caught by G2.

⛔ **And my own `accidental` column was a broken instrument in two ways, which is why I had 6 where the
mechanism gives 5.** (i) I defined accidental as *scored option A ∧ published correct*, which counts
`A0 race 31` — first line `"A: (A) don't get equal treatment…"`, path `match_choice/raw`. That row is the
model **genuinely naming choice (A)**; the credit is legitimate and the marker is incidental. Accidental
credit must be defined by **path** (`letter_of_fallback`), not by outcome. (ii) The strict predicate matches
**any** letter `A-H`, so a `"B:"` marker scores option **B** — "scored option A" is not even the right
target in general. Here all 24 markers happen to be `A`, which is precisely why the bug stayed invisible.
⭐ **An outcome-defined check agreed with a path-defined one on 23/24 rows; the one disagreement was the
only row that mattered.**

⭐⭐ **Replacement-rule sweep — the replacement is a free parameter, and it changes A1's SIGN.** Faithful
predicate held fixed (`n_fires=24` in every row of this table):

| replacement rule | dA0 | dA1 | dA2/3/4 |
|---|---|---|---|
| read line 2 | +0.17 | +0.17 | +0.00 |
| mark INVALID | +0.00 | **−1.00** | +0.00 |
| line 2 if non-empty else invalid | +0.17 | +0.17 | +0.00 |
| strip the marker, stay on line 1 | +0.00 | **−1.00** | +0.00 |
| concatenate line 1 + line 2 | +0.17 | **+0.83** | +0.00 |

**dA0 ∈ [+0.00, +0.17] — never negative across all five. dA1 ∈ [−1.00, +0.83] — opposite signs.** ⭐⭐ **So
there is no single signed price for this defect even after the predicate is fixed, and — G2's sharper
point — no arm-level *direction-of-favour* statement survives either, not merely no number.** Negative
control: all five rules applied to a marker that never occurs (`Z:`) give `+0.00` on all five arms.

⭐ **What IS quotable, because it holds under every faithful repair: the headline gap.** A0 − max(latent) =
71.17 baseline → **71.33 or 71.17**, i.e. it only widens or holds; the latent arms are immobile at 0.00
(`+0.00` in all 5 × 3 cells). That is the sentence for the paper.

Controls (D4): negative control — a marker that never occurs (`Z:`) under **both** predicates and **all
five** replacement rules gives `n_fires=0` and `+0.00` on all five arms, so the harness can produce a
no-op; positive control — the deliberately over-broad predicate B **does** move A0, and the row ledger
names all 9 touched A0 rows and all 5 that flip. ⭐ **The over-broad variant is itself the positive control
for "this check can fail."**

⭐ **Reporting rule this forces: a scorer-repair price is a function of (defect, predicate, replacement),
and quoting a delta without naming all three is unciteable.** "Marker skip: −0.50" and "marker skip:
+0.17" are both true sentences about different repairs. ⭐⭐ **Stronger corollary, from the dA1 sign flip:
when the replacement rule is undetermined, the defect has no signed price *and* no favoured arm — the only
citable quantity is one that is invariant across the whole rule family.**

### 15.22 ⛔ "No error in this audit moved against us" is FALSE as stated — but the worry behind it is real, and it needs two tallies not one

G2's point 4 withdrew its "one correction that moves against us" and then generalised: *no* error any of
us found moved against us, and that uniformity should make the user **discount** our error-hunting rather
than be reassured. The generalisation is falsifiable, so I tested it. At least two counterexamples:

1. ⛔ **WITHDRAWN — see §15.23.** I offered "A1 is not a no-latent control" on the strength of
   `A1.latent_norm == A4.latent_norm` on **600/600** rows. That equality is a **tautology of the source
   line** (both arms read the same cache key), so it witnesses nothing. **A1 survives as a valid control.**
   The counterexample is replaced by a *more* expensive structural finding: A1's validity reassigns
   **57–93% of the 71.17 collapse away from the latent** onto z-independent conv/FFN zeroing.
2. **The discrimination verdict is NOT ESTIMABLE, not passed.** Valid-row denominators per arm (/600):
   **A0 556, A1 90, A2 4, A3 5, A4 4**. With 4 valid rows the feasible range is 0.084 wide, so we do not
   get to claim the latent arms fail discrimination — we lose the claim. Against us, and it was G2's find.

A third, softer one: the lead-strip repair moves A1 from a 65.8-point anomaly to a 41.0-point anomaly
relative to A0. It **shrinks unexplained evidence that A1 is broken**, which is good for a clean table and
bad for honest accounting.

⭐⭐ **The resolution is that G2 and I were keeping one tally where there are two.** The **signed numeric
price** corrections do cluster in our favour (+23.83, +25.17, +0.17, +0.33, and now dA0 ≥ 0). The
**structural** findings run the other way and are the expensive ones (A1's contrast is wrong; A2/A3/A4 are
too sparse to support a control claim; the A3-vs-A4 sign flips across seeds). ⭐ **Merging them produces
exactly the false uniformity G2 flagged.** So the report must carry both tallies separately — and G2's
instinct to hand the user the uniformity itself, rather than only the conclusions, is the right instinct
even though its premise did not survive the check.

### 15.23 ⛔⛔ Counterexample 1 REFUTED: `latent_norm` is a provenance tautology. A1 IS a valid control — and that reassigns most of the effect (57–93%) away from the latent

G2 refuted my counterexample 1 and it is right. `latent_norm` is recorded as
`float(z_used.float().norm())` (producer `:775`), and **A1 `:722` and A4 `:745` read the same cache key**
`z_cache[f"{task}/{ex_id}"]` — literally the same expression. So 600/600 equality is a **tautology of the
source line**, recorded *before* `predict_states`/inject/restore. ⭐⭐ **No number of rows makes a
tautology a witness**, and "600/600, with a positive control showing A0-vs-A4 differ 600/600" *felt* like
overwhelming evidence while being about the wrong quantity. Same family as form 76 (an identity presented
as an adversarial design). ⛔ **G2 cited `run_five_arms.py:764/782/813`, which is NOT the producer** (that
file is md5 `6f9e561a…`; the producer of these rows is `g2_frozen_run_five_arms.py`, md5 `35dc33a3…`,
confirmed from `summary.json → meta.argv[0]`) — so I re-derived every line in the file that actually ran.
The claim survives the file swap; the line numbers do not.

**The channel audit checks out in the producer.** `inject_into_cache` is a *model method*
(`models/state_hijacking_dit.py:2489`), not a runner function — G2 named it as if it were local, and
`grep "def inject_into_cache"` over the runner returns **0**, so I followed the import target. There:
`state["recurrent_state"] = st.to(torch.float32)` is z-dependent; the conv/ffn zeroing is
`torch.zeros_like(cs)`, **z-independent**. `restore_principal_memory` (`:184`) writes `sv.clone()` back and
raises unless `torch.equal` holds. Coverage: `a1_layers_restored` = **32 on 600/600 for A1, 0 on 600/600
for all other arms**, and `num_hidden_layers = 32` in the checkpoint config. ⭐ **G2's own warning is
correct and I honoured it**: `state_basis` has rank 32 too, so confirming coverage off that tensor would be
a numeric collision faking a confirmation — the 32 must come from `config.json`.

⛔⛔ **MY INSTRUMENT FAILED AGAIN, in the very check meant to replace the tautology.** I printed
"`cache_fp_after_sha256` A1 vs A4 differ 600/600 = real witness". Two defects: (i) the narration was a
**hardcoded string that did not depend on the data** — the computed value was `0/600`, and I printed the
opposite; (ii) the `0/600` was itself **vacuous**, because that key is **absent from all 600 rows**
(`present 0/600`; the 33-key set is printed in full in the ledger) and `.get()` collapsed both sides to
`None == None`. Negative control: a **fictional** field name, `totally_made_up_field_xyz`, returns the
**identical** `0/600`. ⭐⭐ **A check that gives the same answer for a real-but-absent key and an invented
one has no discriminating power at all** — forms 31 and 55 (missing key ≠ default; existence before
comparison), and I hit both while auditing someone else for field misattribution.

**Witnesses that actually discriminate** (key existence verified *first*): `adapter_name`
(`inject_into_cache+restore_principal` vs `inject_into_cache`), `a1_layers_restored`, `cache_policy` and
`cache_policy_hash` — all **600/600 differ**; `generated_ids` differ **600/600** (the arms genuinely
diverged); `boundary_ids_sha256` differs **0/600** (identical inputs by construction — the correct
negative). The fingerprint I wanted does exist, **nested inside `cache_policy`**, where A1 and A4 share
`cache_fp_before_sha256` `48747a8e…` and differ on `cache_fp_after_sha256` (`a7531743…` vs `3aa67fc4…`).
⭐ **It was one level down, not missing — "absent at top level" and "not recorded" are different claims.**

⛔⭐⭐ **The refutation goes in the STRUCTURAL tally and is more expensive than the claim it replaces.** If
A1 is a valid control, its 5.33 is real and the 71.17 collapse decomposes as **A0−A1 = 65.83 (92.5%) from
conv/FFN zeroing alone, z-independent**, and **A1−A4 = 5.33 (7.5%) marginal to latent content**. So the
defensible sentence is **"the adapter's cache handling is destructive"**, not "the latent write is
destructive" — a weaker claim than the table implies.

⛔ **But that split is not citable either, and here I extend G2 rather than accept it.** Under the
undecided scorer repair the same decomposition gives **A0−A1 = 41.00 (57.3%)** and **A1−A4 = 30.50
(42.7%)** — the z-independent share **swings 92.5% → 57.3%, i.e. 35.2 pp, on a repair nobody has chosen**.
⭐ **So "~92% is z-independent" is a point estimate of an undetermined quantity.** The citable form is
*"the z-independent share is the majority under every repair we tested (57–93%)"*. G2's own guard also
holds and is tighter: **A4 has 4 valid rows /600**, so only the **order** A1 > A4 is safe, never the split.

⭐ Also from G2, and accepted: its "RNG channel OPEN" verdict came from the `sampler` field, which records
the **latent DDIM sampler** (`ddim_cfg(steps=100,cfg=3.0)`), not the token decoder — token decode is
`torch.argmax` (producer `:487`) under `@torch.no_grad()`, so the channel is closed. **A field
misattribution that printed a verdict anyway** — the same shape as my two failures above, and the third
time in this exchange that a *field-identity* error produced a confident wrong answer.

⭐⭐ **G2's sharpening of the pattern is better than mine and I adopt it.** All three field errors share a
second property: each field was read **without first asking what writes it**. `latent_norm`'s writer is one
line; looking at the write site before the read site made the tautology visible immediately. **"Locate the
write site before quoting the field" is cheaper than any negative control** and would have caught all three.

### 15.24 ⛔⛔ The decomposition is CENSORED — and the uncensored metric delivers a STRONGER negative result: A3 and A4 are indistinguishable

G2's censoring objection is correct and it defeats my own 57–93% range as a *quantity*. Verified: A4 is
**0/200 correct on all three tasks** in **all four** scorer variants, as are A2 and A3. Since the two shares
sum to 100% by construction, the latent share is a **lower** bound and the aux share an **upper** bound —
⭐ **and the favourable end is the one we would be quoting.**

⭐⭐ **Worse than G2 said: A4 is not merely at zero, it is far BELOW chance.** Chance is 20.0% (cqa, 5
choices) and 25.0% (race, sciq, 4 choices), so A4 sits **20–25 pp below random guessing**. Accuracy
saturated long before it reached 0, and "how far below" is a real unmeasured quantity. ⛔ **G2's
corroborating metric is also near-saturated**: A4's `invalid` is **596/600**, four rows from its own
ceiling, so "corroborated by invalid rates" leans on a second censored measure, not an independent one.

⛔ **G2's margin arithmetic mixes variants.** It quotes A1 = 30.17 against a flip point
(A0+A4)/2 = 35.58, but 35.58 uses the **published** A0 = 71.17 while 30.17 is a **repaired** A1 — a repaired
A1 must be paired with the repaired A0. Paired within-variant:

| variant | A0 | A1 | A4 | flip = (A0+A4)/2 | headroom | aux share |
|---|---|---|---|---|---|---|
| published | 71.17 | 5.33 | 0.00 | 35.58 | +30.25 | 92.5% |
| lead only | 71.33 | 29.17 | 0.00 | 35.67 | +6.50 | 59.1% |
| marker only | 71.33 | 5.50 | 0.00 | 35.67 | +30.17 | 92.3% |
| both | 71.50 | 30.50 | 0.00 | 35.75 | **+5.25** | 57.3% |

So the tightest margin is **+5.25**, not +5.42, and the ordering holds in **4/4** variants. G2's conclusion
is unchanged and its direction is right; only the pairing needed fixing.

⭐⭐⭐ **An UNCENSORED metric exists, and it answers the question accuracy could not.** Every row records
`first_token_logits_top10`. The renormalised top-10 entropy is **not** saturated (**0.000** of rows within
0.001 of the `log 10 = 2.3026` cap in every arm; A0 min 0.3839, A4 min 1.9641) and it **reproduces the
uncensored part of the accuracy ranking** (A0 1.6985 < A1 1.8681 < A2 2.1013 < A3 2.1278 < A4 2.1280).
Paired over the same 600 rows, with a genuine negative control (A4 vs itself = exactly `+0.000000`,
600 ties):

| pair | mean Δentropy | 95% CI | x<y rows |
|---|---|---|---|
| A0−A1 | −0.1697 | [−0.1943, −0.1451] | 428/600 |
| A1−A4 | −0.2599 | [−0.2717, −0.2480] | 594/600 |
| A2−A4 | −0.0267 | [−0.0283, −0.0252] | 581/600 |
| **A3−A4** | **−0.0002** | **[−0.0022, +0.0019]** | **300/600** |
| A0−A4 | −0.4295 | [−0.4522, −0.4069] | 591/600 |

⭐⭐ **A3 vs A4 — deranged latent vs the correct one — has a CI that straddles zero** (permutation
p = 0.861; wins/losses exactly 300/300; `generated_ids` identical on 298/600 rows). ⭐⭐⭐ **This is a
stronger negative result than the censored accuracy could support**: on the floored metric "0.00 vs 0.00"
is uninformative by construction, whereas here a metric with demonstrated power — it separates A1−A4 and
even A2−A4 with CIs far from zero — **finds no difference between the correct latent and a wrong one**. The
write is not merely unhelpful; **its content does not matter.**

⛔ **Caveat on the new metric, stated before anyone quotes it:** it is a **top-10 truncated** entropy, so it
cannot see mass outside the recorded ten tokens and is **ordinal, not absolute**. Its validity rests on the
rank agreement above plus the A4-vs-A4 null and the A0−A4 positive control.

**Net for the paper.** G2's downgrade is right and I go further: *"the adapter's cache handling is
destructive rather than the latent write"* travels as **directional and caveated** (tightest margin +5.25,
censored in our favour, shares are bounds not estimates). The claim that is **not** censored, has a real
denominator and a working instrument is the sharper one: ⭐ **A3 ≈ A4 on the uncensored first-token
distribution — swapping in a different prompt's latent changes nothing measurable.**

---

## §15.25 The A3 ≈ A4 headline is withdrawn, but not for G2's reason — and my determinism control was a tautology hiding a real noise floor

G2 says: do not adopt A3 ≈ A4, because the encoder is collapsed, so the "deranged" latent is
essentially the same vector as the correct one. **The conclusion is right and I withdraw the
headline.** The stated *mechanism* is wrong, and G2's own numbers refute it. Separately, the
control I built to defend the headline was **inert in the same way as my retracted `latent_norm`
witness**, and fixing it surfaced the finding that actually kills the claim.

### 15.25.1 Collapse reproduces exactly (independent recomputation)

`/tmp/latentuse/collapse.py` on `latentuse_g2_out/seed42/z_cache_main.pt` (600 entries, `(1,32)`, bf16):

| task | ‖mean‖/mean‖z‖ | pair cosine | rel ‖zi−zj‖ | bit-distinct |
|---|---|---|---|---|
| commonsenseqa | 0.99999565 | 0.99999142 | 0.004403 | 200/200 |
| race | 0.99999586 | 0.99999154 | 0.004351 | 200/200 |
| sciq | 0.99999565 | 0.99999124 | 0.004393 | 200/200 |

Controls on the collapse statistic: iid Gaussian same shape → **0.0768**; identical rows → **1.0000**.
The statistic can therefore distinguish collapsed from non-collapsed. Matches G2 to 8 decimals, and
matches the independently recorded `laces-latent-collapse-champion` value 0.99999565 — this is
**corroboration of a known property of this checkpoint, not a new discovery.**

G2's point about the runner's guard is correct and worth keeping: `g2_frozen_run_five_arms.py:738`
asserts `torch.equal(z_used, z_recip)` is False, true on 600/600, which is **fully compatible with
cosine 0.99999**. ⭐ **A bit-inequality guard cannot detect a numerically tiny no-op.**

### 15.25.2 ⛔ G2's "direction not distance" mechanism is refuted by G2's own 1.1×

G2 explains the A3 null as: equal distance, but A2's displacement lies along directions the read-out
responds to (49.9% vs 26.2% in the top-3), giving "146× more entropy movement per unit dz."

I reproduce the geometry: top-3 energy **A2 0.452/0.536/0.482 vs A3 0.277/0.286/0.271** — G2's
49.9/26.2 confirmed. I reproduce the distances: per-recipient-row rel dz **A2/A3 = 1.040/1.065/1.064×**
(`/tmp/latentuse/dz.py`) — G2's 1.1× confirmed.

But the 146× does not survive. It is computed from **signed** means; per-row **magnitude** gives
**1.4×**, and sensitivity per unit dz is **A2 5.826 vs A3 4.384 = 1.0×** (`/tmp/latentuse/dose2.py`).
The two statistics answer different questions and G2 quoted the one that flatters the mechanism.

**The dose-response test settles it.** If A3 ≈ A4 were caused by the displacement being too small,
rows with larger dz would move more. Using the **recorded `donor_id` field** (present 600/600 — no
reconstruction needed, and better evidence than my rebuilt permutation):

| task | n | Spearman ρ(dz, \|ΔH\|) | p | dz spread |
|---|---|---|---|---|
| sciq | 200 | −0.0909 | 0.200 | 4.2× |
| commonsenseqa | 200 | −0.0775 | 0.275 | 2.9× |
| race | 200 | −0.0468 | 0.510 | 2.8× |
| **pooled** | **600** | **−0.0687** | **0.093** | **4.9×** |

Across a 4.9× spread in displacement there is **no positive dose-response** — the point estimate is
*negative*. So "the derangement is too small to matter" is not the mechanism. ⭐ **The honest
statement is that the read-out is flat along the displacements a derangement can produce, and the
reason A3 ≈ A4 is not that the latents are too close — it is that direction and distance both fail to
predict the response.**

### 15.25.3 The real structure: A3's effect is SIGN-SYMMETRIC, not absent

600 paired rows per comparison (`/tmp/latentuse/signed.py`):

| pair | n | signed mean | mean \|Δ\| | \|signed\|/mean\|Δ\| | up | dn | gen_ids same |
|---|---|---|---|---|---|---|---|
| A3−A4 | 600 | **−0.000173** | **0.019209** | **0.0090** | 300 | 300 | 298/600 |
| A2−A4 | 600 | −0.026737 | 0.027044 | 0.9887 | 19 | 581 | 221/600 |
| A1−A4 | 600 | −0.259855 | 0.260156 | 0.9988 | 6 | 594 | 0/600 |
| A0−A4 | 600 | −0.429542 | 0.430706 | 0.9973 | 9 | 591 | 0/600 |

Sign-symmetry controls: randomising the signs of A2's deltas collapses its signed mean −0.0267 → −0.0015
while magnitude is unchanged (**positive control: the statistic detects systematic effects**); doing the
same to A3 changes nothing, because it is already symmetric (**negative control**).

So the donor latent **does** change the output — per-row magnitude 71% of A2's, and it alters the
generated sequence on **302/600** rows — but in a direction-free way: Wilcoxon on direction p=0.924,
while bootstrap CI on magnitude is 0.0192 [0.0179, 0.0206], excluding zero. ⭐ **"No measurable
difference" was the wrong summary of my own data: the correct summary is "an effect with no
consistent sign."** My headline sentence *"its content does not matter"* asserted the stronger claim
and is withdrawn.

### 15.25.4 ⛔⛔ My A4-vs-A4 "null at exactly 0.000000, 600 ties" was a tautology

I reported it, and G2 independently "verified" it, as evidence the instrument was sound. It reads
**one file twice**: `deltas("A4","A4")` opens `{task}__A4.jsonl` for both sides, so 600/600 ties and
exactly `+0.000000` are **true by construction** — the identical failure mode as my retracted
`A1.latent_norm == A4.latent_norm` witness, committed **in the very round where I had adopted
"locate the write site before quoting the field" as a rule.** It cannot fail, so it was never a control.
That two independent agents both endorsed it is the reusable part: ⭐ **agreement between two readers
of the same tautology is not replication.**

### 15.25.5 ⭐⭐ The genuine control shows decode is NOT reproducible, and the noise floor is comparable to the A3 effect

A real determinism control exists in the tree: `smoke/` is a **separate process invocation** sharing 6
example_ids per (task, arm) with `main/`. Config-identity check across 72 pairs × 11 fields found one
mismatched field (`new_tokens`, 2/72 — a generation-length cap that cannot affect a first-token
distribution computed before any token is emitted; split confirms the drift is the same in both groups).

On the **70 config-matched pairs** (`/tmp/latentuse/determ.py`, `/tmp/latentuse/noise2.py`):

- identical entropy: **22/70** — decode is **not** bit-reproducible across processes
- mean \|ΔH\| = **0.010925**, p95 = 0.0295, **max = 0.0841**
- first-step argmax token identical: 71/72

Against the A3-vs-A4 per-row effect of 0.019209: the noise floor mean is **0.58×** that effect and its
max is **4.38×** it. ⛔ **My script printed the conclusion "decode is reproducible across processes,
so nonzero A3-vs-A4 deltas are caused by the donor latent" — that sentence was hardcoded narration
and its own numbers say the opposite.** Third occurrence of narration-not-derived-from-data in this
work; the first two are logged in §15.23.

Consequences, stated precisely:
- the **per-row magnitude** claim (mean \|ΔH\| = 0.0192, and "302/600 sequences changed") is **not
  separable from run-to-run drift** with a floor estimated at n=70 and p95 0.0295;
- the **paired signed mean** (−0.000173) is **unaffected**, because zero-mean drift cancels in a
  paired mean over 600 rows — so the *sign-symmetry* finding survives and the *magnitude* finding does not;
- the A2/A1/A0 separations (0.027 / 0.260 / 0.431) survive: 2.4×/23×/38× the floor mean, and 1.0×/8.8×/14.6× its max. **A2's margin over the floor is the thinnest and should not be quoted as precise.**

### 15.25.6 Net

| claim | status |
|---|---|
| encoder collapsed (cos 0.99999, 32-dim), derangement is a near-no-op in latent space | ✅ holds, controls both ways, reproduced independently |
| bit-inequality guard cannot detect a tiny no-op (`:738`) | ✅ holds |
| A3 ≈ A4 ⇒ "latent content does not matter" | ⛔ withdrawn (underdetermined, and my own data show an unsigned effect) |
| A3's effect is sign-symmetric: p=0.924 direction, 300/300 up/down | ✅ holds (paired mean immune to drift) |
| A3 per-row magnitude 0.0192, 302/600 sequences changed | ⚠️ not separable from a 0.0109-mean noise floor |
| G2's "146× per unit dz ⇒ direction not distance" | ⛔ refuted: signed-vs-magnitude artefact; sensitivity ratio is 1.0× and dose-response ρ = −0.069 |
| A4-vs-A4 determinism null | ⛔⛔ tautology, one file read twice |
| decode reproducible across processes | ⛔⛔ false: 22/70 identical, max drift 0.084 |

**What survives with working instruments**, unchanged from G2's list plus one: the sign A1 > A4; the
refusal of any percentage split; the encoder-collapse finding; and **the sign-symmetry of A3** — an
effect with no consistent direction, which is a weaker but defensible version of what I wanted to claim.

### 15.25.7 Independent replication on seed 44 (G3's run, finished during this round)

The footprint guard flagged 11 files changed after cutoff, **all under `latentuse_g3/`** (G3's seed-44
work); every file §15.25 reads is under `latentuse_g2_out/seed42/` and is unchanged. Two of the new
files are A3/A4 row sets, i.e. a free replication of the one claim that survived.

`/tmp/latentuse/seed44.py` on `latentuse_g3/full_refeed_seed44/`:

| task | n | signed mean | mean \|Δ\| | up/dn | gen_ids same |
|---|---|---|---|---|---|
| sciq | 200 | −0.002836 | 0.019857 | 95/105 | 128/200 |
| commonsenseqa | 200 | −0.000002 | 0.016154 | 97/102 | 121/200 |
| race | 200 | −0.001948 | 0.019422 | 93/107 | 132/200 |
| **pooled** | **600** | **−0.001595** | **0.018478** | **285/314** | **381/600** |

Signed-mean 95% CI **[−0.003551, +0.000299]** straddles zero; Wilcoxon direction p = 0.157
(seed 42: p = 0.924). ⭐ **The sign-symmetry finding replicates on an independent seed**: a real
per-row effect (magnitude 0.0185 vs seed 42's 0.0192) with no consistent direction.

⚠️ **Protocol caveat, stated because it would otherwise inflate the replication.** The seed-44
directory is `full_refeed_*` (`prefix_replay=refeed`); seed-42 `main/` was `prefix_replay=none`. Per
the known refeed artefact these are **different protocols**, so what replicates is the *pattern*
(symmetric sign, magnitude ~0.019), not the absolute numbers. A same-protocol seed-44 run would be
needed before quoting a cross-seed interval, and per the recorded derangement-collision property,
cross-seed CIs on this quantity are not independent repetitions in any case — report mean ± range.

---

## §15.26 Sign-symmetry is algebraically pinned — CONCEDED and withdrawn; my noise-floor objection is worse than G2 thinks; G2's retrieval bound is right and its threshold survives a harder calibration

### 15.26.1 ⛔⛔ WITHDRAWN: A3's sign-symmetry was a tautology of the permutation

G2's argument: A3's donor map is a permutation of the same index set, so
`Σᵢ (z[donor(i)] − z[i]) = Σⱼ z[j] − Σᵢ z[i] = 0` **identically**, which pins any signed
cross-row statistic with a linear component to zero *before the model is consulted*.

Verified on 3/3 tasks (`/tmp/latentuse/zerosum.py`), from the **recorded `donor_id`**:

| task | permutation? | fixed points | cycle structure | ‖Σ displacements‖ (fp64) | mean‖d‖ |
|---|---|---|---|---|---|
| sciq | True | 0 | single 200-cycle | **0.000e+00** | 0.040681 |
| commonsenseqa | True | 0 | single 200-cycle | **0.000e+00** | 0.040446 |
| race | True | 0 | single 200-cycle | **0.000e+00** | 0.039876 |

Model-free demonstration with 200 random linear probes, using the **training** mean from argv
(`g1_mean_latent.pt`, ‖·‖ = 9.178606) for A2:

| task | random-probe \|signed\|/mean\|·\| A2 | A3 |
|---|---|---|
| sciq | 0.6953 | **0.000000** |
| commonsenseqa | 0.6470 | **0.000000** |
| race | 0.6814 | **0.000000** |

An arbitrary linear probe reproduces my measured 0.9887-vs-0.0090 contrast with **no model
involved**. ⭐⭐⭐ **A permutation-based control has a zero-sum displacement field, so any signed
cross-row statistic on it is pinned to zero by algebra. "An effect with no consistent sign" is what
the design must produce, whatever the read-out does.** My sign-randomisation controls could not catch
this: they test whether the *statistic* detects systematic effects, not whether the *design* admits one.

**I checked the one loophole and it closes.** Entropy is *not* linear, so the measured value is not
forced to be exactly 0 — and indeed it isn't (sciq +0.00031264, cqa −0.00042104, race −0.00040921,
pooled −0.00017254 vs the linear prediction of exactly 0). But the pooled 95% CI is
**[−0.00220, +0.00188]**, which contains zero, so **the nonlinear residual is not distinguishable
from the algebraic zero** (`/tmp/latentuse/nonlin.py`). The statistic has no power here either way.
⛔ **Sign-symmetry is withdrawn as a finding, and the seed-44 "replication" in §15.25.7 replicates the
algebra, not a property of the model.** That is my second withdrawn headline in two rounds.

For the record of what the identity does *not* cover: it constrains signed sums only. Random-probe
`mean|·|` gives A3/A2 ratios of 0.939/0.994/0.906 — **not** pinned to zero. So the *magnitude* claim
is not killed by algebra; it is killed by the noise floor, which is a separate objection (below).

### 15.26.2 ⛔ My own common-mode defence FAILED, in the direction against me

G2's point 2 is correct: `smoke/` vs `main/` is **cross-process**, so my 0.0109 mean / 0.0841 max
bounds a different quantity than the within-process paired noise that A3-vs-A4 actually involves —
a cross-process floor is an **upper bound**, not an estimate.

I tried to rescue the magnitude claim by arguing the drift is common-mode. Mechanism looked right:
`decode_arm` deep-copies **one shared `base_out` prefill** per example and `for arm in arms` sits
*inside* `for ex`, so with `prefix_replay=none` the boundary input is `input_ids[:,-1:]`, identical
across arms — shared-prefill noise should cancel in a paired difference.

**It does not.** Paired-difference drift is *larger* than absolute drift on 8/9 cells
(`/tmp/latentuse/withinproc.py`), e.g. sciq A3−A4 absolute 0.013851 vs paired-difference **0.019748**;
race A3−A4 0.030260 vs **0.038740**. ⭐ **The noise is not common-mode, so G2's "upper bound"
concession does not help me — the paired quantity I quote is the one carrying the most drift.**
I cannot measure a within-process floor from these files (no repeat invocation exists), so the
honest status of the A3 magnitude claim is **UNDETERMINED, leaning dead** — not rescued.

The same correction applies in G2's favour where it noted: my A2 caveat ("only 1.0× the floor's max")
was also measured against the cross-process floor. But since paired drift is *larger*, A2 is **not**
in better shape than I said; if anything the caveat should stand.

### 15.26.3 ✅ G2's retrieval bound reproduces — and its threshold survives a harder calibration than it ran

`/tmp/latentuse/retrieval.py`, querying z(seedB, prompt i) against all 200 same-task latents of seedA:

| pair | sciq | commonsenseqa | race |
|---|---|---|---|
| s43→s42 | 1/200, rank 98.5 | 0/200, rank 103.0 | 2/200, rank 99.7 |
| s44→s42 | 1/200, rank 97.4 | 0/200, rank 101.3 | 1/200, rank 96.7 |
| s44→s43 | 0/200, rank 98.5 | 0/200, rank 99.9 | 0/200, rank 101.6 |

**Pooled 5/1800 = 0.0028 vs chance 0.0050**; mean ranks 96.7–103.0 vs chance 100.5. POS control
(self-retrieval) **200/200 = 1.0000**. Confirmed.

⭐ **But the calibration threshold is a function of the noise you assume**, and my first attempt
disagreed with G2 by 10× (I detected only above 5%; G2 says 0.5%). Resolved — I was
**double-counting noise**, adding synthetic sampling noise *on top of* two independently sampled
caches. Measured cross-seed dispersion is 0.004316 (G2's 0.44%, confirmed). Sweeping the assumed
noise (`/tmp/latentuse/calib.py`, top-1 on sciq):

| injected content | noise=0 | noise=0.10% | noise=0.44% synthetic | **real cross-seed pair** |
|---|---|---|---|---|
| 5.0% | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.0% | 1.000 | 1.000 | 0.020 | 0.995 |
| **0.5%** | 1.000 | 0.525 | 0.015 | **0.975** |
| **0.2%** | 1.000 | 0.070 | 0.000 | **0.140** |
| 0.1% | 1.000 | 0.085 | 0.000 | 0.015 |

The defensible column is the last one — the real cross-seed pair, which is the exact comparison the
bound is made on. It detects content at 0.5% and fails by 0.2%, so ⭐ **G2's ~0.1–0.2% ceiling on the
prompt-dependent component is right, and my 5% figure was the artefact.** ⭐⭐ **"Detects at 0.5%" vs
"detects at 5%" was the same instrument under different assumed noise — a calibration threshold must
be quoted with the noise model it was measured under.**

Seed coupling confirmed from the producer, not inferred: `g2_frozen_run_five_arms.py:684` is
`torch.manual_seed(args.seed + 100003*(tasks.index(task)+1) + i)`, so rs(42,t,i+1) == rs(43,t,i)
exactly — G2's mechanism for the collision property already on record.

### 15.26.4 Net after this round

| claim | status |
|---|---|
| Encoder collapsed (cos 0.99999) | ✅ holds, but **subsumed** by the retrieval bound |
| ⭐ **z cannot identify its own prompt better than chance** (5/1800 vs 0.0050; content < ~0.2% of ‖z‖; POS 200/200) | ✅ **strongest surviving claim — adopt as the finding** |
| Seed coupling from `:684`, cross-seed CIs invalid | ✅ holds, mechanism in the producer |
| A1 > A4 sign | ✅ holds (both scales) |
| Any percentage split (aux vs latent) | ⛔ refused — censored endpoint |
| A3 sign-symmetry / "no consistent sign" | ⛔⛔ **WITHDRAWN — pinned by permutation algebra** |
| A3 per-row magnitude | ⚠️ undetermined; my common-mode rescue failed against me |
| Decode drift across processes (0.0109/0.0841) | ✅ real, but bounds a **different** quantity than I applied it to |

**Why every content-sensitivity arm on this checkpoint is underdetermined, in one sentence:** there is
no prompt-dependent content in z above ~0.2% of its norm for any arm to be sensitive to. That is a
fact about the checkpoint, established with a positive control, a chance baseline and a calibrated
detection floor — and it needs no entropy, no signed mean and no arm comparison.

### 15.26.5 Instrument-failure tally for this round (three agents, one session)

- **Mine:** the A4-vs-A4 tautology (§15.25.4); the hardcoded "decode is reproducible" contradicting its
  own numbers (§15.25.5); the failed common-mode rescue (above); the 10× calibration error from
  double-counted noise (above).
- **G2's, self-reported:** a coherence run using the per-task **sample** mean instead of the training
  mean, which forces A2's field to zero-sum too, printed 0.000000 for **both** arms, and was narrated
  as reproducing a contrast; then a script printing "nonzero only at reduced precision" directly beneath
  three lines reading 0.000000; plus a withdrawn "content = 0.26% of ‖z‖" decomposition whose own
  consistency check failed at 0.87.
- **The coordinator's:** an A3 coherence of 0.0084/0.0033/0.0045 computed on a row **subset** (exact
  value on all 200 is 0.000000 at every precision; first-199 gives 0.0047/0.0057/0.0059) — its own NEG
  control flagged it.

⭐⭐⭐ **Three agents independently produced a verdict that contradicted its own adjacent numbers, in
one session.** The general lesson, paired with the `torch.equal` entry: ⭐⭐⭐ **a control's validity is
a property of its construction geometry, not of whether the substitution occurred — and both the
`torch.equal` guard and my sign-randomisation controls were blind to exactly that.**

---

## §15.27 bf16 caps everything; my retrieval instrument had 51 silent degeneracies; G2's √2 reading is directionally inverted; and after FOUR failed calibrations the fifth passes its gates and VINDICATES G2's ceiling

> **Read §15.27.4 before citing anything here.** Within this single round I wrote down, and then
> falsified, two mutually contradictory verdicts on the ceiling (first "~5%", then "not established",
> then "geometry forbids detection"). The version that survives its controls is the **last** one, and
> it happens to agree with G2. The discarded intermediate claims are kept below on purpose, because
> the sequence is the finding.

### 15.27.1 ✅ bf16 confirmed, and the quantization budget reproduces exactly

Cached latents are `torch.bfloat16`: relative ULP 7.8125e-03, **half-ULP = 0.3906% of a coordinate**.
Independently recomputed per-coordinate rounding budget: **0.1821% of ‖z‖** on all three tasks —
matches G2's 0.1821% to four digits. I adopt G2's consequence: **any "bit-identical latents"
observation is weaker than a ~0.2% bound at this dtype**, so exact-equality arguments buy nothing.

### 15.27.2 ⛔ My retrieval instrument had 51 tied-max rows — the defect G2 had just described, and I quoted the number without checking for it

Audit (`/tmp/latentuse/audit_retrieval.py`, `ties.py`): zero-norm rows 0, NaN 0, but **tied-max rows
5/5/11/11/9/10 = 51 across 6 cells at fp32**. `argmax` breaks ties by lowest index silently, so a tie
at row 0 is an invisible fake hit and a tie elsewhere an invisible fake miss. Mechanism: bf16 inputs
give pair cosines ~0.99999 that collide at fp32 — one row's top two were both
`0.9999975562095642`, separated only at fp64 (`0.9999976854707278` vs `0.999997538666493`).

Verdict survives at fp64 with ties scored as misses: s43→s42 **3/600**, s44→s42 **2/600**,
s44→s43 **0/600**, pooled **5/1800 = 0.0028** vs chance 0.0050; fp64 ties **0**; POS self-retrieval
200/200; a zeroed row now **raises**. ⭐ **Conclusion unchanged, but the number I reported last round
came from an instrument with 51 silent degeneracies.**

**Also checked (new):** mean-centering the probe — the pitfall my own memory file warns about —
changes nothing: pooled **5/1800 raw vs 5/1800 centered**, POS 200/200 both, NEG (shuffled keys)
0/200 both (`/tmp/latentuse/centered.py`).

### 15.27.3 ⛔ G2's √2 mechanism reproduces, but its directional reading is inverted

Reproduced (`sqrt2.py`): pooled paired/absolute **1.4890** vs √2 = 1.4142, pooled corr −0.0834, CI
[−0.530,+0.399], n=18. Mechanism accepted. **But the consequence runs the other way:** inflating the
*floor* makes the effect relatively *smaller* — 0.019209 vs 0.010925 = **1.76×** becomes 0.019209 vs
0.015450 = **1.24×**. ⭐ **√2 inflation moves the comparison AGAINST the A3 magnitude claim.** My
retraction was not "over-corrected in the conservative direction" as G2 argued; this tightens it.
G2's arithmetic is right, its sign is wrong — same class as its 146× reading.

### 15.27.4 ⛔⛔⛔ FIVE calibrations, THREE contradictory verdicts from me in one round: the failure sequence, and the one that survives

| # | design | gates | verdict it would have licensed |
|---|---|---|---|
| 1 | fresh-rounding + synthetic noise, one generator | ⛔ POS 0.235 at c=2%, **non-monotone** | (unusable) |
| 2 | fresh generator per cell, gated | ✅ POS 1.000 / NEG 0.001 / monotone | ground truth **5%**, both G2 variants 0.50% ⇒ "G2 is 10× optimistic" |
| 3 | noise anchored to measured 0.4315% separation | ⛔ **NEG(c=0) = 0.2340** vs required ~0.005 | (refused to read) |
| 4 | "resolvability" geometry: is own cross-seed move < nearest-neighbour spacing? | ✅ POS 200/200, NEG(100×) 0/200 | ceiling **7/1800 = 0.0039 < chance** ⇒ "no-detection is forced by geometry, carries no information" |
| **5** | **inject known content into BOTH seeds' REAL caches, pooled n=1800, binomial vs 1/200** | ✅ **NEG(c=0)=0.0050, POS(c=100%)=1.0000, monotone** | **detects ≥0.10–0.15% ⇒ content < ~0.15% of ‖z‖** |

**Why #2 was wrong:** its simulated same-prompt separation was 3.4525% of ‖z‖ against a measured
0.4315% — 8× too noisy, so its 5% threshold was its own noise model's, not the data's.

**Why #4 was wrong — and this is the instructive one.** Its POS and NEG controls both passed, and it
still produced a false verdict, because the *nuisance quantity it conditioned on was itself the
signal*. Injecting content raises resolvability 3/200 → 22 → 188 → 200/200 as c goes 0 → 0.2% → 0.5%
→ 1%, so "nearest-neighbour spacing" is not a fixed property of the caches — **it is what the content
creates.** ⭐⭐⭐ **"The ceiling is below chance" was circular: I divided by the very thing I was
trying to measure.** A passing POS/NEG pair does not certify that the conditioning variable is
exogenous.

**The surviving instrument (#5, `power2.py`/`robust.py`).** Real caches, real cross-seed noise, a
known content vector added to both sides:

| c (% of ‖z‖) | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|---|
| pooled hits /1800 | 5 | 7 | 16 | 62 | 207 | 944 | 1775 |
| p vs chance | 0.95 | 0.79 | 0.022 | 3e−31 | 5e−203 | 0 | 0 |

Smallest detected c is **0.15%** for 4/5 injection seeds and **0.10%** for the fifth — so the limit is
**0.10–0.15% of ‖z‖**, quoted as a range, not a point. Real data sits at chance (5/1800, p=0.95).

⭐⭐ **Therefore G2's "content ≲0.1–0.2% of ‖z‖" is CORRECT, and I now hold it on an instrument that
passes a zero-signal NEG at exactly chance and a saturating POS.** My §15.27.4-as-first-written
retraction of that ceiling is itself **retracted**. Sequence of my own positions on one quantity in
one round: 5% → "not established" → "geometry forbids it" → **0.10–0.15%**. Three of four were wrong
and each was printed with gates attached.

**Bonus, resolving a scare of my own making:** cos(cross-seed displacement, row content deviation) =
**+0.69**, which looked like noise aligned with signal. An iid synthetic control with **no** shared
prompt component gives **+0.6982**, and 1/√2 = 0.7071 — ⭐ **the +0.69 is a √2 identity, and is
itself consistent with row variation being pure sampling noise** (`robust.py`).

### 15.27.5 Revised net — what I stand behind

| claim | status |
|---|---|
| Latents bf16; quantization 0.1821% of ‖z‖; exact-equality arguments weak here | ✅ holds |
| ⭐ **z does not identify its own prompt above chance** (5/1800=0.0028 vs 0.0050; fp64; ties=miss; POS 200/200; centering-invariant) | ✅ **holds** |
| ⭐⭐ **prompt-dependent shared component < ~0.10–0.15% of ‖z‖** (instrument #5, all gates pass, 5 injection seeds) | ✅ **holds — G2's ceiling vindicated** |
| "no-detection is forced by geometry / carries no information" (my #4) | ⛔⛔ **withdrawn — circular** |
| "ground truth fires at 5%, G2 is 10× optimistic" (my #2) | ⛔ **withdrawn — noise 8× overstated** |
| Encoder collapse (cos 0.99999) | ✅ holds |
| Seed coupling from `:684` ⇒ cross-seed CIs invalid | ✅ holds |
| A1 > A4 sign | ✅ holds |
| Any aux-vs-latent percentage split | ⛔ refused (censored endpoint) |
| A3 ≈ A4 / sign-symmetry / per-row magnitude | ⛔⛔ withdrawn (§15.25–26) |
| G2's √2 ⇒ "your retraction was over-corrected" | ⛔ inverted; it tightens the retraction |

⭐⭐⭐ **Reusable (the real lesson of this round):** a detection bound needs (a) a zero-signal control
at chance, (b) noise matched to a measured quantity from the real data, **and (c) proof that whatever
the instrument conditions on is not itself a function of the signal.** Calibrations #1–#4 each
satisfied at most two. #4 passed both its controls and was still circular — ⭐ **"my POS and NEG
passed" is necessary, not sufficient; ask separately what the denominator is made of.**

G2's five published rows (A0 71.17 / A1 5.33 / A2 0.00 / A3 0.00 / A4 0.00, `prefix_replay=none`,
pin `b55f685b605aae84eea9c75ea1b5efac`) are untouched by all of this.

### 15.27.6 Housekeeping fact that bit me

The task keys are `sciq` / **`commonsenseqa`** / `race`. I wrote `cqa` (its name in the scoring
ledgers) and my `raise`-on-unknown-shape guard turned it into a crash instead of a silent two-task
average — the briefing's "if a shape or key is unknown, raise, do not skip" paid for itself here.

### 15.27.7 ⛔ The standing "appendix blocker" is CONFIRMED, but my path for it was wrong, and the mechanism is not the one I reported

I had been citing `laces-reasoning/paper_ICLR27/lalaland/appendix.tex`. **That file does not exist.**
The paper tree is `DiffRwkv/paper_ICLR27/lalaland/`. Both protected files verify there at the expected
digests (`appendix.tex` = `4bd162948b3300452c9714a6273a23bb`, `main.tex` =
`32f46730821226cc150d3ce0dfef1f9b`), so the hashes I have been reporting were of the right bytes on a
path I was quoting wrongly.

Blocker reconfirmed on **three independent instruments**, with the `\input`-pattern positive control
the memory file demands (the same pattern returns 13 / 1 / 2 on the EED, speech-negotiation-kv and
trajflow-kv papers, so it is not vacuous):

| instrument | result |
|---|---|
| `\input`/`\include{...appendix}` in `main.tex` | **0** (POS control: pattern matches 13× elsewhere) |
| `tab:latent-use-pending` in `main.tex` | **0** (~~defined **2×**~~ → **1 `\label` + 1 `\ref`**, see §15.29.13) |
| `latent-use` in compiled `main.aux` | **0** (POS: `tab:capability` = 6, `app:training` = 1) |
| `pdftotext main.pdf` | `Latent use` 0, `derangement` 0, `A0` 0 |

⭐ **Mechanism correction:** `main.tex` is not missing an appendix — it has `\appendix` at
`main.tex:467` and its own `\end{document}` at `main.tex:993`, i.e. **a 526-line inline appendix**.
`appendix.tex` is a 938-line **orphan** with 93 labels, **68 of which are not defined as `\label` in
`main.tex`** (25 shared; 73 are never `\ref`'d from it — see §15.29.13 for why this one question has
three different integers). So this is not "one table has nowhere to land"; it is **two divergent
appendices**, and the one holding `tab:latent-use-pending` is the one that does not compile. Writing
the five-arm table into `appendix.tex` would produce no change in `main.pdf` — ⭐ **a deliverable that is invisible to the
build is not a deliverable**, and I would have "finished" the task with nothing in the paper.

This is a coordinator decision (which appendix is canonical), not something I can resolve, and it
gates the entire remaining value of this work.

### 15.27.8 Reproducibility pin + one formatting trap in my own output

`/tmp/latentuse/power2.py` is **deterministic**: two consecutive runs are byte-identical (all
generators explicitly seeded; fp64 throughout). Instrument #5's table is reproducible as printed.

⚠️ **Do not quote that script's last line.** It renders the limit with `:.1f`, so the measured
**0.15%** prints as **"< ~0.1%"**. The defensible statement is the **range 0.10–0.15% of ‖z‖**
(0.15% for 4/5 injection seeds, 0.10% for the fifth), not the single rounded figure the script
displays — a `.1f` on a two-significant-digit threshold silently drops the very digit that
distinguishes it from the 0.1821% quantization budget.

---

## §15.28 ⛔⛔⛔ THE HEADLINE OF §15.27 IS FALSIFIED: "z does not identify its own prompt above chance" was measured in the wrong alignment frame. In the noise-identical frame retrieval is 53.6%, i.e. 321× higher.

> ⛔⛔⛔ **SUPERSEDED AND WITHDRAWN BY §15.29 — DO NOT CITE.** The 958/1788 figure below is
> arithmetically correct but MISLABELLED: it retrieves the **noise-mate**, whose prompt is the
> **wrong** prompt. Own-prompt retrieval in the same frame is **3/1788 = 0.0017, below chance**.
> The conclusion of §15.27 ("z does not identify its own prompt above chance") is RESTORED.
> This section is kept only as the record of the error.


**Read this before citing any part of §15.26 or §15.27.** G2 supplied the mechanism; I verified it and it
overturns the one result I had called solid.

### 15.28.1 The mechanism, verified from the producer's own imports

`g2_frozen_run_five_arms.py:83-84` imports `encode_prefix` and `sample_ddim_cfg` **from
`scripts/eval/sample_prefix_suffix_cfg.py`** — so G2's citation is the actual code path, not a
same-named lookalike. In that file:

- `sample_ddim_cfg` line 1: **`z = torch.randn(cond.shape[0], model.latent_dim, ...)`** — the first
  RNG consumption after the seed is set.
- The DDIM loop is closed-form (`eps`, `z0_pred`, `z` updates only) — **no further RNG**.
- `encode_prefix` uses **`model.mu_head(h)`** for `encoder_type == "variational"` — the *mean*, not a
  reparameterised sample — so it draws **no** RNG either.

Producer `:684` seeds with `args.seed + 100003*(taskidx+1) + i`. Therefore
`seed42(task,i) == seed43(task,i−1) == seed44(task,i−2)`: **at that offset the initial noise is
bit-identical** (verified directly: same seed ⇒ `torch.equal` True on bf16 `randn`). Any residual
difference there is caused by the **prompt alone**.

### 15.28.2 The offset sweep — a blind prediction that lands 9/9

Predicted dip: 42→43 at −1, 43→44 at −1, 42→44 at **−2**. Mean relative separation (% of ‖z‖):

| pair / task | o=−3 | o=−2 | o=−1 | o=0 | o=+1 | o=+2 |
|---|---|---|---|---|---|---|
| 42→43 sciq | 0.437 | 0.435 | **0.262** | 0.432 | 0.441 | 0.437 |
| 42→43 cqa | 0.437 | 0.441 | **0.263** | 0.443 | 0.433 | 0.434 |
| 42→43 race | 0.432 | 0.430 | **0.259** | 0.429 | 0.434 | 0.434 |
| 43→44 sciq | 0.441 | 0.438 | **0.259** | 0.435 | 0.439 | 0.434 |
| 43→44 cqa | 0.436 | 0.437 | **0.255** | 0.435 | 0.432 | 0.428 |
| 43→44 race | 0.433 | 0.429 | **0.264** | 0.430 | 0.434 | 0.435 |
| 42→44 sciq | 0.441 | **0.261** | 0.438 | 0.440 | 0.444 | 0.447 |
| 42→44 cqa | 0.439 | **0.259** | 0.437 | 0.435 | 0.434 | 0.439 |
| 42→44 race | 0.431 | **0.256** | 0.432 | 0.427 | 0.435 | 0.435 |

**9/9 minima sit exactly where the seed arithmetic predicts**, ~0.26% against a flat ~0.44% elsewhere.
A prediction made from source code before looking, hitting 9/9, is not a fitted result.

### 15.28.3 ⛔⛔⛔ Retrieval in the correct frame: 958/1788 = 53.6%, versus the 3/1800 I published

| pair / task | offset 0 (my frame) | ALIASED frame | chance |
|---|---|---|---|
| 42→43 sciq | 1/200 | **103/199** | 0.0050 |
| 42→43 cqa | 0/200 | **102/199** | 0.0050 |
| 42→43 race | 1/200 | **111/199** | 0.0050 |
| 43→44 sciq | 1/200 | **108/199** | 0.0050 |
| 43→44 cqa | 0/200 | **112/199** | 0.0050 |
| 43→44 race | 0/200 | **101/199** | 0.0050 |
| 42→44 sciq | 0/200 | **104/198** | 0.0050 |
| 42→44 cqa | 0/200 | **110/198** | 0.0050 |
| 42→44 race | 0/200 | **107/198** | 0.0050 |
| **pooled** | **3/1800 = 0.0017** (p=0.994) | **958/1788 = 0.5358** (p≈0) | 0.0050 |

**Ratio 321×.** Controls on the aliased frame: correct pairing 103/199, **shuffled keys 1/199 = 0.0050
= exactly chance**, so the frame is a real instrument and not a bookkeeping artefact. Two rows are
exactly zero (identical prompts), the rest separate by 0.2618% mean.

⛔⛔ **"z does not identify its own prompt above chance" is WITHDRAWN.** z identifies its prompt
**53.6%** of the time out of 199 candidates when the noise is held fixed. My result was a statement
about the noise, measured in a frame where the noise dominates the prompt by 1.65–1.68×.

⭐⭐⭐ **This is form 96 in the memory file, arriving a second time in the same work**: the previous
instance ("0/9 all at chance" measured in the wrong frame) is already recorded, and I reproduced the
identical error on a different statistic while holding the note that warns about it. **Reporting "no
detection" requires first showing the instrument resolves a signal known to exist in that frame** —
I did run a POS control (self-retrieval 200/200), but self-retrieval is trivially in-frame, so it
certified nothing about cross-seed alignment.

### 15.28.4 What survives, and the corrected decomposition

- **Prompt effect (noise held bit-identical): 0.255–0.264% of ‖z‖**, 9/9 cells.
- **Noise + prompt (non-aliased): 0.427–0.447%**, ratio **1.65–1.68×**.
- My "**< 0.10–0.15% of ‖z‖**" limit (§15.27.4 instrument #5) is **not wrong as arithmetic but is
  answering a different question**: it injected content shared across *both* members of a pair whose
  noise differs, so it bounds only the **noise-invariant, cross-draw-shared** component. It does
  **not** bound the prompt's total displacement, which is 0.26% and plainly detectable. G2's
  independent `cos(d1,d2)` bound (≤0.044%, n=589) is on that same restricted quantity and agrees.
- **Encoder collapse still holds** (cos 0.99999, ‖mean‖/mean‖z‖ = 0.99999565) — and is now the more
  interesting fact: the prompt *is* encoded, at ~0.26% of ‖z‖, on top of a mean that dominates. A
  0.26% signal that supports 53.6% retrieval is real but tiny, which is entirely consistent with the
  decoder ignoring it.

### 15.28.5 ⛔ G2's 0.1821% withdrawal — accepted, and I unadopt it

G2 withdraws its "bf16 quantization RMS = 0.1821% of ‖z‖" and asks me to unadopt it. **Accepted.** Its
reasoning is sound: bf16 rounding is **deterministic**, so two runs of the same computation cannot
carry independent additive rounding noise (no `2q²` term), and z is bf16 end-to-end so there is no
pre-quantization value for q to be the error of. ⭐⭐ **My "reproduced to four digits" reproduced its
formula, not a property of the data** — we shared a premise, which is exactly why we agreed. Four-digit
agreement between two parties is worthless when both compute the same wrong expression.
**Consequence:** any floor-relative multiple built on 0.1821% is void, which includes the
1.76× → 1.24× arithmetic in §15.27.3. The **sign** of that correction still stands (inflating a floor
weakens a floor-relative claim), but the A3 reading should not rest on a floor-relative multiple at
all. What survives from §15.27.1 is only the qualitative point that **bit-identity arguments buy
nothing at this dtype**.

### 15.28.6 G2's seed43 non-determinism — affects my replicate arguments

G2 reports seed42 smoke-vs-full **18/18** bitwise identical, seed44 **18/18**, but seed43 only
**6/18**, differing rows at 0.2312% of ‖z‖, with config keys identical and none of the 12 differing
rows matching any of the 1800 rows (so not a mislabel or index shift; the shift hypothesis predicted
1 clean + 2 fully-differing tasks, observed 1/6, 2/6, 3/6). **My §15.28.3 uses seed43 in 6 of 9
cells**, so those inherit it. Mitigation already in the data: the 42→44 pair does **not** involve
seed43 and gives 104/198, 110/198, 107/198 = **321/594 = 54.04%**, p≈ 0, **107× chance** — so the
conclusion does not depend on seed43 at all. (⚠️ The seed43-free hit count `321` and the pooled
ratio `321×` are an unrelated numerical coincidence — different quantities, never to be conflated
when quoting.) The **0.26% point value** should be quoted as a **range 0.14–0.27%** per G2's disclosure.

### 15.28.7 Corrected net

| claim | status |
|---|---|
| ⭐⭐ **z encodes its prompt: 958/1788 = 53.6% retrieval with noise held fixed** (NEG shuffled = 0.0050 = chance; 42→44 pair seed43-free) | ✅ **new headline** |
| Prompt displacement ~0.26% of ‖z‖ (quote as 0.14–0.27%) | ✅ holds |
| "z does not identify its prompt above chance" (3–5/1800) | ⛔⛔ **WITHDRAWN — wrong frame** |
| "content < 0.10–0.15% of ‖z‖" | ⚠️ **re-scoped**: bounds only the noise-invariant shared component, not the prompt effect |
| bf16 quantization RMS 0.1821% | ⛔ **withdrawn** (G2's, and my adoption of it) |
| §15.27.3's 1.76×→1.24× floor arithmetic | ⛔ void (built on 0.1821%); the sign of the argument survives |
| "no-detection forced by geometry" (my #4) | ⛔⛔ withdrawn — circular |
| Encoder collapse (cos 0.99999) | ✅ holds, and is now the interesting tension |
| Seed coupling ⇒ cross-seed CIs invalid | ✅ holds — **and it is also what made this measurement possible** |
| A1 > A4 sign | ✅ holds |
| A0 77.8 vs A4 0.0 under `prefix_replay=none`; writing is destructive | ✅ holds (unaffected) |

⭐⭐⭐ **Reusable:** *the seed coupling that invalidates cross-seed CIs is simultaneously a free
controlled experiment* — deterministic seed collisions hold the noise fixed and isolate the input's
effect. I spent three rounds treating `:684` purely as a defect to be worked around; it was the
single most informative feature of this dataset.

---

## §15.29 ⛔⛔⛔ §15.28 IS WITHDRAWN. The 958 was real but I mislabelled WHAT it retrieves: the noise-mate, whose prompt is the WRONG prompt. Own-prompt retrieval is 3/1788 — AT chance (not below it, see §15.29.8). The §15.27 headline was right after all.

**This supersedes §15.28 entirely.** G2 caught it; I verified every step independently and it is correct.

> ⚠️ **§15.29.2 below originally read "below chance" on a null of 8.94 and an upper-tail p of 0.994.
> Both were wrong — see §15.29.8, which is the authoritative version of that one claim.** The label
> finding (noise-mate, not own prompt) is unaffected; only the strength and direction of the
> own-prompt statement changed. The lines are kept, struck through, so the error stays auditable.

### 15.29.1 The arithmetic, which settles it with no data at all

My §15.28 code used `o=-1; q,k = A[-o:], B[:o]`, i.e. `q[m] = A[1+m]`, `k[m] = B[m]`. Then:

```
seed(A[1+m]) = 42 + 100003*(tk+1) + (1+m)
seed(B[m])   = 43 + 100003*(tk+1) + m     = 42 + 100003*(tk+1) + (m+1)   -> EQUAL
```

Verified for tk ∈ {0,1,2}, m ∈ {0,7,50,198}. So my "diagonal" paired **example index `1+m` in seed42
with example index `m` in seed43** — equal noise seed, **example indices differing by one**, therefore
**different prompts**. ⛔ **A "hit" on my diagonal means the nearest neighbour is the row with the SAME
NOISE DRAW, and the prompt it belongs to is the wrong prompt.** The prompt is merely what stops the
distance from being zero.

### 15.29.2 Scoring both targets over the same candidate set

No slicing: for query `z(42,i)` over all 200 candidates in the other seed, score (a) the noise-mate
`i−shift` and (b) the query's **own** prompt `i`.

| pair / task | noise-mate | own-prompt | chance |
|---|---|---|---|
| 42→43 sciq | 103/199 | **1/199** | 0.0050 |
| 42→43 cqa | 102/199 | **0/199** | 0.0050 |
| 42→43 race | 111/199 | **1/199** | 0.0050 |
| 43→44 sciq | 108/199 | **1/199** | 0.0050 |
| 43→44 cqa | 112/199 | **0/199** | 0.0050 |
| 43→44 race | 101/199 | **0/199** | 0.0050 |
| 42→44 sciq | 103/198 | **0/198** | 0.0050 |
| 42→44 cqa | 110/198 | **0/198** | 0.0050 |
| 42→44 race | 106/198 | **0/198** | 0.0050 |
| **pooled** | **956/1788 = 0.5347** (p < 1e−300) | **3/1788 = 0.0017** | 0.0050 |

~~Own-prompt is **below** chance (expected 8.9 hits, observed 3). Ratio 319×.~~ ⛔ **Struck: wrong null
AND wrong tail — see §15.29.8.** The correct reading is **at chance**: the crowded null is 4.18, not
8.94, and the lower tail is P(X≤3) = 0.399. ⭐ **In ~99.5% of the 958 "hits" the retrieved neighbour's
prompt is the wrong prompt** — this part stands and is what settles the label.

### 15.29.3 The confound that decides the label, reproduced

If consecutive examples shared text, a hit at `i−1` would be *prompt* similarity and G2's label would
be wrong instead of mine. Control: hold prompt adjacency fixed and **remove the shared noise** by
staying within one seed (`z(s,i)` vs `z(s,j)`, self excluded). Top-1 lands on `i±1`:

| | sciq | cqa | race |
|---|---|---|---|
| seed42 | 2/200 | 3/200 | 4/200 |
| seed43 | 3/200 | 0/200 | 0/200 |
| seed44 | 4/200 | 1/200 | 2/200 |

Chance = 2/199 = 0.0101 ⇒ ~2.0 expected. **Flat.** Adjacency is excluded. Also, from the recorded
`prompt_sha256` (no prompt text copied anywhere): **0 consecutive exact duplicates and 0 duplicates at
all** in each of the three 200-row sets.

⚠️ **G2's RACE caveat is real and I reproduce it:** lag-1 correlation of `prompt_len_chars` is
**+0.159** in race (shuffled −0.005±0.071, **z = +2.29**), sciq −0.110, cqa +0.051. So consecutive RACE
prompts are *not* independent in length, consistent with shared passages. **The within-seed latent
control above is what actually excludes adjacency** — I must not claim prompt lengths are uncorrelated.

### 15.29.4 Rank test with a MEASURED null — independently confirms no prompt signal

Drop the noise-mate from the candidate set entirely and ask where the own prompt ranks. The null must
be **measured**, not the analytic 100.0, because removing a near-certain rank-1 shifts it:

| task | own-prompt rank | measured null | z |
|---|---|---|---|
| sciq | 98.2 | 100.0 ± 4.2 | −0.41 |
| cqa | 101.2 | 99.4 ± 3.9 | +0.45 |
| race | 101.2 | 99.7 ± 3.4 | +0.45 |

**Stouffer pooled z = +0.28, not significant.** My numbers differ slightly from G2's (it reports
91.4/100.2/95.9 vs null ~98, pooled −0.98) but the verdict is identical and neither is significant.

### 15.29.5 Why high retrieval accuracy is evidence *against* my §15.28 reading

Matched-noise pairs differ by 0.14–0.27% of ‖z‖ (prompt only); every other candidate differs by ~0.44%
(prompt + noise). The neighbour is selected by **shared noise**. ⭐⭐ **A larger prompt effect would
LOWER this retrieval accuracy**, because it would push the noise-mate away. Accuracy *rising* is
evidence the prompt matters **less**. I had the direction of the inference backwards.

### 15.29.6 Corrected net — §15.27's headline is restored

| claim | status |
|---|---|
| ⭐⭐ **z does not identify its own prompt above chance** — five independent readings: my 5/1800 (offset 0), own-prompt 3/1788 (aliased), both tie rules, the rank test (z=+0.28), G2's cos-direction null | ✅ **RESTORED as the headline** |
| "own-prompt is **below** chance / 319× / p=0.994" | ⛔ **WITHDRAWN (mine) — wrong null and wrong tail, §15.29.8. Correct: AT chance.** |
| The fp32/cos mate-deleted cell (16/1788, p=0.0215) | ✅ **DISCHARGED by the joint (top-1, mean-rank) test, §15.29.9** — any shared prompt component consistent with the data is **< ~0.015% of ‖z‖**, so the excess cannot be one. No longer needs a quarantine banner, but still quote it with the sweep. |
| "z encodes its prompt: 53.6% retrieval" (§15.28) | ⛔⛔⛔ **WITHDRAWN — that is noise-draw retrieval, wrong-prompt in ~99.5% of hits** |
| The 958/1788 number itself | ✅ correct, **relabelled**: noise-mate retrieval, not prompt retrieval |
| Prompt displacement 0.14–0.27% of ‖z‖ (noise held fixed) | ✅ holds — but it does **not** imply prompt identifiability |
| "< 0.10–0.15% of ‖z‖" noise-invariant component | ✅ holds, correctly scoped |
| bf16 q = 0.1821% and "93.5–95.4% of variance removed" | ⛔ both withdrawn (G2's, and my adoption) |
| Encoder collapse (cos 0.99999) | ✅ holds |
| A0 77.8 vs A4 0.0, writing is destructive | ✅ holds (untouched) |
| A1 > A4 sign | ✅ holds |

### 15.29.7 ⭐⭐⭐ The lesson, and it is a new form

**A correctly-executed control with a valid positive and negative control can still answer the wrong
question, because the identity of the retrieved TARGET is a separate fact from the retrieval rate.**
My aliased frame had a genuine NEG control (shuffled keys = 0.0050 = exactly chance) and it passed —
because shuffling destroys the noise pairing too. The control certified that *something* systematic
was being found; it could not say *what*. ⭐ **Print the identity of the argmax, not only whether it
equals the index you assumed it should equal.** I checked `argmax == arange`, which silently encoded my
assumption that position `m` on the diagonal *was* the own prompt.

Three rounds, three positions on this one question (at chance → 53.6% encoded → at chance), and the
survivor is the original. ⭐⭐ **Two of my three reversals were caused by the same diagonal-index
confusion in opposite directions**; the frame that "fixed" the wrong-frame error introduced a
wrong-target error of exactly the same shape.

### 15.29.8 ⭐⭐ CORRECTION (G2, verified by me): own-prompt is AT chance, not below it. Two defects — a wrong null and a wrong tail — and I had BOTH.

G2 retracted its own "below chance" framing and I reproduced both defects in my §15.29.2. Independently
confirmed with `/tmp/latentuse/crowded.py` (pooled n=1788, fp64/cos, mate wins 956, own wins 3):

**Defect 1 — the null was not crowded.** `1788/200 = 8.94` assumes own-prompt could win all 1788
queries. It cannot: the noise-mate wins 956 of them, and those queries are unavailable. The own-prompt
null is **(queries the mate did not win)/199 = 832/199 = 4.18**. A 30-rep randomisation that keeps the
real geometry and only relabels which non-mate column is "own" gives **4.63 ± 2.02** — agrees.

**Defect 2 — the tail did not match the claim.** I printed `P(X≥3) = 0.994` (recomputed: 0.788) while
arguing a *lower*-tail claim. ⭐ **`P(X≥k)` can never support "below chance."**

| null | E[hits] | observed | P(X≤obs) — the tail the claim needs | reading |
|---|---|---|---|---|
| crowded (correct) | 4.18 | 3 | **0.399** | **at chance** |
| uncrowded 1/200 (my old, wrong) | 8.94 | 3 | 0.022 | would read "below" — but the null is invalid |

⭐ **"Below chance" fails under the correct null, and the only null that produced it was the broken
one.** Defensible statement: **own-prompt retrieval is at chance in both directions.**

**The mate-deleted control is setting-dependent, in the dangerous direction.** Deleting the mate frees
all 199 slots (null 8.98). 12 settings = {fp32,fp64} × {cos,dot,l2} × {mate kept, mate deleted}:

| statistic | fp32/cos | fp32/dot | fp32/l2 | fp64/cos | fp64/dot | fp64/l2 |
|---|---|---|---|---|---|---|
| own, mate kept (null 4.2) | 7 | 9 | 4 | 3 | 9 | 4 |
| own, mate deleted (null 9.0) | **16** | 9 | 11 | 14 | 9 | 11 |
| ties at max | 78 | 0 | 174 | 0 | 0 | 174 |

**1 of 12** cells is nominally significant in the excess direction: fp32/cos mate-deleted, 16 vs 9.0,
p=0.0215. ⛔ **I get 1/12, G2 reported 3/12** — since resolved: G2's third axis is the **tie rule**, so
its three are three tie-rule variants of my one cell. We agree exactly on the minimum p and hence on
Bonferroni **0.257** / Šidák **0.229**.
A 2000-rep permutation null (`/tmp/latentuse/permnull.py`) reproduces the analytic null (8.92–9.09 vs
8.98) and gives a **sweep-level max-statistic p = 0.1005**. Family-wise, ≥1 of 6 cells reaches its own
95th percentile in **34.7%** of null reps. The one significant cell is exactly the fp32/cos corner that
produced the 958, i.e. the corner that would resurrect the retracted headline — so it must never be
quoted without this sweep attached.

> ⚠️ **DO NOT QUOTE A BARE FAMILY-WISE VERDICT — see §15.29.12.** Across family definitions the FW p
> spans roughly **0.029–0.116 and crosses 0.05**, so "at chance even with the mate removed" is not
> stateable as a bare verdict from the family-wise statistic alone. My own 6-cell family gives
> **0.1112 ± 0.0133** (5 null seeds, and identically under both tie conventions). The own-prompt
> **at-chance** conclusion does **not** rest on this statistic — it rests on the crowded null
> (P(X≤3)=0.399) and on §15.29.9's joint test, both of which are unaffected.

**D4 controls on the sweep instrument** (`/tmp/latentuse/sweep_ctrl.py`, `neg_replicate.py`):
POS (a shared per-example vector at 3% of ‖z‖) → **1788/1788 in 4/6 cells, 6/6 fire**; NEG (i.i.d.
gaussian, no prompt signal by construction) → **per-cell false-positive rate 0.000–0.075 over 40
replicates, mean 0.20 cells firing** vs 0.05 expected. ✅ calibrated; can both fire and fail.

⛔⛔ **A broken-instrument call I made and then retracted within the same check.** My *first* NEG run
was a single noise draw; it fired in 2/6 cells and I wrote "BROKEN INSTRUMENT — report loudly." Forty
replicates show ≥2 cells firing in **10%** of pure-noise draws. ⭐⭐ **A negative control evaluated once
has no false-positive rate — it has an outcome. Reporting a defect from n=1 is the same n=1 error the
defect would consist of.** Two of the twelve cells are also *literally the same test*: fp32/dot vs
fp64/dot and fp32/l2 vs fp64/l2 agree on **100.00%** of 1788 argmaxes ⇒ ⭐ **"12 settings" is at most 8
independent ones; a multiplicity correction over duplicated cells is anti-conservative.**

⭐ **Fourth failure form for the collection:** *a mechanism check that explains away a null can itself
be a function of the retrieval setting, and its one significant corner can point straight back at the
conclusion that was jointly retracted.* Companion rule: ⭐ **every floor must name its statistic** —
mean rank moves at ~0.05% of ‖z‖, top-1 at 0.1–0.2%, cos-direction at 0.02–0.05%; "the floor is 0.15%"
is meaningless without saying which.

**Unchanged by this correction:** the label finding (the 958 retrieves the noise-mate), the adjacency
exclusion, the RACE length caveat, the 0.14–0.27% displacement range, and the five published rows.

### 15.29.9 ⭐⭐ The fp32/cos cell is now settled by measurement, not by a banner. Joint (top-1, mean-rank) test reproduced; G2's conclusion holds, two of its framings do not.

G2 answered blocking item 2 with a joint test: ask what **mean rank** must accompany a shared prompt
component large enough to produce the observed **top-1 = 16**. Reproduced independently in
`/tmp/latentuse/{joint_rank,neg_variants,boundary}.py`, fp32/cosine, noise-mate deleted, n=1788.

**Its observation reproduces exactly.** top-1 first/hit/miss = **16/16/15**, mean rank **98.725** —
identical to G2's three-decimal figure across all three tie rules. ✅

> ⚠️ **But the 16 is not a property of the data — see §15.29.11, Defect 2.** `argmax` breaks exact ties
> by column index and fp32 `cos ≈ 1` cancels to ~1e−7, so the count moves to **15.53 ± 0.50** under
> random column permutation (reversed order gives 15). The order-free reading is ~15.5, p ≈ 0.0215–0.0406.
> **Quote the order-stable cell instead: fp64/cos gives 14.00 ± 0.00 under every permutation.** This
> only weakens the one nominally significant cell, so the discharge below is unaffected.

**Finding 1 — the excess is not a shared prompt component.** top-1 alone cannot separate: the real 16
lies inside the injected range for **every** c from 0.005% to 0.075% of ‖z‖ (means 14.0–19.5, ranges
covering 16). Mean rank does separate, and it puts the real data at the null:

| c (% of ‖z‖) | top-1 [min–max] | mean rank | z of real vs this c | verdict |
|---|---|---|---|---|
| 0 (real) | 16 | 98.725 | — | — |
| 0.005 | 14.0 [12–16] | 98.75 ± 0.06 | −0.43 | **consistent** |
| 0.010 | 14.1 [12–16] | 98.54 ± 0.10 | +1.75 | **consistent** |
| 0.015 | 13.9 [11–16] | 98.28 ± 0.10 | +4.52 | excluded |
| 0.035 | 14.2 [9–18] | 95.84 ± 0.21 | +13.53 | excluded |
| 0.050 | 15.6 [14–17] | 92.78 ± 0.21 | +27.89 | excluded |
| 0.075 | 19.5 [16–23] | 85.24 ± 0.31 | +43.17 | excluded |
| 0.100 | 31.3 [26–40] | 75.44 ± 0.28 | +82.09 | excluded |

⇒ **any shared prompt component consistent with the data is < ~0.015% of ‖z‖**, far below the 0.10%
top-1 floor. The 16-vs-9 excess cannot be a shared prompt component. **Blocking item 2 is discharged:
the fp32/cos cell can no longer be read as "own-prompt retrieval above chance."**

> ⚠️ **SCOPE, required in any sentence quoted out of this file (see §15.29.10, Finding 5):** this
> injection is **homogeneous** — all 200 prompts share the same component. The bound therefore excludes
> a shared channel present in **more than a few percent** of prompts; a channel confined to **≲2%** of
> prompts survives (midrank sparsity sweep: f=0.05 excluded, f=0.02 and f=0.01 consistent). Do not
> state the bound unconditionally.

**Finding 2 — ⛔ "jointly consistent c: EMPTY SET" is over-stated.** The set is **not** empty: it
contains c = 0, 0.005%, 0.010%. G2's ladder started at 0.035%, so it never probed the region where
consistency lives, and an empty-set claim over a half-open ladder is unobtainable — ⭐ **c = 0 is
consistent with itself by construction, so any correctly-built joint test has a non-empty consistent
set and its output is an UPPER BOUND, never an emptiness.** The bound is the stronger statement anyway.

**Finding 3 — ⛔ its "NEG 98.751 ± 0.140" is not the null it is labelled as.** An sd of 0.140 is
arithmetically impossible for a sampling null of a mean of 1788 ranks: uniform on 1..199 has per-query
sd 57.45, so the mean has sd **1.359** analytically and **1.375** measured (400 prompt-relabel reps,
mean 100.021). The 0.140 reproduces exactly as an **unshared same-magnitude injection**
(c=0.035% unshared → 98.755 ± 0.189), i.e. a *magnitude-matched paired control* — a good control, and
it makes the right point (magnitude alone does not move the statistic; only **sharing** does), but its
sd is the injection-draw sd, not the null sd, and it must not be called "NEG (no shared component)".
⭐ **Both nulls agree on the verdict**: against the true relabel null the real mean rank gives
**z = −0.94** ⇒ at chance. Conclusion unaffected; only the error bar was ~10× too tight.

**Finding 4 — ⛔ "c=0 reproduces the observation bit-exactly, so it is its own positive control" is a
tautology w.r.t. the injection.** The injection is guarded by `if c > 0`, so at c=0 it is *skipped*:
the c=0 row runs the same code path as the plain measurement and **cannot fail**. It validates the
measurement (fp32/cos/mate-delete/argmax reproduces 16), not the injection. ⭐ **A row that a code
branch makes unreachable-to-perturbation is not a control for that perturbation** — related to the
known trap that *the `c=0` row of an injection study is a positive control wearing a null's label*.
The genuine positive control here is the **monotone ladder** (top-1 16→31, mean rank 98.7→75.4 as c
goes 0→0.1%), which is present and passes.

**Sweep count reconciled.** G2's "3/12" were three **tie-rule** variants (first/miss/hit) of the single
fp32/cos cell — its third axis is the tie rule, mine was mate-kept/deleted. So it is **one** cell
counted three times, not three agreeing settings; G2 has accepted this phrasing. Both of us: min
p = 0.0215, Bonferroni 0.257, Šidák 0.229. ⛔ **The sentence that stood here — "three independent
family-wise calibrations bracket the sweep: my 0.1005, the coordinator's 0.0700, both > 0.05" — is
WITHDRAWN; see §15.29.12.** The coordinator's 0.0700 was inflated by an unmasked mate in its null
(true 0.0455), and across family definitions the FW p **crosses 0.05**. My own 0.1005/0.1112 stands for
*my* 6-cell family, but "both > 0.05" was never a property of the family-wise statistic in general.
G2 also verified my per-cell analytic null by 3000-rep prompt-relabel permutation (E 8.88–8.89,
sd 2.94–2.95, inflation 0.98× ⇒ the binomial is sound and slightly conservative), and its refixed
floors, computed **in fp32/cos** rather than fp64/l2 and with 8–12 draws, are top-1 **0.10%** and
mean-rank **0.02%** — consistent with the < 0.015% bound above.

⭐ **G2 disclosed two defects of its own behind the floors it gave me earlier, both of which had been
biasing toward detection**: a rank shift computed against a hardcoded 99.6 from a different candidate
configuration (that loop's own c=0 is 95.52 ⇒ every shift biased −4.08 toward detection), and single
draws per c — to which my "a negative control evaluated once has no false-positive rate, only an
outcome" applies verbatim. It also notes the reference for rank must be NEG **at the same c**, not
c=0, since injecting any vector moves the statistic (with the wrong reference, c=0.02% printed
z=+16.19, sign reversed).

### 15.29.10 ⭐⭐ Root cause of two of the three framing defects: STRICT ranks + ties. Midrank repairs both. And the check that caught my own two-bug detour was a hand-checkable toy row, not the data.

G2 traced my Correction 2 (the ~10×-too-tight null sd) and its own sign reversal to one cause: **strict
rank counts only strictly-greater neighbours, so ties depress the statistic, and the amount of
depression is cell-dependent.** Verified independently in `/tmp/latentuse/midrank3.py`, n=1788,
400-rep prompt-relabel nulls per cell, mate-deleted.

| cell | rows w/ tie at own | real strict | real midrank | null strict | null midrank | z strict | z mid |
|---|---|---|---|---|---|---|---|
| fp32/cos | 1331 | 98.725 | 99.536 | 99.23 ± 1.35 | **100.02** ± 1.35 | −0.37 | −0.36 |
| fp64/cos | 1 | 99.603 | 99.603 | 100.04 ± 1.31 | **100.04** ± 1.31 | −0.33 | −0.33 |
| fp32/l2 | 1624 | **95.519** | 97.452 | 98.18 ± 1.32 | **100.07** ± 1.31 | −2.01 | −1.99 |
| fp64/l2 | 1624 | 95.519 | 97.452 | 98.09 ± 1.33 | **99.99** ± 1.33 | −1.93 | −1.91 |
| fp64/l2 on **normalised** z | 1 | 99.603 | 99.603 | 99.94 ± 1.29 | 99.94 ± 1.29 | −0.26 | −0.26 |
| **|norm difference| alone** | 18 | 98.866 | 98.871 | 99.99 ± 1.39 | 100.00 ± 1.39 | −0.81 | −0.81 |

**Finding 1 — the L2 "deficit" is a tie artefact, and there is no hidden norm channel.** Following my
own Correction 2 through nearly overturned the bound: if the null sd is 1.36 rather than 0.140, the L2
cell's c=0 strict mean rank of **95.519** (which I reproduce exactly) sits ~3.3 sd below 100, which
would mean a real channel L2 sees and cosine *cannot* detect by construction, since cosine discards
norm. Against its **own** relabel null it is **z = −2.01, not −3.3**. Two isolating controls close it:
L2 on **normalised** z gives z = −0.26, and **|norm difference| alone** gives z = −0.81. ⭐ **The scare
came entirely from scoring one cell against another cell's null.**

**Finding 2 — the strict-rank null is cell-dependent; the midrank null is not.** Ties at the own-prompt
entry: fp32/cos 1331/1788, fp64/cos **1**/1788, L2 1624/1788. Strict nulls therefore span
**98.09–100.04** (spread 1.95) while midrank nulls span **99.94–100.07** (spread 0.13, centred on 100).
⇒ ⭐⭐ **Score every cell against its OWN measured null, or use midranks. A null imported from another
dtype/metric cell is not a null.** This also explains the earlier puzzle that measured c=0 mean rank
was 99.60 rather than the analytic 100.0 — that gap is ties, not a signal.

**Finding 3 — fp32/cos barely moves, so §15.29.9's joint test is untouched.** strict → midrank changes
z by 0.01 there (−0.37 → −0.36). G2's ladder confirms it: fp64/l2 c=0.010% went strict **+42.96** →
midrank **−6.10** (the sign reversal disappears and monotonicity is restored), while fp32/cos c=0.035%
moved only −20.95 → −20.65. ⇒ ⭐ **print a monotonicity check on every power ladder; a non-monotone
ladder is a broken instrument, not a small effect.**

**Finding 4 — ⛔⛔ my own first midrank implementation had two coupled bugs, and the data could not
reveal them.** `(R.unsqueeze(1)>R.unsqueeze(2)).sum(1)` counts values strictly **less**, while the
mate-exclusion indicator I paired with it assumed "greater" — the two errors partly cancelled, so the
output looked plausible: all six midrank nulls at ~99.0 (not 100) and all six z-signs flipped versus
G2 while the magnitudes matched. ⭐⭐ **The signature of a transposed comparator is right magnitude,
wrong sign, uniform across cells** — a same-sign disagreement is a finding, a sign-flipped one with
matching magnitudes is my bug. What caught it was a **5-element hand-checked toy row**
(`[9,5,7,7,1]`, mate deleted ⇒ strict `[3,1,1,4]`, midrank `[3,1.5,1.5,4]`). ⭐ **A hand-computed
expected vector is asymmetric in exactly the way a mean is not, which is why it fired.**

> ⛔⛔ **CORRECTION (G2, verified by me in `/tmp/latentuse/identity_vacuous.py`) — I originally credited
> the catch to the identity "mean midrank over all 199 candidates ≡ 100.000" and recommended it as a
> gate. It is VACUOUS for the bug it claims to catch.** Reversing a ranking is `r → n+1−r`, whose mean
> is still `(n+1)/2`, so the identity reads **100.000000 for the correct comparator and 100.000000 for a
> clean transposition — difference exactly `0.00e+00`** while the max per-entry disagreement is
> **198.0 = n−1**, the fingerprint of two rankings being mutual reversals. Two entirely different
> rankings, identical gate reading. My v1 *did* violate it (99.0696), but only because of the **second**,
> coupled bug (the mate indicator); a single clean transposition passes. ⭐⭐ **Any statistic invariant
> under `r → n+1−r` — mean, sum, variance — cannot detect a direction error, so "it held in all six
> cells" was never evidence of correctness.**
>
> **Replacement gate, asymmetric by construction and verified to fire:** the most-similar candidate must
> have midrank ≈ 1. Correct **1.038 PASS**; clean transposition **198.962 FAIL**; my v1 **198.917 FAIL**.
> That is the one-liner to assert before reading any z.

**Finding 5 — scope limit: the exclusion is not unconditional.** Both our injections were **homogeneous**
(all 200 prompts share the same component). G2 re-ran the coordinator's sparsity sweep under midrank:
f=1.00 z=+26.30 excluded, f=0.10 z=+3.99 excluded, f=0.05 z=+4.36 excluded, **f=0.02 z=+2.82
CONSISTENT, f=0.01 z=+2.38 CONSISTENT**. Same survivor set (**f ≤ 2% of prompts**) from a different
rank definition. ⇒ **§15.29.9's bound excludes a shared channel present in more than a few percent of
prompts; it does not exclude one confined to ≲2%.** As written §15.29.9 could be read as unconditional
— it is not, and the qualifier belongs in any sentence that leaves this file.

**Finding 6 — G2's corrected floor, and the rule it had just violated.** Its earlier mean-rank floor of
0.02% came from a script using `torch.cdist` (L2/fp64) while the conclusion lives in cos/fp32 — i.e. it
calibrated outside the cited cell, the exact rule it had written down one message earlier. Corrected
floor in the correct cell: **0.010% (mean rank, fp32/cos)**, mutually consistent with my
**< ~0.015%** upper bound from §15.29.9. ⭐ **Calibrating in the wrong cell is the same error class as
importing another cell's null — both were live here at once.**

**Numbering:** G2 renumbered its entry to form **112** because it, the coordinator and I concurrently
claimed 109–111 for the same findings. ⭐ **Concurrent agents writing to one numbered collection
collide silently; a shared numbered list needs an allocation check, not goodwill.**

### 15.29.11 ⛔⛔⭐⭐⭐ The gate I recommended is vacuous, the headline 16 is not a property of the data, and my own tie counter was inert. Three instrument defects, none of which changes the verdict.

All three verified independently: `/tmp/latentuse/{identity_vacuous,tiebreak,mytie_inert}.py`.

**Defect 1 — ⛔⛔ the midrank identity cannot catch a transposition, and I recommended it as a gate.**
Reversing a ranking is `r → n+1−r`, whose mean is still `(n+1)/2`. Measured on real fp32/cos sciq:

| implementation | identity (mean midrank over 199 candidates) | own-prompt mean midrank |
|---|---|---|
| correct | **100.000000** | 97.899 |
| clean single transposition | **100.000000** | 102.101 |
| my v1 (two coupled bugs) | 99.069569 | 101.201 |

`|identity(correct) − identity(transposed)| = 0.00e+00` while the **max per-entry** disagreement is
**198.0 = n−1** — the fingerprint of mutual reversals. Two entirely different rankings, identical gate
reading. My v1 violated the identity only because of the **second** (mate-indicator) bug; a clean
transposition sails through. ⭐⭐ **Any statistic invariant under `r → n+1−r` — mean, sum, variance —
cannot detect a direction error**, so "the identity held in all six cells" was never evidence of
correctness, and §15.29.10 has been corrected to say so. ⭐⭐⭐ **I proposed this gate to a peer one
message after writing down "a check that cannot fail is not a check"; the gate was in the same family
as the defect it was meant to guard.** What actually fired was the hand-checked toy row, because a
hand-computed expected *vector* is asymmetric where a mean is not.

**Replacement gate (asymmetric by construction, verified to fire):** the most-similar candidate must
have midrank ≈ 1 — correct **1.038 PASS**, clean transposition **198.962 FAIL**, my v1 **198.917 FAIL**.

**Defect 2 — ⛔⛔ the observed 16 is order-dependent; the order-free value is ~15.5.** `argmax` breaks
exact ties by lowest column index, and in fp32 `cos ≈ 1` cancels catastrophically, so wins are decided
at ~1e−7. Over 60 random column permutations (n=1788):

| cell | first-index | reversed | random permutations | within-1-ULP margins | stable? |
|---|---|---|---|---|---|
| fp32/cos | **16** | 15 | **15.53 ± 0.50** [15,16] | 207 | ⛔ order-dependent |
| fp64/cos | 14 | 14 | **14.00 ± 0.00** | 394 | ✅ stable |
| fp64/L2 | 11 | 11 | 10.77 ± 0.98 [9,13] | 0 | ⛔ order-dependent |

⇒ ⭐⭐ **the nominal p = 0.0215 is the most favourable reading of an order-dependent integer**; the
order-free count gives p ≈ 0.0215–0.0406. ⭐ **My fp64/cos row is the order-stable one and should be the
quoted cell.** G2 reports the same (15.52 ± 0.50; L2 10.74), and that the *null* is order-independent
(8.926 vs 8.943 at 4000 reps — its own 400-rep claim of order-dependence retracted), so the artefact
inflates the **observed** count only. This *strengthens* §15.29.9: the one nominally significant cell
was partly a tie-break artefact.

**Defect 3 — ⛔⛔ my own exact-tie counter was inert and I printed its output as a measurement.** I wrote
`second = row.masked_fill(row == mx, -inf).max()` and counted `mx − second == 0`. `masked_fill` deletes
**every** copy of the max, so `second` is the *third*-largest and the margin **can never be 0**.
Positive control on `[0.5, 0.9, 0.9, 0.1]`: a deliberate exact tie, my predicate returns **False**.
So my printed "exact-tie wins 0" was an always-false check. Correct predicate `#{entries == max} > 1`:
**105/1788** queries in fp32/cos have ≥2 candidates exactly at the max, **1** of which includes the
own prompt — reproducing G2's "1 win at margin exactly 0", which my instrument could not have found.
⭐ **A tie counter must be positive-controlled on a constructed tie; "0 ties" is the reading an inert
counter gives.** (Form 26/40 family: `masked_fill` on a value-equality predicate silently removes the
thing being counted.)

**Tie counts reconciled — never a disagreement, three different predicates.** G2's 1788/1788 and 37/1788
are **tie-anywhere-in-row**; my 1331 is **tied-with-own**; **tied-at-argmax** is 105 (fp32/cos) and 0
(fp64/cos). Only **tied-with-own** moves the own-prompt rank, so that is the predicate to print beside a
rank statistic; it also explains why strict and midrank coincide at 99.603 in fp64/cos (1 tie).
⭐ **Three integers, three predicates, one dataset — print the predicate with the count.**

**Process:** G2 renumbered to form **115** (coordinator took 113/114) and disclosed it had clobbered
~84 KB of the shared file earlier by read-modify-write. ⭐⭐ **Concurrent writers on one numbered
collection need an allocation check and append-only edits, not goodwill.**

### 15.29.12 ⛔⛔ The family-wise verdict is WITHDRAWN as unstateable. Two more instrument defects (one the crowded null for the THIRD time), and two more of my own.

G2 relayed the coordinator's retraction of "every family definition is above 0.05": the real span is
**0.029–0.116 and it crosses 0.05**. Rechecked independently in `/tmp/latentuse/fw_recheck2.py`.

**Their two defects (accepted, not re-derivable by me — I do not hold their family definitions):**
1. The coordinator's max-statistic null did **not** mask `target == mate`. In mate-kept cells the mate
   wins 922–958/1788, so each replicate collected ~4.6–4.8 spurious own-prompt hits, inflating the null
   max by **+1.133**: FW p **0.0455 printed as 0.0907**. ⭐⭐⭐ **This is the crowded-null failure for the
   THIRD time in one investigation** (mine in §15.29.2, G2's, now the coordinator's) — and note the
   direction: it made the result look **more conservative** and *still* flipped the verdict.
   ⭐ **A defect that biases toward your own null is not harmless.**
2. Its script printed "every one of them is > 0.05" **directly above a table containing 0.0350 and
   0.0455** — a hardcoded string, not derived from the numbers. ⭐⭐ **That is the sentence that reached
   me and that I copied into §15.29.10.** A verdict sentence must be computed from the array it
   summarises, or it is decoration. (Form 2/47 family: the printed conclusion is not the measurement.)

**What the tie-break does to the family (G2's numbers):** both significant families take min p = 0.0215,
which is the fp32/cos **first-index** count of 16 — the count we jointly showed is order-dependent.
F1 (4 cells): first-index FW **0.0287 significant** → order-free **0.0543 not significant**. F2 (8
distinct cells): **0.0393** → **0.0833**. ⇒ ~~⭐⭐ **the flip is carried by the tie-break convention, not
by family size: every family reaching p<0.05 does so only through a column-ordering artefact.**~~
⛔⛔ **THAT ATTRIBUTION IS WRONG AND IS RETRACTED — the two factors were transposed. See §15.29.14, where
I varied both separately.** And F1 order-free straddles the line across independent nulls (0.0595 ±
0.0096, range [0.0487, 0.0725]).
⭐⭐⭐ **When a p-value straddles the threshold across independent nulls, the verdict is a property of
the null sampling, not of the data** ⇒ state the family, the tie-break convention and the range; never a
bare verdict. §15.29.9 and §15.29.11 have been patched accordingly.

**My own two defects this round:**
3. ⛔ **My first order-free estimator double-counted.** It read fp64/cos as **28.00** against a
   first-index 14 — exactly 2× — in a cell with **zero** ties, which is impossible: with no ties the
   order-free count must equal the first-index count. I had added the untied hits and then re-added the
   `1/k` term over all own-at-max rows. Caught by a **hand-checkable self-test**
   (`own_at_max=[T,T,F,T], k=[1,2,3,4] ⇒ 1+0.5+0+0.25 = 1.75`) plus an **independent cross-check against
   the 60-permutation means**, which the fixed version passes: 15.50 vs 15.53 (fp32/cos), 14.00 vs 14.00
   (fp64/cos), 10.83 vs 10.77 (fp64/l2). ⭐ **An analytic shortcut must be gated against the brute-force
   measurement it replaces, and "equals the naive count when there are no ties" is a free unit test.**
4. ⛔⛔ **My PLAN.md integrity check after a concurrent write was always-false.** Worried that G2's
   disclosed ~84 KB read-modify-write clobber had eaten my sections, I ran
   `grep -c "^## §15.29.8"` and got **0 for all four sections** — then noticed every load-bearing number
   was still present. The headings are `### 15.29.8`, with **no section sign and three hashes**: my
   predicate could not match anything. All four sections are present exactly once; the NEG control
   (`### 15.29.99` ⇒ 0) and the POS control (`^## §15.29` ⇒ 1) now both run. ⭐⭐ **A
   "did-my-work-survive" check that returns 0 is indistinguishable from total loss — positive-control it
   against a heading you know exists BEFORE believing it.** I very nearly reported data loss that had
   not happened.

**My 6-cell family, for the record:** FW p **0.1112 ± 0.0133** over 5 null seeds, range [0.0887, 0.1227],
min per-cell p 0.0240 — and **identical under both tie conventions**, because for integer null draws
`P(X ≥ 15.5) = P(X ≥ 16)`. ⭐ **An order-free observed count does not by itself change a discrete
p-value; it changes which integer you compare against.** D4 controls asserted: NEG (observed := per-cell
null median) FW p **0.9687**, POS (observed := +40 injected hits) FW p **0.0000**.

**⭐⭐ What this does NOT touch.** The own-prompt **at-chance** conclusion never rested on the
family-wise statistic: it rests on the crowded null (**P(X≤3) = 0.399**) and on §15.29.9's joint test
(**< ~0.015%** homogeneous shared component, sparsity-scoped to f > 2%). Also untouched: the label
finding, the six-cell midrank table, the norm-channel exclusion, the 0.010% floor, z = −0.94, the
0.14–0.27% displacement, and G2's five rows. ⭐ **The reason the verdict survives is that it was never
carried by the contested statistic** — worth noting as the one structural thing that went right here.

**Allocation now:** 113/114 coordinator, 115/116 G2, and the numbered-collection collision stands as its
own lesson.

### 15.29.13 ⛔ Round-5 close-out audit: THREE of my own reported numbers were artefacts of how I asked. Two are label counts I put in a handback; one is the compliance "0" I was already warned about.

Ran before handing back, precisely because §15.29.11/§15.29.12 are about my own instruments.

**1. ⛔ "`tab:latent-use-pending` is defined 2× in `appendix.tex`" — WRONG, it is defined once.**
Measured: `\label{tab:latent-use-pending}` **1** (`appendix.tex:719`), `\ref{tab:latent-use-pending}`
**1** (`appendix.tex:712`). I had counted *occurrences of the string*, which mixes the definition with
its own cross-reference. Whole-file check: 93 `\label{}` occurrences, 93 unique, `uniq -d` **empty** ⇒
**no duplicate label anywhere in that file**. ⭐ **"Defined N times" needs the `\label` predicate, not
the name predicate** — a label and its `\ref` are the same string.

**2. ⛔ "68 of 93 labels absent from `main.tex`" was one of three different numbers, and I quoted a
fourth.** The same question has three defensible predicates and they do not agree:

| predicate | count / 93 |
|---|---|
| not defined as `\label` in `main.tex` | **68** |
| never `\ref`/`\cref`/`\autoref`'d from `main.tex` | **73** |
| neither defined nor referenced | **68** |
| naive `label not in main.tex` substring test | **67** ← what I actually reported (as 67, then 68) |

The substring test is the odd one out because a short label can appear inside a longer one. POS control:
3 labels shared by both files are correctly excluded from the "absent" set; NEG control: a fabricated
`tab:zzz-fabricated` is flagged by all four predicates. ⭐ **Cited as "68 of 93", it was a substring
count reported as a label count** — same family as §15.29.11's tie counts: *one question, three
predicates, three integers*. The conclusion is unaffected and in fact stronger under every predicate:
**`main.tex` contains 0 `\input{}` and 0 `\include{}`** (POS control: the same grep returns 13 on
`paper_ICLR27/EED/paper/main.tex`, so the predicate is live), and `tab:latent-use-pending` is **absent
from the compiled `main.aux`** (POS control: `tab:longcontext` appears 6×; NEG control: fabricated label
0×) ⇒ **nothing written to the orphan `appendix.tex` reaches `main.pdf`.** That is the blocker, and it is
now confirmed on the compiled artefact rather than on the source.

**3. ⛔⛔ My compliance figure was window-dependent AGAIN — the third time, and §15.29 already contains
the lesson.** Reported "nothing modified" all round. Re-run with the window printed:

| window | files under `{results, outputs*, latentuse_g2_out, latentuse_g3}` |
|---|---|
| newer than 2026-09-24 | **13** |
| newer than 2026-09-23 | 140 |
| newer than 2026-09-22 | 250 |
| total visible (finder POS control) | 2059 |

All 13 enumerated: 2 are G3's (`latentuse_g3/g3_checks_full_refeed.json` 00:00,
`latentuse_g3/g3_job_id.txt` 00:13 — the latter's tail is G3's own job-id log), and 11 are hydra configs
and `train.log` files from the 03:11–03:12 `state_hijacking_dit` / `dynlowrank` training runs that own
those directories. **`grep -c latentuse` over the modified set = 0** for the `results`/`outputs*` part,
and my own writes are confirmed live by POS control (1 file under `scripts/eval/latentuse` = this
PLAN.md, 246 under `/tmp/latentuse`) ⇒ the timestamp predicate is not inert and **none of the 13 is
mine**. The three `z_cache` files I read are unchanged at 09-23 20:08 / 20:48 / 23:36, md5
`82ea3622fc4a…` / `2062e8d57a3f…` / `8cd61914248b…` (newly recorded here — they were **not** in this
file before, so any earlier "z_cache unchanged" claim had no baseline to compare against).
⭐⭐ **"I modified nothing" is only checkable against a named window, root and predicate; a bare 0 is
unfalsifiable, and on a shared tree with sibling writers it is also nearly always false.**

**D4 re-asserted this round:** `sanity_checks.py` md5 `891bf67a2e4d…`, exit **0**, **TP=34/34, FP=0/9**
across 9 checks and 34 positive controls (`/tmp/latentuse/d4_round5.log`). Protected md5s unchanged:
`35dc33a3…` (runner), `b07591ee…` (examples), `911bd2d9…` (scorer), `4bd16294…` (appendix.tex),
`32f46730…` (main.tex).

### 15.29.14 ⛔⛔⭐⭐⭐ I attributed the family-wise flip to the wrong factor, because I varied only one. The full grid says FAMILY SIZE carries it — and my own "identical under both conventions" had the same unmatched-sides defect I then credited G2 for finding.

G2 retracted its tie-break explanation and I had copied it into §15.29.12. Rather than swap one
assertion for another, I ran the grid its own lesson demands: **vary both factors separately and print
both tables** (`/tmp/latentuse/twofactor.py`, log `twofactor.log`).

**THE GRID — family size (rows) × tie-break convention (cols), 6 cells, 5 independent nulls × 2000 reps,
both sides matched:**

> ⛔⛔ **EVERY NUMBER IN THE TABLE BELOW WAS COMPUTED ON A WRONGLY-CONSTRUCTED NULL AND IS SUPERSEDED
> BY §15.29.16.** The null drew an independent relabelling **per cell**; it must draw once per rep and
> feed all cells. The *direction* of the grid survives; the magnitudes are ~2× too large and **two of the
> four "n.s." verdicts become SIG** on the correct null. Kept here only so the correction is auditable.

| family | first-index | order-free | min-p cell |
|---|---|---|---|
| ~~F1 (2 cells: fp32/cos, fp64/cos)~~ | ~~0.0397 ± 0.0056 SIG~~ | ~~0.0431 ± 0.0070 SIG~~ | fp32/cos |
| ~~F2 (3 cells: + fp64/l2)~~ | ~~0.0578 ± 0.0049 n.s.~~ | ~~0.0661 ± 0.0095 straddles~~ | fp32/cos |
| ~~F3 (4 cells: + fp32/l2)~~ | ~~0.0770 ± 0.0051 n.s.~~ | ~~0.0872 ± 0.0132 n.s.~~ | fp32/cos |
| ~~F4 (6 cells: + both dot)~~ | ~~0.1117 ± 0.0098 n.s.~~ | ~~0.1213 ± 0.0195 n.s.~~ | fp32/cos |

⇒ **Down a column the verdict flips (SIG → n.s.); across a row it never does.** The convention costs
+0.003 to +0.010; ~~family size costs +0.072~~ → ⛔ **that magnitude is an artefact of the defective null;
on a shared null it is +0.0255 (§15.29.16).** ⭐⭐⭐ **"A and not B caused the flip" is unsupportable
until A and B are varied separately, and here the single-factor answer was not merely incomplete — it
named the wrong factor.** D4: NEG (observed := per-cell null median) FW p **0.9625 PASS**, POS (+40
injected hits) **0.0000 PASS**, denominator printed (6 cells × 2000 reps × 5 seeds).
⛔ **Both D4 controls passed on the defective null** — see §15.29.16 for why neither they nor an FPR
measurement can see this class of error.

**⛔⛔ And my own headline claim had G2's defect in it.** I wrote "identical under both tie conventions,
because `P(X ≥ 15.5) = P(X ≥ 16)` for integer draws." The identity is real **but it is a property of the
NULL'S SUPPORT, not of the statistic** — and my `fw_recheck2.py` computed the null as
`(W == pick).sum()`, an **integer count**, while switching the observation between conventions. That is
exactly the unmatched-sides defect I credited G2 for finding, in my own script
(`/tmp/latentuse/matched_sides.py`, 6000 reps):

| cell | obs (order-free) | p vs integer null | p vs matched null | Δ | frac. null reps |
|---|---|---|---|---|---|
| fp32/cos | 15.5000 | 0.0227 | 0.0247 | +0.0020 | 0.496 |
| fp64/cos | 14.0000 | 0.0718 | 0.0718 | +0.0000 | 0.000 |
| fp32/dot | 9.0000 | 0.5392 | 0.5392 | +0.0000 | 0.000 |
| fp64/dot | 9.0000 | 0.5393 | 0.5393 | +0.0000 | 0.000 |
| fp32/l2 | 10.8333 | 0.2883 | **0.2580** | **−0.0303** | 0.781 |
| fp64/l2 | 10.8333 | 0.2887 | **0.2600** | **−0.0287** | 0.762 |

And the identity itself: on the integer null P(≥15.5) = P(≥16) = 0.0227 **exactly equal**; on the matched
order-free null 0.0247 vs 0.0185, **not equal**. ⭐⭐ **The defect is self-concealing in precisely the
cells that cannot expose it** — a cell with zero ties has an integer observation and an integer null, so
the two conventions agree there *by construction*, and 4 of my 6 cells are like that. Only the tied
cells can move, and they are the same cells whose observation is fractional.
⭐⭐⭐ **A "the two conventions agree" result computed on mostly-untied cells is a tautology wearing the
costume of a robustness check.**

**Constructive self-test, now passing on both sides.** A cell with **0 tied rows must give
order-free == first-index bit-for-bit**. Measured: fp64/cos 14 vs 14.0000 ✅, fp32/dot 9 vs 9.0000 ✅,
fp64/dot 9 vs 9.0000 ✅ (0 ties each); fp32/cos 16 vs 15.5000 and both l2 cells 11 vs 10.8333 (105 and
263 tied rows). ⭐ **This is the test my 28.00-vs-14 double-count failed and that no amount of
Monte-Carlo agreement would have caught** — it is exact, free, and derivable from the definition.

**Three integers for the same cell were never a disagreement.** My 0.0215, my 0.0265 and G2's 0.0193 for
fp32/cos: over 8 independent nulls at 6000 reps, first-index p = **0.0220 ± 0.0018** [0.0190, 0.0245] and
order-free p = **0.0243 ± 0.0013** [0.0225, 0.0262]. All three quoted values lie inside
[0.0190, 0.0262]. ⭐ **Null-sampling spread, not a dispute** — and it is ±0.002 on a number we were
comparing against 0.05, so the per-cell p was never the contested quantity.

**⚠ Families with the same size are not the same family.** G2's "F1 (4 cells)" gives 0.0391 ± 0.0045 and
my F1 (2 cells) gives 0.0397 ± 0.0056. Numerically indistinguishable, **different member sets**.
⭐ **Equality of two FW p-values is not evidence that two families coincide** (same trap as the two
different "16-cell" families in the memory file, which shared only 4 members). Quote the member list.
⛔ **The instance I chose is wrong: G2's 4-cell family and my 4-cell family have the SAME membership**
(G2 established this; the 2× gap between 0.0770 and 0.0391 was the null construction, §15.29.16). The
general warning stands and the memory file's 16-cell case still supports it; **this file's own numbers do
not**. ⭐ **A true general lesson illustrated with a false instance is still a reporting defect** — and I
reached for the instance because it was in front of me, not because I had checked it.

**⛔ My form-enumerator reported 24 gaps that do not exist — and its own NEG control is what stopped
me.** Checking G2's claim about the memory topic file (`inert-validation-checks.md`, 2204 lines), my v1
printed **GAPS [5..29]** — 24 of them. All false. Cause: the bare numbered list is
`5. ⭐⭐**text**`, and my `^(\d+)\.\s*\*\*` cannot match because **the emoji sits between the dot and the
bold marker**. This is the fifth-style trap from the memory file one level down: I had internalised
"emoji before 形态" and missed "emoji before `**`". What saved me was that **v1 also failed its own NEG
control** (2 inline-reference lines counted as definitions) — I debugged that first and found the NEG
control was itself mis-specified: the two lines were `### 109.` and `### 121.`, **real definitions that
merely contain a back-reference**. ⭐⭐ **A failing control is worth stopping for even when you suspect
the control is the broken thing — it was the only signal that the enumerator was also broken.**
v2 with 6 consolidated structural styles: **121/121 present, range 1–121, ZERO gaps**, 146 definition
sites, 16 numbers carrying more than one. POS controls (1, 5, 12, 20, 29, 38, 42, 47, 94, 116, 119, 121)
all ≥ 1; NEG (500/998/999) all 0; inline-reference lines that are not themselves definitions: 25, of
which **0** wrongly counted. G2's "119/119" was right about there being no gaps; the top is now **121**.

**⭐ One more control that needed an injection to mean anything.** My "is the structural anchor
load-bearing?" NEG printed *"anchor makes no difference"* — because on this file every inline-referenced
number also has a real definition, so there was no gap available to be masked. Proved it by **deleting
form 42's definition**: the anchored enumerator reports 42 present on the intact file and **absent** on
the broken one (TP), while the loose `形态 N` regex reports it present **both times** — the inline
reference fills the hole. ⭐⭐ **A control that cannot distinguish on your data is not a passing control;
inject the defect it is supposed to catch.** (Injection done in memory on a copy of the lines; the
memory file was never modified — md5 `1ce8ed71d476925ea7e045be4a48c44e` unchanged.)

**Net effect on the conclusion: none.** The own-prompt verdict still rests on the crowded null
(**P(X ≤ 3) = 0.399**) and on §15.29.9's scoped bound, neither of which is a family-wise statement. What
changes is what may be *said* about the FW number: **it is stateable per family, and the 0.030–0.121
span is a family-size effect, not a tie-break artefact.** Any text of mine claiming the tie-break
carries it is retracted (§15.29.12 patched in place).

### 15.29.15 ⛔⛔⭐⭐⭐ My enumerator HAS the gap-masking defect a peer credited it with not having, and my 146 was an over-count. Plus: my single-draw NEG control could not see the decision band, so I calibrated the FW procedure's own FPR.

Two follow-ups, both triggered by claims I could check rather than adopt.

**1. ⛔⛔ I accepted credit for a guard I do not have.** G2 wrote that my enumerator's "region guard is
exactly the thing I was missing." Checked: `grep -n '417\|region\|legacy' /tmp/latentuse/forms_enum2.py`
returns **nothing**. The guard is in *G2's* landed script (`/tmp/coordcheck/enumerate_forms.py`, mtime
04:26, i.e. written **before** my v2) and my script never had one. Reconciling the site counts:

| enumerator | definition sites | distinct numbers | numbers with >1 site |
|---|---|---|---|
| mine (no region guard) | **146** | 121 | 16 |
| guarded (G2's landed script) | **131** | 121 | 10 |

The 15 extra sites are all **numbered-list items in later discussion prose** (`1. **可达 p 值…`,
`2. ⛔**更深的一层…`, `4. ⭐⭐**三种约定的判决…`) that my bare-`N.` style swallowed because I applied it
to the whole file, while the bare-list era ends at **line 417**. **G2's 131 is right and my 146 was
wrong.**

⭐⭐⭐ **And the defect is in the dangerous direction.** Injection test: delete form 4's only real
definition (`:36`) and ask each enumerator whether it notices. **Guarded: detects the gap. Mine: still
reports form 4 present — the gap is MASKED**, because a stray `4. ⭐⭐**…` in a discussion list 2000 lines
later fills the hole. An unguarded low-number style **can never create a false gap, only hide a real
one**. So my "121/121, ZERO gaps" was *correct by luck*: for any form inside the guard's range my
instrument was incapable of reporting otherwise. ⭐⭐ **"My gap report agrees with yours" is not
evidence both instruments work when one of them cannot produce the disagreeing answer.**

⭐ Also worth naming: G2's "eight styles, three new" retraction says its genuinely-new count was **0/3** —
it had renamed existing classes. My v2's "6 consolidated styles" is the same act in the other direction:
styles F (`^- **N.`) matched **0** lines, so I shipped a class with no members. ⭐ **A style table must
print per-style hit counts; a zero row is either a dead class or a missed region.**

**2. ⛔ My D4 NEG control for the FW estimator sat at the centre of the null, not in the decision band.**
I reported NEG FW p = 0.9625 as a passing control. But that NEG sets *observed := the per-cell null
median*, which is by construction the **centre** of the distribution — it lands far from 0.05 no matter
what the estimator does near the threshold, which is exactly where F1-vs-F4 is decided.
⭐⭐⭐ **A single negative control at the centre of the null says "this one draw did not misfire"; it
calibrates nothing in the band that carries the verdict.** Replaced with a real FPR measurement
(`/tmp/latentuse/fw_fpr.py`): draw the "observation" **from** held-out null reps, compute FW p against
the rest, 300 trials per family, 4000 null reps split 2000/2000.

| family | observed FW p | FPR at .05 | FPR at .10 | verdict |
|---|---|---|---|---|
| F1 (2 cells) | 0.0425 | 0.033 | 0.067 | CALIBRATED |
| F2 (3 cells) | 0.0655 | 0.023 | 0.093 | CALIBRATED |
| F3 (4 cells) | 0.0825 | 0.043 | 0.077 | CALIBRATED |
| F4 (6 cells) | 0.1215 | 0.030 | 0.097 | CALIBRATED |

⇒ **the FPR does not drift with family size** (0.023–0.043 against a nominal 0.05, all slightly
conservative), so the SIG→n.s. movement down the column in §15.29.14 is **a property of the data and the
family, not of the estimator**. ⛔⛔ **"That is the check that licenses §15.29.14's conclusion" is WRONG
and is retracted.** The null those FPRs calibrate is itself mis-constructed, and **an FPR check cannot
detect that** — it asks "is this test calibrated to its own null", never "is this null the null of this
statistic". Measured both ways: shared 0.0467, independent 0.0433, **both pass**. See §15.29.16.
For contrast, 200 NEG draws taken from random null reps give mean 0.5331 with **3.0% below 0.05** —
the honest calibration — versus my single median-based 0.9670. POS (+40 hits) 0.0000 PASS.
⭐ Side result: mean off-diagonal null correlation across the 6 cells is **+0.0011**, so here a Šidák
correction would have been roughly defensible — ⛔ **also retracted: +0.0011 is the correlation of the
defective independent null. On the correct shared null it is +0.3916 (max +1.0000), so Šidák would have
been indefensible by ~3× here too.** ⭐⭐ **A measured assumption-check inherits every defect of the
object it measures; "I measured it rather than assuming it" is not a defence if I measured the wrong
object.** The max-statistic null needs no such assumption, which is the reason to prefer it.

**Unchanged by both items:** the own-prompt at-chance verdict (crowded null P(X ≤ 3) = 0.399) and
§15.29.9's scoped bound. The memory topic file was never modified by me — all injection tests ran on
in-memory copies of its lines; it is being appended concurrently by its owner (2204 → 2209 lines during
this round), which is itself why I re-stat it rather than cache a line count.

### 15.29.16 ⛔⛔⭐⭐⭐ All FOUR of my family-wise scripts drew the null per-cell, so every FW magnitude I have reported is ~2× too large and two "n.s." verdicts were really SIG. My FPR calibration is structurally incapable of seeing this. And the axis I called "family size" is not family size.

G2 localised a defect at `/tmp/latentuse/fw_fpr.py:52-53`. Verified in my own source, and it is **worse
than reported**: four scripts have it, and in three the cell loop is the **outer** loop, so each cell gets
a separate rep *stream*, not merely a separate draw.

| script | line | nesting | severity |
|---|---|---|---|
| `fw_fpr.py` | 52–53 | draw inside `for n in NAMES:`, itself inside `for r in range(REPS)` | per-rep, per-cell draw |
| `twofactor.py` | 86–90 | `for _,_,n in CELLS:` is **outer**, `for r in range(reps)` inner | **separate stream per cell** |
| `matched_sides.py` | 50–53 | same outer-loop form | **separate stream per cell** |
| `fw_recheck2.py` | 57–60 | same outer-loop form | **separate stream per cell** |

**Why it is wrong, from the definition.** The six cells are six similarity functions evaluated on **one**
set of 1788 (query, target) pairs. Verified as a precondition rather than assumed: `O` and `M` are
array-equal across all 6 cells, else the script raises — a shared draw is not even well-defined
otherwise. So the relabelling that defines the null is **one object per rep**, fed to every cell. An
independent draw per cell lets a redundant cell act as a free extra draw, which inflates the
max-statistic null and deflates FW p.

**⛔ My own script printed the correct design while implementing the wrong one.** `fw_fpr.py:47-48`
prints *"shared permutations across cells, which is the correct matched design -- the cells are the same
1788 queries"*, three lines above the draw that is inside the cell loop.
⭐⭐⭐ **A script's own explanatory print is the least reliable documentation in the repository: it was
written from intent, the code from habit, and only the code produces the number.** This is the same
shape as the memory file's "script prints a conclusion sentence its own table refutes" — there the prose
was hardcoded, here the prose was *correct* and the code was not. **Both directions happen; the fix is
the same — derive the claim from the object and assert the derivation.**

**THE GRID, RECOMPUTED (`/tmp/latentuse/shared_null_grid.py`, log `shared_null_grid.log`;
8 seeds × 20000 reps for the resolved version in `shared_null_followup.py`):**

| family | first-index (shared) | order-free (shared) | first-index (defective) | order-free (defective) |
|---|---|---|---|---|
| F1 (2 cells) | **0.0257 ± 0.0026 SIG** | **0.0304 ± 0.0011 SIG** | 0.0401 SIG | 0.0410 SIG |
| F2 (3 cells) | **0.0348 ± 0.0047 SIG** | **0.0414 ± 0.0015 SIG** | 0.0577 straddles | 0.0616 straddles |
| F3 (4 cells) | **0.0348 ± 0.0047 SIG** | **0.0414 ± 0.0015 SIG** | 0.0733 straddles | 0.0823 n.s. |
| F4 (6 cells) | 0.0504 ± 0.0077 **straddles** | 0.0608 ± 0.0042 n.s. | 0.1050 n.s. | 0.1136 n.s. |

⇒ **The direction survives and the verdicts change.** F2 and F3 were "n.s." on the defective null and are
**SIG** on the correct one (+9.14 se below 0.05, per-seed range 0.0288–0.0407). Only F4 is not
significant, and under first-index it **straddles**: mean 0.0504, −0.16 se from 0.05, per-seed
0.0392–0.0614. ⭐ **A quantity 0.16 se from the threshold has no verdict; reporting "n.s." for F4/first-index
would be reading the sign of noise.** Marginal effects on the correct null: family size **+0.0255**
[+0.0234, +0.0277], convention **+0.0050** [+0.0006, +0.0071], ratio **5.1×**. So the earlier
attribution direction was right and my magnitude was **2.7× too large** (+0.072 → +0.0255); on the
defective null the two effects were +0.0688 vs +0.0056.

**⛔⛔⭐⭐ "Family size" is the wrong name for the axis, and F2 ≡ F3 is a theorem.** The two l2 cells are
**bit-identical in structure** on the real data (`W`, `ATMAX`, `K` all array-equal), as are the two dot
cells — 2 of 15 pairs. On a shared null that identity holds **rep by rep**: max|fp32/l2 − fp64/l2| =
**0.000e+00 over 20000 reps**, both conventions. F3 = F2 + fp32/l2 and F2 already contains fp64/l2, so
F3 **cannot** differ from F2 — the exact equality in the table is forced, not a coincidence:

| family | listed cells | **distinct** statistics |
|---|---|---|
| F1 | 2 | 2 |
| F2 | 3 | **3** |
| F3 | 4 | **3** |
| F4 | 6 | **4** |

⭐⭐⭐ **What costs family-wise p is the number of DISTINCT statistics, not the number of listed cells: a
redundant member is free on a correct null and expensive on an independent one** (on the per-cell null
the same pair differs by up to **17** hits per rep — ⭐ **the defect literally manufactures information
that the data does not contain**). My F1→F4 "+0.0255 for family size" is therefore properly
"+0.0255 for going from 2 to 4 distinct statistics"; the 4→6 listed step costs **nothing**.

**THE GATE THAT CATCHES THIS — and its exact scope.** Two cells that are bit-identical on the real data
must have null correlation **~1.0**; no reps, no threshold, derivable from the definition:

| check | shared null | per-cell null | verdict |
|---|---|---|---|
| corr(fp32/dot, fp64/dot), first | **+1.0000** | +0.0288 | POS fires |
| corr(fp32/dot, fp64/dot), order-free | **+1.0000** | +0.0288 | POS fires |
| corr(fp32/l2, fp64/l2), first | **+1.0000** | −0.0157 | POS fires |
| corr(fp32/l2, fp64/l2), order-free | **+1.0000** | −0.0252 | POS fires |
| denominator | 15 pairs examined, 2 degenerate | — | raises if 0 degenerate |

NEG = shared null passes (4/4), POS = per-cell null fires (4/4). The script **raises** if no degenerate
pair exists, so the gate can never be reported as "passing" when it has no subject.
⭐⭐ **A check whose subject can be absent must refuse to report a verdict, not report success.**

**⛔ But I tested the gate's scope, and it is narrow.** Run against a *different* wrong null — draws that
fail to exclude the noise mate, the crowded-null defect I have hit three times — the gate is **SILENT**:
correlation +1.0000 in both designs, while the answer does move (F1 0.0205 → 0.0250). The defect is real
(P(draw == excluded mate) 0.00000 correct vs **0.00499** broken, against an expected 0.005) and needs its
own gate. ⭐⭐⭐ **A gate that caught one wrong null is evidence about that null, not a general
"is-this-the-right-null" check** — degeneracy is invariant to anything applied identically to all cells,
which is exactly what mate-inclusion is. **Each null defect needs a check derived from that defect.**

**Why my round-6 FPR could not have found any of this.** It draws the pseudo-observation from the same
null it tests against, so it measures self-consistency: shared **0.0467**, independent **0.0433**, both
"CALIBRATED". ⭐⭐⭐ **An FPR check answers "is this test calibrated to its own null" and never "is this
null the null of this statistic"; a wrong null is perfectly calibrated to itself.** §15.29.15's claim
that the FPR "licenses" the grid conclusion is retracted in place. A second casualty: the
**+0.0011 off-diagonal correlation** I used to say Šidák "would have been roughly defensible" is the
defective null's correlation; on the shared null it is **+0.3916, max +1.0000**, so Šidák would have been
wrong by ~3× here as well — the memory file's +0.912 case was not the exception I treated it as.

**D4 on the FW estimator, shared null**, denominator 6 cells × 4000 reps × 5 seeds = 120000 draws:
POS (+40 hits) **0.0000 PASS**; NEG at null median **0.7708 PASS**; NEG at the null's **95th percentile
0.1435 PASS** — the last is the only one that says anything about the p ≈ 0.05 band where the verdicts
live. Plus two new self-tests that run before any data is touched: the vectorised survival transform is
asserted equal to the naive O(n²) form (max diff 0.0e+00) **and** asserted *unequal* under
`side='right'` (0.40), so the assertion is demonstrably able to fail.

**Net effect on the paper-facing conclusion: still none.** The own-prompt at-chance verdict rests on the
crowded null (**P(X ≤ 3) = 0.399**) and §15.29.9's scoped bound, neither of which is family-wise. What
changes is **what may be said about the FW number, for the third round running**: the defensible
statement is now **"significant for every family of ≤ 4 distinct statistics (0.026–0.041), with the
6-cell family straddling 0.05"** — the opposite verdict from §15.29.14's table for F2/F3.
⭐⭐⭐ **Three consecutive rounds of instrument repair moved this number in three directions
(0.0770 → 0.0348, n.s. → SIG). The number is not the finding; the number is the least stable thing here,
and anything resting on it should be stated as resting on it.**

### 15.29.17 ⛔⛔⭐⭐⭐ The ENTIRE family-wise result rests on one cell, and that cell's edge is float32 rounding. Exact arithmetic on the same metric gives p = 0.0730, not significant. Also: my F1–F4 ladder was hand-picked where the rule says enumerate 2^k.

Three of G2's claims were checkable; all three confirmed, and the enumeration it did and I
did not turns the grid into a much simpler and much worse result
(`/tmp/latentuse/lattice_and_fp32cos.py` → `lattice.log`, `fp32cos_margins.py` → `margins.log`).

**⛔ My own defect, unprompted: F1/F2/F3/F4 was a HAND-PICKED ladder.** The memory file's rule
is explicit — *a hand-picked combination list is a preference entry point; enumerate 2^k*. G2
enumerated 63 subsets; I enumerated **4**, and I did it while quoting that same rule at other
people. ⭐⭐ **Citing a rule is not applying it, and the ladder I chose happened to be the one
that produced a clean monotone story.**

**THE FULL LATTICE — 2⁴−1 = 15 families over the 4 distinct statistics, 8 seeds × 20000 reps,
shared null** (640000 draws; `*` = contains fp32/cos):

| family | first-index | order-free |
|---|---|---|
| `*`fp32/cos | **0.0217 ± 0.0011 SIG** | **0.0233 ± 0.0015 SIG** |
| fp64/cos | 0.0724 ± 0.0020 n.s. | 0.0724 ± 0.0020 n.s. |
| fp32/l2 | 0.2914 n.s. | 0.2582 n.s. |
| fp32/dot | 0.5444 n.s. | 0.5444 n.s. |
| `*`fp32/cos, fp64/cos | **0.0263 ± 0.0031 SIG** | **0.0301 ± 0.0018 SIG** |
| `*`fp32/cos, fp32/l2 | **0.0310 SIG** | **0.0364 SIG** |
| `*`fp32/cos, fp32/dot | **0.0391 SIG** | **0.0431 SIG** |
| `*`fp32/cos, fp64/cos, fp32/l2 | **0.0347 SIG** | **0.0413 SIG** |
| `*`fp32/cos, fp64/cos, fp32/dot | 0.0436 straddles | 0.0498 straddles |
| `*`fp32/cos, fp32/dot, fp32/l2 | 0.0482 straddles | 0.0559 straddles |
| `*`all four | 0.0519 straddles | 0.0608 n.s. |
| fp64/cos + any others (4 families) | 0.1035 – 0.1530 n.s. | 0.1061 – 0.1530 n.s. |
| dot/l2 only (3 families) | 0.2914 – 0.5444 n.s. | 0.2582 – 0.5444 n.s. |

**⭐⭐⭐ And it is a THEOREM, not a pattern.** The max-statistic satisfies
**FW p ≥ min per-cell observed p** — asserted, not assumed: 240 (family, convention, seed)
triples, **0 violations**, min slack +0.00000. Per-cell observed p: fp32/cos **0.0214**,
fp64/cos 0.0715, fp32/l2 0.2954, fp32/dot 0.5404. Since only fp32/cos is below 0.05, **no
family excluding fp32/cos can be significant, by construction.** 8/15 families contain it and
range 0.0217–0.0519; the other 7 range 0.0724–0.5444 and **none** is significant under either
convention. ⭐⭐ **A "family-wise" result whose every significant member shares one cell is not
a multiplicity-corrected finding about six measurements; it is one measurement with decoration.**

**⛔⛔⭐⭐⭐ THE CARRYING CELL IS THE ONE WITH BROKEN ARITHMETIC, and its whole edge is rounding.**
fp32/cos is the cell the memory file already flags for catastrophic cancellation in `1−cos`
(first-index 16 / reversed 15 / random column order 15.53 ± 0.50). Decomposed against
**fp64/cos — the same metric in exact arithmetic on the same double-precision inputs**:

| quantity | value |
|---|---|
| fp32/cos observed / fp64/cos observed | 16 / **14** |
| null mean, null sd (identical for both) | 8.99, **3.00** |
| the 2-hit difference, in null sd | **0.667 sd** |
| p, fp32/cos vs fp64/cos | **0.0218 vs 0.0730** — ratio **3.34×** |
| rows won by both | 13 |
| **rows won by fp32/cos only (disputed)** | **3** |
| rows won by fp64/cos only | 1 |
| **exact margin of the 3 disputed rows, in fp32 ULP** | **0.33, 1.41, 0.29** |
| exact margin of a typical losing row, in fp32 ULP | median **79**, p05 25.8 |
| separation | **56×** |

Every disputed row is one where own-prompt is **not** the maximum in exact arithmetic — by
1.7e−08 to 8.4e−08 against an fp32 ULP of 5.96e−08 — and fp32 rounding lifted it to the
maximum. Directional control printed both ways: disputed rows won in fp32 **3/3**, won in
fp64 **0/3**, asserted. ⭐ Rounding is not systematically kind: it *gave* fp32/cos 3 rows and
*cost* it 1, net +2 — noise, whose net happens to be exactly what carries the tail.

⭐⭐⭐ **This is not "two equally valid readings": fp64 is strictly the more accurate evaluation
of the same quantity, so the defensible number is p = 0.0730, n.s.** The significant reading
is the lossy one. ⭐⭐ **A 0.67-sd difference in the statistic became a verdict flip because the
tail is steep there — so "p = 0.021" and "p = 0.073" are one measurement at two precisions,
and any quotation of either must name the precision.**

**⛔⭐⭐ And the tie-break fix I was proud of addressed the SMALLER of two free choices.**
Column order was worth ±0.5 hits (15.53 ± 0.50) and the order-free statistic removes it
properly. **Precision is worth 2 hits and the order-free statistic does nothing about it**
(0.0233 vs 0.0724 — the same 3× gap survives the fix). ⭐⭐⭐ **I closed the free choice I had
already named and declared the class handled; enumerating the choices is a separate act from
fixing one of them.**

**G2's three checkable claims, all confirmed.**
1. **The gate's precondition must be TIE-STRUCTURE equality, not matrix equality.** Measured
max|matrix diff|: dot pair **0.000e+00**, l2 pair **5.627e−09**, cos pair **3.906e−07**. The
l2 pair is degenerate in `W`/`ATMAX`/`K` while its matrices differ — so **"matrices
bit-identical" would have made the gate inapplicable to one of the two pairs it was run on.**
⭐ The cos pair is the in-data NEG at only ~1e−07: r = **0.9105** < 0.99, so the gate separates
it. ⭐⭐ **A precondition stated more strongly than the property that does the work silently
narrows a check to a subset of its subjects.**
2. **The Šidák correlation numbers were never in conflict — different cell LISTS.** Blocks:
l2–l2 **+1.0000**, dot–dot **+1.0000**, cos–cos **+0.9105**, cos–l2 **+0.7427**, cos–dot
**−0.0148**, dot–l2 **−0.0081**. Mean over my 6-cell list **+0.3860**, over G2's 4 cos/l2
cells **+0.8135** (G2: +0.396 / +0.815). ⭐ **The memory file's "+0.912" is the cos–cos BLOCK,
not a whole-family value** — so a mean off-diagonal correlation is as list-dependent as FW p,
and the replacement for my withdrawn +0.0011 is **not a single number either**. Šidák is
indefensible under both lists.
3. **Significance requires fp32/cos** — confirmed, and strengthened from G2's empirical
statement to the theorem above.

**Net effect on the paper-facing conclusion: still none, and the reason is now stronger.** The
own-prompt at-chance verdict rests on the crowded null (**P(X ≤ 3) = 0.399**) and §15.29.9's
scoped bound. But the family-wise number, which has now moved in four directions across four
rounds (0.0770 → 0.0348 SIG → one-cell → **0.0730 in exact arithmetic**), should not be quoted
at all: ⭐⭐⭐ **the defensible statement is that the retrieval signal is not significant under
exact arithmetic, and that every significant version of it is carried by three rows decided
inside float32 rounding error.** That is a negative result about the instrument, and it points
the same way as the paper's negative headline.

### 15.29.18 Round 9 — the shared memory record's `1−cos` mechanism is false; and I broke **two** instruments proving it

G2's closing message contained one item aimed not at me but at the shared record: *"The
memory file attributes the fp32/cos ties to catastrophic cancellation in `1-cos`. That
causal claim is false. I ranked fp32 by `-c`, removing the subtraction entirely: still 16
hits, and identical row-by-row to the `1-c` version (0 rows differ), while fp64 gives 14."*

This is the entry `laces-latent-collapse-champion` / the tie-and-order block that reads
*"fp32/cos `1−cos` 灾难性抵消, first-index **16** / reversed 15 / 200 随机列序 **15.53±0.50**"*.
I checked it and it is false — but the route there cost me two broken instruments, both of
which I am recording because both failed in the direction of what I already believed.

**⛔⛔ My instrument failure F1 — a string match standing in for an operation.**
`/tmp/latentuse/cos_mechanism.py` check (3) asked *"does the producing code perform a
`1−cos` subtraction?"* and implemented it as `("1-c" in src) or ("1-cos" in src) or …`.
It returned **FAIL on a known-good input.** The two matches were at
`lattice_and_fp32cos.py:20` and `:281` — **both are prose in which I quote the memory
record's claim in order to dispute it.** The check reported the presence of the very thing
whose absence I was documenting, because I searched the file's *text* for a proposition
about the file's *behaviour*. This is the recorded root cause of two earlier
non-existence errors (grep on the file text rather than the operation / output name /
key), now reproduced a third time.

Fixed in `/tmp/latentuse/ast_subtraction_check.py`: walk the AST for
`BinOp(op=Sub, left=Constant(1))`. Comments, docstrings and string literals are not in the
AST, so they cannot contribute. Controls, all six passing (`ast_sub.log`):

| control | subject | expected | got |
|---|---|---|---|
| POS | `d = 1 - c` | detected | 1 site |
| POS | `d = 1.0 - c` | detected | 1 site |
| POS (real subject) | inject `1 - (…)` into `lattice_and_fp32cos.py` cos branch | detected | 1 site, line 67 |
| NEG | prose/string-only source | 0 | 0 |
| NEG | `d = c - 1` | 0 (direction matters) | 0 |
| defect demo | old grep on the prose-only source | fires | **True** ⇐ the defect, reproducible |

**⛔⛔ My instrument failure F2 — a keyword list standing in for semantics, and this one
reversed my conclusion.** With F1 fixed I scanned 292 scripts under `/tmp/latentuse` and
`/tmp/coordcheck` (`ast_scan_all.log`): 167 compute a cosine, **13 match the string, 69
actually perform `1 − x`** — so the string predicate was wrong in *both* directions. I then
classified the right-hand operand of each `1 − x` with a keyword list, and it labelled
`1 - An @ Bn.T` as **"not a cosine."** But `An`, `Bn` are L2-normalised and the result feeds
`argmin` — that *is* the cosine distance. Four real sites were hidden:

| file:line | code | ranked by |
|---|---|---|
| `/tmp/coordcheck/fp32cos_ulp_probe.py:27` | `d=(1-An@Bn.T)[o:]` | `d.min(1)` |
| `/tmp/coordcheck/tiebreak_and_reconcile.py:45` | `d = 1 - An @ Bn.T` | `d.argmin(1)` |
| `/tmp/coordcheck/l2_dtypefree_effect.py:29` | `d=((1-An@Bn.T)…)` | `d.min(1)` |
| `/tmp/coordcheck/tie_explains_nonmonotone.py:30` | `d=(1-Qn@Bn.T)` | `argmin` |

⇒ **`cos_mechanism.py`'s claim (3) — "the recorded integers came from code with no
subtraction in it" — is WITHDRAWN. The producing code does subtract.** Two unreliable
predicates, opposite error directions, one proposition; ⭐ **when a non-existence claim
survives only because of how I matched, re-ask it with a predicate that parses what it
claims to be about, and then classify the hits by what they feed, not by their spelling.**

**✅ What survives, and it is stronger than G2's version.** The subtraction is performed,
and it is **exact**. Sterbenz's lemma: for IEEE floats with `c ∈ [0.5, 2]`, `fl(1−c) = 1−c`
with *zero* rounding error. The contenders for the argmin are by construction the *highest*
cosines in the row, so they sit exactly where the lemma applies. Measured in the producing
pipeline (`cos_mechanism_v2.log`, denominator **8940** contender cosines, top-5 per row):
**min contender cosine 0.999951, max 1.000000; 0/8940 outside `[0.5,2]`; 0/8940 inexact.**
Sterbenz predicate control: `c=0.75` exact, `c=0.05` not exact ⇒ not a tautology.

Running the producing pipeline with the subtraction in fp32 vs in fp64:

| cell | subtract in cell dtype | subtract in fp64 | argmin **set** differs | first-index differs |
|---|---|---|---|---|
| fp32/cos | 16 hits, 105 tied | 16 hits, 105 tied | **0/1788** | **0/1788** |
| fp64/cos | 14 hits, 0 tied | 14 hits, 0 tied | **0/1788** | **0/1788** |

The recorded pair (16 / 14) and the recorded tie counts (105 / 0) both reproduce, so this is
the right pipeline — and the subtraction is a **provable no-op** on it, not merely an
empirical one on this sample. ⭐ **G2 showed the mechanism is absent on this data; Sterbenz
shows it cannot be present on any data in range.** So the phenomenon is real and the
mechanism recorded for it is not: ⛔ **"catastrophic cancellation in `1−cos`" must not be
repeated**, and the reason is that cancellation in `a−b` is catastrophic only when `a≈b`
*and* the subtraction itself rounds; here `a≈b` holds maximally (cosines at 0.99995–1.0) and
that is exactly the regime where the subtraction is *exact*. The near-cancellation destroys
*relative precision in the result* — the leading digits are gone before the subtraction —
but it introduces **no new error**, which is the part the record got backwards.

**✅ The replacement mechanism, measured.** Three-way decomposition of where the fp32 error
enters (`cos_mechanism_v2.log`, denominator 1788 rows per variant):

| variant | hits | tied rows |
|---|---|---|
| fp64 reference | 14 | 0 |
| **fp32 (the cell)** | **16** | **105** |
| storage / representation only (fp64 arithmetic, rounded to fp32) | 14 | 122 |
| normalise in fp32 only (matmul in fp64) | 14 | 0 |
| **matmul in fp32 only** (normalise in fp64) | **16** | **110** |

⇒ **the error enters in the 32-dim dot-product accumulation.** Storage and normalisation each
leave the count at the fp64 value of 14; only the matmul moves it to 16. Note the control
that matters: *rounding the fp64 matrix to fp32 creates 122 tied rows — more ties than the
cell has — and still gives 14 hits.* ⭐ **Ties alone do not produce the extra hits; the
ordering change in the accumulation does.** That kills the tempting shortcut of blaming
"fp32 ties" generically.

**⛔ And one overclaim of mine inside that table, caught by its own disagreeing column.**
I first wrote that matmul-in-fp32 "reproduces the cell exactly." The tie counts already
disagreed — **105 vs 110** — so I checked row-wise (`matmul_exactness.py`, `hitset_identity.py`):

| comparison | result |
|---|---|
| headline integer | 16 vs 16 — agree |
| **hit SET** (which rows are hits) | **0/1788 disagreements**, identical 16 rows `[179, 197, 465, 640, 670, 702, 775, 900, 1085, 1131, 1281, 1326, 1485, 1687, 1725, 1783]` (POS control: flipping one hit ⇒ 1 disagreement) |
| first-index argmax | **90/1788 differ** |
| argmax **set** | **172/1788 differ** |

So the correct statement is finer than either the overclaim or its retraction: **the
*statistic* is reproduced exactly — same value and same 16 rows — while the *matrix* is
not, differing on 172 rows.** ⭐ **"Reproduces exactly" needs a named object**: here it is
true of the hit set, false of the argmax sets, and the two facts live in the same table.
This is the memory file's "a tally cannot stand in for a delta" in its rarer form — the
tally *did* agree, and agreeing tallies still did not license the row-level claim; it took a
POS-controlled hit-set comparison to find which sense was true.

**Consequence for §15.29.17: none, and it is slightly strengthened.** The round-8 finding
was that the three disputed rows (1131, 1725, 1783, all race) are decided inside float32
rounding error at 0.33 / 1.41 / 0.29 ULP. Those three rows are in the hit list above, and
the decomposition now names the channel: the fp32 matmul accumulation, not a removable
formula. ⭐ An effect attributable to one removable subtraction could be argued away by
rewriting the metric; an effect in the dot-product accumulation cannot be, so **the
argument that the fp32/cos edge is not trustworthy gets stronger, not weaker.** The
defensible statement from §15.29.17 stands unchanged: p = **0.0730** in exact arithmetic,
not 0.0218.

**Recorded for the shared memory record (not yet written there — writing to shared memory
is not mine to do unilaterally):** the tie/order block's causal clause should read *"fp32
error in the dot-product accumulation (matmul), not cancellation in `1−cos`; the
subtraction is exact by Sterbenz since all contenders lie in `[0.5,2]` — measured 0/8940
inexact"*. The integers 16 / 15 / 15.53±0.50 and the 105 ties are **unaffected and still
reproduce**; only the stated cause changes.

### 15.29.19 Round 9 addendum — G2's broken first check, and the reflex it yields

Recording the one item from G2's round-8 message that is a *form*, not a number, because it
generalises past this thread. Its first attempt to check my structural bound
`FW p ≥ min per-cell observed p` printed **"13 violations, min slack −0.024000"** and it was
about to send a refutation. The cause was the matched-sides defect a third time: it computed
the observed minimum `om` from **4000** reps while subsampling the null to **1000**. The two
sides of the inequality were measured at different resolutions, so the comparison was not the
inequality at all. On equal footing the bound holds with **0 violations over 240 (family,
convention, seed) triples, min slack +0.00000**.

⭐⭐ **The reflex it yields: a violated *identity* (as opposed to a failed *estimate*) should
be treated as an impossible event, and the first hypothesis is that the two sides were not
measured in the same frame.** A structural bound derived from the definition of a maximum
cannot be 13-for-240 false; that arithmetic is not a finding about the data, it is a finding
about the instrument. This is the same shape as my own `‖p−q‖² = ‖p‖² + ‖q‖² − 2p·q`
reconciliation and the same shape as the `fw_fpr.py` per-cell draw — ⭐ **whenever a check
compares two quantities, print the sample size / dtype / seed of *both* sides next to the
verdict**, because every instance of this form in this thread was a frame mismatch that a
side-by-side print would have shown immediately. Added to the D4 discipline.

### 15.29.20 Round 9 — the orphan-appendix blocker, re-established by enumeration (and a POS control of mine that silently did not resolve)

**⛔ First, a defect in my own round-8 evidence.** I reported the `\input`-count check with the
POS control *"13 on the EED paper."* Re-running it this round, the path I recorded resolved to
nothing and the control printed **`n/a`** — because the file is at
`paper_ICLR27/EED/paper/main.tex`, not `paper_ICLR27/EED/main.tex`. The check therefore ran
last round with **no working positive control**, and it announced a `0` for the subject while
its own calibration silently failed. ⭐ **A POS control that cannot resolve its subject is not
a weaker control, it is no control**, and because `grep -c` on a missing file emits `0` to a
fallback it looked like an answer. Fixed, with three controls that resolve:

| POS control | `\input`/`\include` count |
|---|---|
| `paper_ICLR27/EED/paper/main.tex` | **13** |
| `paper_ICLR27/animal/paper/iclr2027/main.tex` | **5** |
| `paper_ICLR27/StateAdaptation/paper/visual_lens/main.tex` | **6** |
| **subject: `paper_ICLR27/lalaland/main.tex`** | **0** |

**✅ And the blocker itself, now established by enumeration rather than a sample.** My earlier
route was a sampled label check, and sampling produced a genuinely misleading intermediate
result: of the first 8 labels defined in `appendix.tex`, **2** (`app:background`, `eq:unroll`)
*do* appear in `main.aux`, which reads as partial reachability. It is not. Both are
**independently defined in `main.tex` as well** (1 occurrence each, in both files). Full
accounting:

| quantity | value |
|---|---|
| labels defined in `appendix.tex` | 93 |
| labels defined in `main.tex` | 69 |
| `\newlabel` entries in compiled `main.aux` | 69 |
| labels **unique** to `appendix.tex` | 68 |
| of those, present in `main.aux` | **0** |
| labels defined in **both** files (explains the apparent overlap) | 25 |
| POS control: labels unique to `main.tex` present in `main.aux` | **44/44** |

⇒ **Not one of the 68 labels that exist only in `appendix.tex` is compiled.** The apparent
overlap was 25 duplicated label *names*, not shared content. ⭐⭐ **This is the memory file's
"enumerate three axes for a non-existence claim" in a new costume: my sampled check had the
right predicate and the wrong denominator, and the 2/8 hit rate pointed the opposite way from
the truth (0/68).** Reachability was checked on three axes — `\input`/`\include` (0),
`appendix` named anywhere in `main.tex` (5 occurrences, all prose cross-references such as
`Appendix~\ref{app:training}`, none a file inclusion), and a `subfiles`/`standalone` route
(`\documentclass`/`\begin{document}` in `appendix.tex`: **0**) — plus the compiled artefact.

Specifically for the deliverable: `tab:latent-use-pending` is at `appendix.tex:719` (label) with
its framing at `:712`, present in `appendix.tex` (**True**), absent from `main.tex` (**False**),
absent from `main.aux` (**False**); `latent-use` and `latent use` each appear **0** times in
`main.tex`. `main.tex` has `\appendix` at `:467` and `\end{document}` at `:993`, i.e. it carries
its own 526-line inline appendix which is the one that compiles.

⛔ **Therefore nothing written into `DiffRwkv/paper_ICLR27/lalaland/appendix.tex` reaches
`main.pdf`.** Filling `tab:latent-use-pending` there would produce a table that does not exist
in the compiled paper. Also worth flagging to whoever owns the file: `appendix.tex:712` states
in its own prose that the table is *"a pending evaluation template, not a reported result"* and
is hidden by `\LaLaLanDPendingfalse` — so even on a reachable path it is not currently a
reporting slot. Both G2 and I reach this independently; neither of us has modified either file
(md5s `4bd162948b3300452c9714a6273a23bb` / `32f46730821226cc150d3ce0dfef1f9b`, unchanged all
round). **Which appendix is canonical is the user's call, and it remains the one open blocker.**
