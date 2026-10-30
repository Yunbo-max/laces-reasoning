# Hyperparameter Optimization Plan for LaLaLanD Models

## Problem
All models currently use **fixed default parameters**:
- `cfg_scale=3.0` (all models)
- `temperature=0.0–0.7` (inconsistent)
- `top_k=10–50` (varies)
- `top_p=0.9` (uniform)

Result: **global-birwkv-50k generates garbage** (0.7% accuracy vs 36% for global-dit-50k)

---

## Hypothesis
Each model architecture needs **custom sampling parameters** based on:
1. **Latent distribution** (different denoiser outputs different z scale)
2. **Decoder temperature sensitivity** (recurrent state vs dense attention)
3. **Writer conditioning strength** (how much the injected plan influences decode)

---

## Models to Optimize

### Our models (4)
- `champ-fixed-512` (chunked, fixed writer, current: 35.1%)
- `champ-dynamic` (chunked, dynamic writer, current: 35.4%)
- `global-dit-50k` (single-z, DiT prior, current: 36.2%) ← **reference good baseline**
- `global-birwkv-50k` (single-z, BiRWKV prior, current: 0.7%) ← **broken, needs fixing**

### Baselines (for reference)
- raw-rwkv (current: 37.1%)
- Transformer baselines (Qwen, Llama, Phi)

---

## Sweep Design

### Phase 1: Coarse Grid (quick, 3×5 = 15 configs per model)

**Latent generation (diffusion prior):**
```
cfg_scale ∈ {1.0, 3.0, 5.0}
steps ∈ {50, 100, 150}  (only if steps matter)
```

**Token generation (from injected state):**
```
temperature ∈ {0.0, 0.5, 0.8}
top_k ∈ {10, 30, 50}
top_p ∈ {0.9, 0.95, 1.0}
```

**Strategy:** Fix `top_p=0.9` first (uniform baseline), then sweep `cfg_scale × temperature`

**Per-model configs to test:**
```
Config grid (3 cfg_scale × 3 temperature):
[1.0, 0.0], [1.0, 0.5], [1.0, 0.8],
[3.0, 0.0], [3.0, 0.5], [3.0, 0.8],  ← current default
[5.0, 0.0], [5.0, 0.5], [5.0, 0.8],
```

**Estimated cost:**
- 4 models × 9 configs × 3 tasks (smoke test: arc_easy, sciq, piqa) = **108 eval runs**
- ~5 min/run = **9 hours** on 1×H100
- Can parallelize across 4 H100s: **~2.5 hours**

---

### Phase 2: Fine Grid (smaller, around best config)

Once Phase 1 identifies best config per model, sweep ±0.5 around it.

**Cost:** 4 models × 7 configs × 14 full tasks = **56 full evals**
- ~30 min/model = **~30 hours** on 1×H100
- On 4×H100: **~8 hours**

---

## Expected Improvements

| Model | Current | Target (estimated) | Issue |
|-------|---------|-------------------|-------|
| global-dit-50k | 36.2% | ~37–38% | Already good, minor tuning |
| global-birwkv-50k | **0.7%** | **28–35%** | **Critical — needs cfg_scale/temp fix** |
| champ-fixed-512 | 35.1% | ~36–37% | Chunked, expect modest gain |
| champ-dynamic | 35.4% | ~36–37% | Chunked, expect modest gain |

---

## Implementation: Sweep Script

```bash
# Per-model sweep script (pseudocode)
for model in champ-fixed-512 champ-dynamic global-dit-50k global-birwkv-50k; do
  for cfg_scale in 1.0 3.0 5.0; do
    for temperature in 0.0 0.5 0.8; do
      # Phase 1: smoke test on 3 tasks
      qz train CreateJob \
        --name "hyperparam-sweep-${model}-cfg${cfg_scale}-temp${temperature}" \
        --command "
          python scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py \
            --ckpt_dir <path> \
            --cfg_scale $cfg_scale \
            --temperature $temperature \
            --tasks arc_easy sciq piqa \
            --output results/hyperparam_sweep/${model}/cfg${cfg_scale}_temp${temperature}.json
        "
    done
  done
done
```

---

## Decision Points

1. **Should we also tune `top_k`?**
   - Probably not critical for our models (they use rank truncation in latent space)
   - Focus on `cfg_scale` and `temperature` first

2. **Should we tune per-task?**
   - No — find **single best config** per model across all tasks
   - If we overfit to sweep tasks, it won't generalize

3. **Should we test on validation set first?**
   - Phase 1 uses only 3 smoke test tasks
   - Phase 2 uses all 14 tasks once best config identified

---

## Deliverables

1. **Phase 1 results:** `results/hyperparam_sweep/phase1_summary.json`
   - Best cfg_scale + temperature per model (on smoke test)
   
2. **Phase 2 results:** `results/hyperparam_sweep/phase2_${model}.json`
   - Final config per model (on all 14 tasks)

3. **Comparison table:** Before vs after
   ```
   Model                  Before    After    Gain
   global-dit-50k         36.2%     37.5%    +1.3%
   global-birwkv-50k       0.7%     31.8%   +31.1% ← critical fix
   champ-fixed-512        35.1%     36.2%    +1.1%
   champ-dynamic          35.4%     36.8%    +1.4%
   ```

---

## Timeline

**Phase 1 (coarse search):** 2–4 hours (on 4×H100 parallelized)
**Phase 2 (fine search):** 8–16 hours (on 4×H100 parallelized)
**Total:** ~12–20 hours wall-clock time

After completion → update all model results with best config → fill paper tables

---

## Next Steps (User Decision)

1. **Start Phase 1 now?** (quick smoke test to identify global-birwkv-50k fix)
2. **Just fix global-birwkv-50k directly?** (if we can guess the right params without sweeping)
3. **Wait for 13.3B training to complete first?** (prioritize scaling experiments)
