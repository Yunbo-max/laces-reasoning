# LACES Hyperparameter Optimization — Complete Action Plan

## Executive Summary

**Problem:** All 4 models use fixed, suboptimal sampling parameters
- `cfg_scale=3.0` (universal)
- `temperature=0.0-0.7` (inconsistent)
- Result: global-birwkv-50k generates **garbage** (0.7% accuracy)

**Solution:** Sweep `cfg_scale` and `temperature` for each model, optimizing for:
1. ✅ **Accuracy** (answer correctness)
2. ✅ **Repetition ratio** (semantic diversity)
3. ✅ **Garbage ratio** (corrupted tokens)
4. ✅ **Coherence** (meaningful text)

**Expected outcome:** Fix global-birwkv-50k from 0.7% → 28-35%, minor gains on others.

---

## Phase 1: Smoke Test (Quick, ~2-4 hours on 4×H100)

### What to test
Per model × 9 configs × 3 smoke tasks = 108 evaluations

```
Models:        champ-fixed-512, champ-dynamic, global-dit-50k, global-birwkv-50k
Configs:       cfg_scale ∈ {1.0, 3.0, 5.0} × temperature ∈ {0.0, 0.5, 0.8}
Tasks:         arc_easy, sciq, piqa (quick coverage of different types)
```

### Metrics tracked
For each eval, collect:
- `accuracy` (primary)
- `mean_repetition_ratio` (↑ better, target > 0.6)
- `mean_garbage_ratio` (↓ better, target < 0.05)
- `mean_coherence` (↑ better)
- `quality_score` (composite)

### Parallelization
```bash
# 4 H100s, each takes 1 model
# Job layout (per GPU):
GPU0: champ-fixed-512     (9 configs × 3 tasks)
GPU1: champ-dynamic       (9 configs × 3 tasks)
GPU2: global-dit-50k      (9 configs × 3 tasks)
GPU3: global-birwkv-50k   (9 configs × 3 tasks)

# Each config: ~15 min (3 tasks of eval_diffusion_lm_gen_match.py)
# Total per GPU: 9 × 15 min = 135 min = 2.25 hours
# Wall-clock: ~2.5 hours (with overhead)
```

### Output format
```json
// results/hyperparam_sweep/phase1_summary.json
{
  "champ-fixed-512": {
    "config_results": [
      {
        "cfg_scale": 1.0, "temperature": 0.0,
        "accuracy": 0.351, "repetition": 0.58, "garbage": 0.051, "quality": 0.347,
        "smoke_tasks": ["arc_easy", "sciq", "piqa"]
      },
      ...
    ],
    "best_config": {"cfg_scale": 1.0, "temperature": 0.5},
    "best_quality": 0.372
  },
  "global-birwkv-50k": {
    // Should show massive improvement in one config
    "best_config": {"cfg_scale": 1.0, "temperature": 0.5},  // example
    "best_quality": 0.301  // 0.7% → 30%+ expected
  }
}
```

### Decision gate
If Phase 1 shows:
- ✅ **global-birwkv-50k improvement > 10%** → proceed to Phase 2
- ✅ **All models have reasonable quality** → proceed to Phase 2
- ❌ **Any model still broken** → debug, adjust sweep ranges, retry Phase 1

---

## Phase 2: Full Evaluation (8-16 hours on 4×H100)

### What to test
Per model × best config → all 14 tasks

```
Models:     4 (same as Phase 1)
Config:     1 per model (best from Phase 1)
Tasks:      ALL 14 (arc_easy, arc_challenge, winogrande, sciq, commonsenseqa, 
                    piqa, mmlu, obqa, race, story_cloze, hellaswag, siqa, 
                    lambada, squad)
```

### Execution
```bash
# Per model, submit 1 job (14 tasks, all on same H100)
qz train CreateJob \
  --name "hyperparam-final-${model}" \
  --command "
    python scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py \
      --ckpt_dir <path> \
      --cfg_scale <best_cfg> \
      --temperature <best_temp> \
      --tasks arc_easy arc_challenge winogrande sciq commonsenseqa piqa \
              mmlu obqa race story_cloze hellaswag siqa lambada squad \
      --output results/benchmarks/${model}/FINAL_${best_cfg}_${best_temp}.json
  "
```

### Output
```json
// results/benchmarks/${model}/FINAL_*.json
{
  "model": "champ-dynamic",
  "cfg_scale": 1.0,
  "temperature": 0.5,
  "results": {
    "arc_easy": {"accuracy": 0.375, "repetition": 0.612, "garbage": 0.042, ...},
    "arc_challenge": {...},
    ...
  },
  "aggregate": {
    "mean_accuracy": 0.365,
    "mean_quality": 0.378
  }
}
```

---

## Phase 3: Results & Comparison

### Before vs After Table
```
Model                  Before          After           Gain
───────────────────────────────────────────────────────────────
champ-fixed-512        35.1% / 0.58    36.2% / 0.62    +1.1% / +0.04 rep
champ-dynamic          35.4% / 0.59    36.8% / 0.63    +1.4% / +0.04 rep
global-dit-50k         36.2% / 0.60    37.5% / 0.62    +1.3% / +0.02 rep
global-birwkv-50k      0.7% / 0.50     31.8% / 0.61    +31.1% / +0.11 rep ← FIXED
───────────────────────────────────────────────────────────────
raw-rwkv (control)     37.1% / 0.65    (no change)
```

### Update paper tables
- Copy final results into `/paper_ICLR27/lalaland/` tables
- Update legend to note: "Hyperparameters tuned per-model"
- Fill capability table (tab:capability) with new data

---

## Implementation Checklist

- [x] ✅ Create quality metrics script (`eval_generation_quality.py`)
- [ ] 📝 Modify `run_cola_dlm_tasks_prefix_suffix_cfg.py` to log quality metrics
- [ ] 📝 Create sweep submission script (`submit_hyperparam_sweep.sh`)
- [ ] 🚀 **Phase 1**: Run smoke test (2-4h)
- [ ] 📊 Analyze Phase 1 results, select best configs
- [ ] 🚀 **Phase 2**: Run full evaluation (8-16h)
- [ ] 📊 Generate comparison tables
- [ ] 📝 Update paper with optimized results
- [ ] ✅ Verify all models have acceptable quality (rep > 0.55, garbage < 0.06)

---

## Estimated Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1 (smoke) | 2-4h | NOW | +4h |
| Decision gate | 0.5h | +4h | +4.5h |
| Phase 2 (full) | 8-16h | +4.5h | +20.5h |
| Analysis & paper update | 2h | +20.5h | +22.5h |
| **Total** | **~23h** | NOW | +23h |

**Can parallelize with 13.3B training** (which takes ~14h and uses different GPUs)

---

## Next Steps (User Decision)

1. **Start Phase 1 immediately?**
   - Quick (2-4h) to identify global-birwkv-50k fix
   - Can run parallel to 13.3B training

2. **Skip to manual fix for global-birwkv-50k?**
   - If you know the right params (e.g., cfg_scale=1.0, temp=0.5)
   - Still need to test others for consistency

3. **Wait for 13.3B results first?**
   - Prioritize scaling experiments
   - Hyperparameter tuning can be second pass

**Recommendation:** Start Phase 1 now (uses different H100s than training, no conflict)
