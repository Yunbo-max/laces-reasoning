# SII Training Status Report — 2026-09-18 ~14:41 UTC

## Active Training Tasks (4/4 running) ✅

### 13.3B Scaling Experiments

| Model | Steps | Loss | Progress | Est. Time to Finish |
|-------|-------|------|----------|-------------------|
| global-birwkv | 4000 | 1.8368 | 8.0% | ~838 min (14h) |
| chunked-birwkv | 4090 | 1.8676 | 8.2% | ~825 min (13.7h) |
| global-birwkv-nb32 | 4020 | 1.8116 | 8.0% | ~836 min (13.9h) |
| chunked-birwkv-nb32 | 3950 | 1.7871 | 7.9% | ~847 min (14.1h) |

**All 4 tasks alive and converging**, currently training ~0.08 step/s, ~2.5 samples/s per task

**GPU Memory:** 67–71 GB (stable across tasks)

---

## Completed/Ready Tasks (from earlier)

### 2.9B variants (completed, evaluated)
- ✅ champ-fixed-512 (fixed basis, trajectory)
- ✅ champ-dynamic (dynamic basis, trajectory)  
- ✅ global-dit-50k (single-z DiT)
- ✅ global-birwkv-50k (single-z BiRWKV) — Note: produced 0.5% mean on fix2 (needs investigation)

### Evaluation results (all landed)
- ✅ Fix2 (6 new datasets): all 11 models scored
- ✅ Difflm v6: llada-8b, dream-7b, llada-moe-7b loaded successfully
- ⏳ traj_cfg PPL: pending (chunked model evaluation, will fix corrupted 66k PPL)

---

## Next Milestones

1. **13.3B training** → ~14 hours to 50k steps
   - Global and chunked variants will be ready for evaluation soon after
   - Can then swap results into paper scaling table (tab:scaling)

2. **traj_cfg PPL** (for 2.9B chunked models)
   - Script ready: `/laces-reasoning/submit_traj_cfg_ppl.sh`
   - Blocks paper appendix PPL table fill

3. **Paper filling** (after traj_cfg PPL lands)
   - fix2 data → main text capability table
   - traj_cfg PPL → appendix LM table

---

## Current Monitoring

- All 4×13.3B tasks are healthy (no errors in logs)
- Losses stable and reasonable (~1.78–1.87)
- Memory usage stable (~67–71GB per task)
- **Estimated convergence time:** ~13.7–14.1 hours from now (0:00–01:00 UTC tomorrow)

**Do you want me to:**
1. Keep monitoring and auto-fill results when they land?
2. Submit traj_cfg PPL jobs now (if you can `qz login`)?
3. Start investigation on global-birwkv-50k's 0.5% anomaly?
