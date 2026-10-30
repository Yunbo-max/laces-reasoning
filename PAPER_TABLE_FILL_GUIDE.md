# Filling Paper Tables — LaLaLanD ICLR27

## Ready-to-fill data

### Table 1 (Tab:capability): 6 new datasets (Fix2 results)
From `/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/benchmarks/fix2_scores_summary.json`:

| Dataset | champ-fixed-512 | champ-dynamic | global-dit-50k | raw-rwkv | Qwen2.5-3B | Llama-3.2 | Phi-2 |
|---------|----------|----------|----------|----------|----------|----------|----------|
| arc_easy | 13.0 | 15.5 | 15.2 | 15.8 | 13.2 | 18.8 | 19.0 |
| arc_challenge | 5.8 | 4.2 | 7.8 | 5.0 | 2.8 | 11.0 | 9.5 |
| winogrande | 26.5 | 26.0 | 25.2 | 34.0 | 19.0 | 18.8 | 22.8 |
| sciq | 20.5 | 25.0 | 25.8 | 28.0 | 20.0 | 24.8 | 26.0 |
| commonsenseqa | 7.0 | 8.5 | 6.8 | 6.2 | 6.5 | 12.5 | 11.8 |
| piqa | 1.2 | 2.0 | 1.0 | 0.5 | 1.2 | 9.5 | 3.2 |

**Status:** ✅ All scored, ready to insert into Table 1

---

### Appendix PPL Table (Tab:lm): Language modeling perplexity
**Pending:** traj_cfg PPL for champ-fixed-512 and champ-dynamic (waiting for job completion)

**Currently available:**
- champ-fixed-512 raw: 16.09 (wikitext2), 11.7 (pg19)
- champ-dynamic raw: 16.09 (wikitext2), 11.7 (pg19)
- champ-fixed-512 prefix_cfg: ❌ CORRUPTED (66k) — will be replaced by traj_cfg
- champ-dynamic prefix_cfg: ❌ CORRUPTED (66k) — will be replaced by traj_cfg
- global-dit-50k raw/prefix_cfg: ✅ Normal
- All baselines: ✅ Ready

---

## Actions

### Immediate (user or automated)
1. **If you have qz login:** Run `bash /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/submit_traj_cfg_ppl.sh`
   - This will submit 2 jobs for traj_cfg PPL (chunked models)
   - Expected turnaround: ~1-2 hours per job

2. **Once traj_cfg results land:** Read the new JSON files and manually swap them into the paper table (replace "66k" rows)

### Paper insertion points
- **Main text table:** See Table 1 caption in main.tex around line 250
- **Appendix PPL table:** See tab:lm caption in appendix around line ~~TBD~~

### Investigation: global-birwkv-50k → 0.5% mean
- This model is a global (single-z) variant, so should use single-z scoring
- 0.5% is suspiciously bad for any model
- Check: Does the model actually generate valid text? Or did the eval script silently truncate?
- Log: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/qz_eval_logs/fix2-global-birwkv-50k.log`
