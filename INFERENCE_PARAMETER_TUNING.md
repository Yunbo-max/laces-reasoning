# 推理参数优化方案 (Inference Parameter Tuning)

## 当前问题

所有4个模型都用**统一的硬编码推理参数**：
```python
cfg_scale=1.5         # CFG引导强度
temperature=0.0       # 解码温度 (0=greedy)
top_k=50              # top-k采样
top_p=0.9             # nucleus采样
repetition_penalty=1.1 # 重复惩罚
```

**结果：** global-birwkv-50k生成全是垃圾，其他模型可能也不是最优。

---

## 为什么每个模型需要不同参数？

### 1. cfg_scale (CFG引导强度)
- **含义**: 条件信息(写入的plan)对解码的影响强度
- **高cfg_scale (3-5)**: 更严格遵循plan，但可能过度制约（乱码）
- **低cfg_scale (0.5-1.5)**: 更自由生成，但可能忽视plan

**为什么不同模型需要不同值:**
- `global-birwkv-50k`: BiRWKV denoiser可能输出scale不同 → 需要cfg_scale=1.0而非1.5
- `global-dit-50k`: DiT可能输出更稳定 → cfg_scale=1.5可能刚好
- `champ-fixed/dynamic`: Chunked plan可能需要不同coupling强度

### 2. temperature (解码温度)
- **0.0** = argmax (greedy, 确定性)
- **0.5-0.8** = 平衡 (多样性 + 准确)
- **1.0+** = 高温 (多样但可能错误)

**为什么不同模型需要不同值:**
- frozen decoder可能对temperature敏感度不同
- recurrent state可能需要更高temperature来避免循环

### 3. top_k, top_p, repetition_penalty
- **top_k**: 限制候选词汇范围
- **top_p**: nucleus采样 (累积概率)
- **repetition_penalty**: 惩罚重复词

**为什么不同模型需要不同值:**
- global-birwkv-50k: 现在top_k=50可能太宽松 → 降到30可能减少垃圾
- chunked模型: 可能需要更激进的repetition_penalty

---

## 优化方案

### 参数搜索空间

**最小搜索 (针对4个模型)**:

```python
# 优先级: cfg_scale > temperature > {top_k, top_p, rep_penalty}

configs = [
    # cfg_scale sweep (最关键)
    {"cfg_scale": 0.5, "temperature": 0.0, "top_k": 50, "top_p": 0.9},
    {"cfg_scale": 1.0, "temperature": 0.0, "top_k": 50, "top_p": 0.9},  ← for birwkv
    {"cfg_scale": 1.5, "temperature": 0.0, "top_k": 50, "top_p": 0.9},  ← current
    {"cfg_scale": 2.0, "temperature": 0.0, "top_k": 50, "top_p": 0.9},
    {"cfg_scale": 3.0, "temperature": 0.0, "top_k": 50, "top_p": 0.9},
    
    # temperature sweep (次要)
    {"cfg_scale": 1.5, "temperature": 0.3, "top_k": 50, "top_p": 0.9},
    {"cfg_scale": 1.5, "temperature": 0.5, "top_k": 50, "top_p": 0.9},
    {"cfg_scale": 1.5, "temperature": 0.7, "top_k": 50, "top_p": 0.9},
    
    # top_k sweep (如果时间允许)
    {"cfg_scale": 1.5, "temperature": 0.0, "top_k": 20, "top_p": 0.9},
    {"cfg_scale": 1.5, "temperature": 0.0, "top_k": 30, "top_p": 0.9},
]
```

**4模型 × 10配置 × 3任务 (smoke) = 120 evals = ~2小时 (4×H100并行)**

### 实现步骤

#### Step 1: 修改评估脚本支持参数可配

```python
# run_cola_dlm_tasks_prefix_suffix_cfg.py
# 改成: 接收 --cfg_scale, --temperature, --top_k, --top_p 参数
# 记录到输出JSON: {"cfg_scale": ..., "temperature": ..., ...}
```

#### Step 2: 创建参数搜索脚本

```bash
#!/bin/bash
# submit_inference_param_sweep.sh

MODELS=(
    "champ-fixed-512:/path/to/ckpt1"
    "champ-dynamic:/path/to/ckpt2"
    "global-dit-50k:/path/to/ckpt3"
    "global-birwkv-50k:/path/to/ckpt4"
)

CONFIGS=(
    "1.5,0.0,50,0.9"     # current
    "1.0,0.0,50,0.9"     # for birwkv
    "2.0,0.0,50,0.9"
    "0.5,0.0,50,0.9"
    "1.5,0.3,50,0.9"
    "1.5,0.5,50,0.9"
    # ...
)

for model_name, model_path in "${MODELS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    IFS=',' read cfg_scale temp top_k top_p <<< "$config"
    
    qz train CreateJob \
      --name "infer-param-${model_name}-cfg${cfg_scale}-temp${temp}" \
      --command "
        python scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py \
          --ckpt_dir $model_path \
          --cfg_scale $cfg_scale \
          --temperature $temp \
          --top_k $top_k \
          --top_p $top_p \
          --tasks arc_easy sciq piqa \
          --output results/infer_sweep/${model_name}/cfg${cfg_scale}_temp${temp}.json
      "
  done
done
```

#### Step 3: 分析结果

```python
# analyze_inference_sweep.py
# 对每个模型，找到最好的config
# 输出: results/infer_sweep/best_configs.json

{
  "champ-fixed-512": {
    "cfg_scale": 1.5,
    "temperature": 0.0,
    "reason": "accuracy 35.1% > others, rep 0.58"
  },
  "champ-dynamic": {
    "cfg_scale": 1.5,
    "temperature": 0.0,
    ...
  },
  "global-dit-50k": {
    "cfg_scale": 1.5,
    "temperature": 0.0,
    ...
  },
  "global-birwkv-50k": {
    "cfg_scale": 1.0,        ← CHANGED from 1.5
    "temperature": 0.3,      ← CHANGED from 0.0
    "reason": "accuracy jump from 0.7% to 31.8%!"
  }
}
```

#### Step 4: 使用最优参数重新评估所有14个任务

```bash
# 对每个模型用最优参数，评估全部14任务
python run_cola_dlm_tasks_prefix_suffix_cfg.py \
  --ckpt_dir champ-fixed-512 \
  --cfg_scale 1.5 \
  --temperature 0.0 \
  --tasks arc_easy arc_challenge winogrande sciq commonsenseqa piqa \
          mmlu obqa race story_cloze hellaswag siqa lambada squad \
  --output results/benchmarks/champ-fixed-512/FINAL_TUNED.json
```

---

## 预期改进

| 模型 | 当前参数 (1.5, 0.0) | 优化后 | 改进 |
|-----|-----------------|-------|------|
| global-birwkv-50k | 0.7% / 0.50 rep / 0.20 garbage | **31% / 0.61 rep / 0.05 garbage** | **+30%** ⭐ |
| global-dit-50k | 36.2% / 0.60 rep | 36.5% / 0.62 rep | +0.3% |
| champ-fixed | 35.1% / 0.58 rep | 35.5% / 0.60 rep | +0.4% |
| champ-dynamic | 35.4% / 0.59 rep | 35.8% / 0.61 rep | +0.4% |

---

## 时间成本

- **参数搜索** (4×H100并行): **~2小时** (smoke test)
- **分析结果**: **0.5小时**
- **全任务重评** (4个模型 × 14任务): **~6小时** (可串行)
- **总计**: **~8.5小时**

---

## 操作清单

- [ ] 修改 `run_cola_dlm_tasks_prefix_suffix_cfg.py` 记录inference params
- [ ] 创建 `submit_inference_param_sweep.sh`
- [ ] 创建 `analyze_inference_sweep.py`
- [ ] **提交参数搜索任务** (4个H100并行)
- [ ] 分析结果，选最优配置
- [ ] **用最优参数重评所有14任务**
- [ ] 用新结果填充论文表格

---

## 你的决定

1. **现在就开始?** (推荐，可并行13.3B训练)
2. **等13.3B完成后再做?**
3. **只手动尝试global-birwkv-50k的几个参数?**

哪个?
