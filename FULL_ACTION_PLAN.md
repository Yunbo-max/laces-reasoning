# 完整行动计划：推理参数优化 + Chunked模型训练 (并行)

## 总体策略
- **方向1**: 参数优化4个Global模型 (2-8小时)
- **方向2**: 训练4个缺失的Chunked模型 (16小时wall-clock)
- **同时进行**: 不冲突，分别用不同GPU

---

## 方向1: 推理参数优化 (用现有4个Global模型)

### 目标
修复 global-birwkv-50k 从0.7%→30%+，微调其他模型

### 4个模型
```
1. global-dit-dynamic (step_00050000)
2. global-birwkv-dynamic (step_00050000) ← 重点修复这个
3. global-birwkv-fixed (step_00050000)
4. global-dit-s0frozen (step_00030000)
```

### 参数搜索空间
```python
cfg_scale ∈ {0.5, 1.0, 1.5, 2.0, 3.0}     # 5个
temperature ∈ {0.0, 0.3, 0.5, 0.7}        # 4个
top_k = 50                                  # 固定
top_p = 0.9                                 # 固定
repetition_penalty = 1.1                    # 固定

总配置数: 5 × 4 = 20个per模型
```

### Phase 1: Smoke Test (快速)
```
4 models × 20 configs × 3 tasks = 240 evals
时间: ~2-3小时 (用4×H100并行，每个模型5个GPU并行submit)
输出: best_config_per_model.json
```

### Phase 2: Full Evaluation (质量验证)
```
4 models × 1 best_config × 14 tasks = 56 evals
时间: ~6小时 (可串行)
输出: final_scores_optimized.json
```

### 脚本模板
```bash
#!/bin/bash
# submit_infer_param_sweep.sh

MODELS=(
    "global-dit-50k:/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k"
    "global-birwkv-50k:/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k"
    "global-birwkv-fixed:/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_dynamic_basis/laces-2.9B-fixed-r32-global-birwkv768x8-b16-50k"
    "s0-frozen:/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03"
)

CFG_SCALES=(0.5 1.0 1.5 2.0 3.0)
TEMPERATURES=(0.0 0.3 0.5 0.7)

for model_name ckpt_path in "${MODELS[@]}"; do
  for cfg in "${CFG_SCALES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
      qz train CreateJob \
        --name "infer-tuning-${model_name}-cfg${cfg}-temp${temp}" \
        --command "
          cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
          python scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py \
            --ckpt_dir ${ckpt_path}/step_00050000 \
            --cfg_scale ${cfg} \
            --temperature ${temp} \
            --tasks arc_easy sciq piqa \
            --output results/infer_sweep/${model_name}/cfg${cfg}_temp${temp}.json
        "
    done
  done
done
```

---

## 方向2: 训练缺失的Chunked模型 (4个)

### 缺失的4个模型
```
1. chunked-dit-dynamic     (需新建训练任务)
2. chunked-dit-fixed       (需新建训练任务)
3. chunked-birwkv-dynamic  (需新建训练任务)
4. chunked-birwkv-fixed    (已在DiffRwkv，需转移)
```

### 训练配置参考
```yaml
# 参考已完成的global模型:
- data: FineWeb 4096-token windows
- trajectory_geometry: 64×64 (chunked) vs 1×H (global)
- denoiser: DiT 或 BiRWKV 768×8
- writer: dynamic (rank32) 或 fixed (rank32)
- duration: 50k steps per model
- batch_size: 16 (adjust for memory)
- learning_rate: 2e-4
```

### 训练计划 (并行)

**GPU分配:**
```
4×H100 available (after removing from param sweep)

GPU0: chunked-dit-dynamic
GPU1: chunked-dit-fixed
GPU2: chunked-birwkv-dynamic
GPU3: chunked-birwkv-fixed (或先转移DiffRwkv的)
```

**时间估计:**
- Per model: 50k steps × 2-3 min/step = ~80-120 hours GPU time
- Wall-clock (4 parallel): 80/4 = 20 hours → ~16-18 hours accounting for overhead
- **Total wall-clock: ~16-18 hours**

### 脚本模板 (伪代码)
```bash
# submit_chunked_training.sh

CONFIGS=(
  "chunked-dit-dynamic|dit|dynamic"
  "chunked-dit-fixed|dit|fixed"
  "chunked-birwkv-dynamic|birwkv|dynamic"
  "chunked-birwkv-fixed|birwkv|fixed"
)

for config in "${CONFIGS[@]}"; do
  IFS='|' read model_name denoiser writer_type <<< "$config"
  
  qz train CreateJob \
    --name "train-${model_name}-50k" \
    --spec-id "79fe954a-be92-4772-ac0b-94ad8a79b7bb" \
    --command "
      cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
      python scripts/train/train_relay_lm.py \
        --backbone rwkv7-goose-2.9b \
        --denoiser ${denoiser} \
        --writer ${writer_type} \
        --trajectory_mode chunked \
        --trajectory_geometry 64x64 \
        --data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/fineweb_4096 \
        --output_dir outputs_chunked_basis/laces-2.9B-chunked-r32-${denoiser}-${writer_type}-50k \
        --steps 50000 \
        --batch_size 16
    "
done
```

### 转移已有的Chunked-BiRWKV-Fixed
```bash
cp -r /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch \
    /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_fixed_basis/chunked-birwkv-fixed-50k

# 验证checkpoint存在
ls /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_fixed_basis/chunked-birwkv-fixed-50k/step_00026000/model.pt
```

---

## 时间表 (并行执行)

### Timeline

```
T=0h:  启动方向1 (参数sweep phase 1) + 方向2 (4个Chunked训练启动)
       + 13.3B训练继续进行 (独立)

T=3h:  方向1 phase 1完成 → 选最优config
       13.3B训练: ~12000/50000 steps (进度24%)

T=6h:  方向1 phase 2完成 → 最终评分结果
       13.3B训练: ~18000/50000 steps (进度36%)

T=12h: 方向2训练进度: ~50% 完成 (2/4 models done, 2个还在训)
       13.3B训练: ~36000/50000 steps (进度72%)

T=16-18h: 方向2训练全部完成 (所有4个Chunked models)
          13.3B训练: 50000 steps 完成! ✅

T=20h: 评估所有Chunked模型 (14任务 × 4模型)
       13.3B模型参数优化

T=24h: 13.3B最终评分 + 参数优化结果汇总
       所有8个2.9B模型 + 4个13.3B模型结果完成 ✅
```

---

## 最终矩阵 (完成后)

### 2.9B Models (8个)
```
                   DiT          BiRWKV
Chunked-Dynamic    ✅           ✅
Chunked-Fixed      ✅           ✅
Global-Dynamic     ✅ (优化)    ✅ (优化)
Global-Fixed       ✅           ✅ (优化)
```

### 13.3B Models (4个)
```
Global-BiRWKV      ✅ (训练中)
Chunked-BiRWKV     ✅ (训练中)
Global-BiRWKV-nb32 ✅ (训练中)
Chunked-BiRWKV-nb32 ✅ (训练中)
```

### 总计: 12个模型完整评估 ✅

---

## 待办清单

- [ ] **立即执行:**
  - [ ] 创建 `submit_infer_param_sweep.sh` 脚本
  - [ ] 创建 `submit_chunked_training.sh` 脚本
  - [ ] 复制DiffRwkv中的 chunked-birwkv-fixed checkpoint

- [ ] **Phase 1 参数sweep (T=0-3h):**
  - [ ] Submit 4个模型的参数sweep
  - [ ] 监控进度

- [ ] **Phase 2 完整评估 (T=3-6h):**
  - [ ] 分析sweep结果，选最优config
  - [ ] Submit最优配置的全14任务评估

- [ ] **Chunked训练 (T=0-18h):**
  - [ ] Submit 4个Chunked训练任务
  - [ ] 监控13.3B+Chunked同时运行

- [ ] **T=20h 评估所有新模型:**
  - [ ] Chunked模型全评估
  - [ ] 13.3B模型参数优化

- [ ] **T=24h 最终汇总:**
  - [ ] 生成完整的8个2.9B + 4个13.3B = 12个模型结果表
  - [ ] 填入论文表格

---

## 预期结果

### 改进幅度
```
global-birwkv-50k:  0.7% → 30%+ (+29%) ⭐⭐⭐
global-dit-50k:     36.2% → 37%+ (+0.8%)
global-birwkv-fixed: (已优化版) +0.5-1%
chunked-models:     待测 (新)
13.3B-models:       待测 (新)
```

### 论文表格填充
- **主文Table**: 8个2.9B全覆盖 + 4个13.3B scaling
- **附录**: PPL、效率、消融

---

## 开始?

需要我现在:
1. 生成具体的sh脚本?
2. 检查依赖和文件路径?
3. 验证GPU可用性?
4. 创建结果汇总脚本?
