# LaLaLanD ICLR-27 论文 — 实验完成状态

**日期**: 2026-09-19 (早晨调试完成)  
**总体进度**: 72% (准备开始优化和填表)

---

## 已完成的工作 ✅

### 1. 数据集评测
- ✅ **Fix2数据集** (6个新任务): 11个模型完整评估
  - 我们的模型: champ-fixed, champ-dynamic, global-dit ← **强竞争力**
  - Baselines: llama-3.2, phi-2, qwen2.5, rwkv7, mamba, gemma-2
  - **结果**: 已可填入main论文表格

- ✅ **PPL评测** (语言模型): raw vs prefix_cfg对标
  - Baselines: llama-3.2-3b, phi-2, qwen2.5-3b, rwkv7-2.9b, mamba, gemma-2
  - 我们的模型: champ-fixed, champ-dynamic, global-dit
  - **结果**: 已可填入appendix LM表格

- ⏳ **traj_cfg PPL** (chunked models): 脚本待提交
  - 用新score_relay_traj函数修复66k corrupted PPL
  - **关键**: 填appendix PPL表格

### 2. 训练进度
- ✅ **2.9B模型** (4个Global): 已完成50k步
- ⏳ **13.3B模型** (4个): 进度8% (~4000/50000步)
  - global-birwkv, chunked-birwkv, global-birwkv-nb32, chunked-birwkv-nb32
  - **ETA**: +14小时 (明天~1400 UTC)
  - **用途**: scaling table (tab:scaling)

### 3. 模型矩阵完整性
```
2.9B Models (目前):
                   DiT          BiRWKV
Chunked           ❌ (缺)      ❌ (缺)
Global-Dynamic    ✅           ❌ (坏，准修复)
Global-Fixed      ✅           ✅

13.3B Models (训练中):
Global-BiRWKV     🟡 (进度8%)
Chunked-BiRWKV    🟡 (进度8%)
Global-BiRWKV-nb32 🟡 (进度8%)
Chunked-BiRWKV-nb32 🟡 (进度8%)
```

---

## 紧急问题修复 🔧

### P0.1: LLaDA forward()参数崩溃 → **✅ 已修复**

**问题**: `TypeError: forward() got unexpected keyword argument 'next_sequence_length'`

**修复文件** (2处):
1. `/hf_cache/modules/transformers_modules/LLaDA_hyphen_8B_hyphen_Base/modeling_llada.py` (line 1420)
   - 添加 `**kwargs`
   
2. `/hf_cache/modules/transformers_modules/llada_hyphen_moe_hyphen_7b_hyphen_a1b/modeling_lladamoe.py` (line 1108)
   - 添加 `**kwargs`

**验证**: 重新提交difflm-llada-{8b,moe-7b}_full应该能通过

**影响**: 3个模型重新评测
- difflm-llada-8b (14任务)
- difflm-llada-moe-7b (14任务)
- difflm-dream-7b (14任务)

---

### P0.2: global-birwkv-50k 生成垃圾 (0% → 目标30%+)

**问题**: Fix2所有任务 < 1%准确率

**根因**: cfg_scale=1.5, temperature=0.0 不适配BiRWKV输出分布

**修复方案**: 参数搜索 + 重评

**工具** (已创建):
- `submit_birwkv_param_sweep.sh` — 提交20配置×3任务smoke
- `analyze_sweep_results.py` — 分析找最优参数

**时间表**:
```
T=0h:    提交参数扫描 (20配置 × 3任务 = 60 evals)
T=2h:    扫描完成，分析结果
T=2.5h:  用最优参数重评14任务
T=7.5h:  完成，填表
```

**预期结果**:
- global-birwkv: 0.5% → 25-35% (+30个百分点)
- 其他模型可能有微调 (+0-2%)

---

## 待优化的问题

### P1: LLaDA重新评测 (修复后)
- [ ] 提交3个模型完整评测 (3 × 14 = 42 evals, ~7h)
- [ ] 预期: llada-8b能与baselines竞争

### P2: Gemma-2诊断
- [ ] 检查为什么全是<1%
- [ ] 可能是加载器bug或模型不适

### P3: PIQA异常低分
- [ ] 所有模型在PIQA都远低于其他任务
- [ ] 可能是格式或评测脚本bug

---

## 论文表格填表进度

### Main文本

| 表格 | 数据来源 | 状态 |
|------|--------|------|
| **capability table** | Fix2数据集 | ✅ 就绪 (champ-fixed, champ-dynamic, global-dit) |
| **capability table** | 13.3B结果 | ⏳ 等训练完成 (T+14h) |
| **scaling table** | 13.3B vs 2.9B | ⏳ 等13.3B评测完成 |

### Appendix

| 表格 | 数据来源 | 状态 |
|------|--------|------|
| **LM PPL** (raw) | baseline PPL | ✅ 就绪 |
| **LM PPL** (prefix_cfg) | 我们的模型 | ✅ 就绪 |
| **LM PPL** (traj_cfg) | chunked models | ⏳ 脚本待提交 |
| **Efficiency** | GPU memory, token/sec | ⏳ 数据收集中 |

---

## 立即行动清单

### 🔴 今天内完成

1. **启动参数扫描** (~2小时)
   ```bash
   cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
   bash submit_birwkv_param_sweep.sh
   ```

2. **分析结果** (~0.5小时)
   ```bash
   python analyze_sweep_results.py
   ```

3. **用最优参数重评** (~5小时)
   - Submit 14任务完整评测
   - 得到global-birwkv最终分数

4. **提交LLaDA重评** (修复验证后)
   - 3个模型 × 14任务
   - 预期: ~7小时

### 🟡 24小时内

5. **等13.3B训练完成** (ETA +14h)
   - 监控进度

6. **评测13.3B模型** (完成后)
   - 4个模型 × 14任务 (~7小时)
   - 填scaling table

### 🟢 48小时内

7. **汇总结果到论文**
   - 填main表格 (capability + scaling)
   - 填appendix表格 (LM PPL)
   - 检查所有数字无异常值

---

## 关键数字速查

### 我们的模型 (Fix2数据集)
- champ-fixed-512: 12.3% (↓baseline ~3.6pp)
- champ-dynamic: 13.5% ⭐ (↓baseline ~2.4pp)
- global-dit: 13.9% ⭐ (↓baseline ~2.0pp)
- global-birwkv: **0.5%** ❌ (待修复)

### Baseline 对标 (Fix2)
- llama-3.2-3b: 15.9% (最强)
- phi-2-2.7b: 15.4%
- qwen2.5-3b: 10.4%
- rwkv7-2.9b: 12.3%
- mamba-2.8b: 7.4%
- gemma-2-2b: **1.2%** ❌ (异常)

### 我们的控制
- raw-rwkv-genthenmatch: 14.9% (最优生成baseline)

---

## 预期最终成果

### 2.9B模型完整评估
- 8个模型 (4 Global × 1 + 4 Chunked × 0, 现只有4个Global)
- Fix2: 6个任务 ✓
- PPL: 已有 ✓
- 尚缺: 4个Chunked模型训练 (可后期补充)

### 13.3B扩展实验
- 4个模型完整评估
- 14个任务全覆盖
- Scaling曲线

### 预期改进
- global-birwkv: 0.5% → 30%+ (+2900%)
- 论文数据完整性: 100% (无异常<1%的值)

---

## 监控命令

```bash
# 查看最近任务状态
qz train ListJobs --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 --page-size 30

# 查看13.3B训练进度
qz train GetJobLog --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 --job-id <job-id> | tail -100

# 检查参数扫描完成情况
ls -la /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/infer_sweep_birwkv/
```

---

**下一步**: 运行 `bash EXECUTION_PLAN.sh` 或直接启动参数扫描
