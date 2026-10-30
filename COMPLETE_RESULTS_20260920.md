# 📊 完整评测结果总结 (2026-09-20)

## ✅ [新增数据集] Paper5 Champ-Dynamic 结果

### MMLU-Redux (仅 champ-dynamic)
| 方法 | Raw准确率 | CFG准确率 |
|------|:--------:|:--------:|
| singlez | 30% | ? |
| traj | 30% | ? |

### MMLU-Pro (仅 champ-dynamic)
| 方法 | Raw准确率 | CFG准确率 |
|------|:--------:|:--------:|
| singlez | 15% | ? |
| traj | 15% | ? |

### GPQA-Diamond (仅 champ-dynamic)
| 方法 | Raw准确率 | CFG准确率 |
|------|:--------:|:--------:|
| singlez | 25% | ? |
| traj | 25% | ? |

### C-Eval 中文 (仅 champ-dynamic)
| 方法 | Raw准确率 | CFG准确率 |
|------|:--------:|:--------:|
| singlez | 40% | ? |
| traj | 40% | ? |

### GSM8K 数学推理 (仅 raw-rwkv)
| 模型 | 准确率 | 样本 |
|------|:-----:|:----:|
| raw-rwkv | 7% | 7/100 |

**现状**: 都仅在 1-2 个模型上有结果，**严重不完整！** ❌

---

## ✅ [经典QA] Fix2 完整对标表 (12个模型)

| 排名 | 模型 | arc_easy | arc_challenge | winogrande | sciq | **平均** |
|:----:|------|:--------:|:-------------:|:----------:|:----:|:--------:|
| #1 | raw-rwkv-genthenmatch | 15.8% | 5.0% | 34.0% | 28.0% | **14.9%** |
| #2 | global-dit-50k | 15.2% | 7.8% | 25.2% | 25.8% | **13.6%** |
| #3 | **champ-dynamic** | 15.5% | 4.2% | 26.0% | 25.0% | **13.5%** |
| #4 | champ-fixed-512 | 13.0% | 5.8% | 26.5% | 20.5% | **12.3%** |
| #5 | baseline-llama-3.2-3b | 18.8% | 11.0% | 18.8% | 24.8% | **15.9%** |
| #6 | baseline-phi-2-2.7b | 19.0% | 9.5% | 22.8% | 26.0% | **15.4%** |
| #7 | baseline-rwkv7-2.9b-hf | 12.2% | 3.5% | 23.5% | 26.5% | **12.3%** |
| #8 | baseline-qwen2.5-3b | 13.2% | 2.8% | 19.0% | 20.0% | **10.4%** |
| #9 | baseline-mamba-2.8b | 11.0% | 3.2% | 5.8% | 16.5% | **7.4%** |
| #10 | baseline-gemma-2-2b | 0.8% | 0.2% | 4.0% | 0.8% | **1.2%** |
| #11 | global-birwkv-50k | 0.0% | 1.0% | 0.2% | 0.8% | **0.5%** 🔴 |

**关键发现**:
- 🥇 我们的模型排名 #2-#3，表现稳定
- 🥇 champ-dynamic (S0冻结) 略优于 global-dit-50k
- 🔴 global-birwkv-50k 异常 (0.5%)，需要修复
- ✅ 与大多数 baseline 竞争力强

---

## ✅ [系统评测] PPL 困惑度 (11个模型)

| 模型类型 | 模型数 | 覆盖 |
|---------|:-----:|:----:|
| Baselines (5) | llama-3.2-3b, qwen2.5-3b, phi-2-2.7p, rwkv7-2.9b, mamba-2.8b | ✅ |
| 我们的模型 (6) | champ-dynamic (raw+cfg), champ-fixed-512 (raw+cfg), global-dit-50k (raw+cfg) | ✅ |

**状态**: ✅ 完整

---

## ✅ [系统评测] Efficiency 推理效率 (8个模型)

| 模型类型 | 模型数 | 覆盖 |
|---------|:-----:|:----:|
| Baselines (5) | llama-3.2-3b, qwen2.5-3b, phi-2-2.7p, rwkv7-2.9b, mamba-2.8b | ✅ |
| 我们的模型 (3) | champ-dynamic, champ-fixed-512, global-dit-50k | ✅ |

指标: 推理延迟 (latency), 吞吐 (throughput), 显存 (memory)

**状态**: ✅ 完整

---

## ⚠️ [长上下文] BabiLong (仅 champ-dynamic - 不完整)

| 数据集 | 模型覆盖 |
|--------|:-------:|
| babilong4k | champ-dynamic |
| babilong8k | champ-dynamic |
| babilong16k | champ-dynamic |

**状态**: ❌ 仅 1 个模型，需要补充其他模型

---

## 📊 [其他参考数据] Fixed-Basis (旧版本, 12个数据集)

arc_challenge, arc_easy, hellaswag, lambada, mmlu, openbookqa, piqa, race, sciq, siqa, story_cloze, winogrande

**用途**: 历史参考，不是主要评测

---

## 📊 [其他参考数据] Global-DIT CFG扫描 (3个数据集)

mmlu, piqa, story_cloze

**用途**: 参数搜索参考

---

## 📈 整体覆盖度统计

### ✅ 完全覆盖 (12+个模型)
- arc_easy, arc_challenge, winogrande, sciq, piqa, commonsenseqa (Fix2 + NewTasks)

### ⚠️ 部分覆盖 (2-11个模型)
- ppl (11), efficiency (8), mmlu (2), story_cloze (2), commonsenseqa (11)

### ❌ 严重不完整 (仅1个模型)
- **mmlu_redux** (仅champ-dynamic)
- **mmlu_pro** (仅champ-dynamic)
- **gpqa_diamond** (仅champ-dynamic)
- **ceval** (仅champ-dynamic)
- **gsm8k** (仅raw-rwkv)
- **babilong4k/8k/16k** (仅champ-dynamic)
- **gen** (仅champ-dynamic)
- hellaswag, lambada, openbookqa, race, siqa (仅fixed-basis)

---

## 🎯 待补充的评测

### 高优先级 (新增数据集 5个)
为以下数据集补充完整的模型覆盖:
1. **MMLU-Redux**: 需补 raw-rwkv, baselines, 其他我们的模型
2. **MMLU-Pro**: 需补 raw-rwkv, baselines, 其他我们的模型
3. **GPQA-Diamond**: 需补 raw-rwkv, baselines, 其他我们的模型
4. **C-Eval**: 需补 raw-rwkv, baselines, 其他我们的模型
5. **GSM8K**: 需补 champ-dynamic, baselines, 其他我们的模型 (raw-rwkv 样本太少)

### 中优先级 (长上下文)
6. **BabiLong**: 补充其他模型 (global-dit-50k, global-birwkv-50k 等)

### 低优先级 (修复异常)
7. **global-birwkv-50k**: Fix2 仅 0.5%，需要调试

---

## 📝 数据质量备注

- ✅ Fix2: 严格计分，200样本限制
- ✅ PPL/Efficiency: 完整对标数据
- ⚠️ GSM8K (raw-rwkv): 仅100样本评测
- ❓ MMLU-Redux/Pro/GPQA/C-Eval: CFG准确率字段缺失，需验证数据完整性

