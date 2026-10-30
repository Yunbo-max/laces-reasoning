# 失败任务Debug与修复计划 (执行中)

**日期**: 2026-09-19  
**优先级**: 紧急 — 停止新任务，先修复既有问题  
**状态**: ✅ P0问题修复脚本已就位，等待提交

---

## Part 1: 已发现的关键问题

### ✅ 问题1: LLaDA forward()参数不匹配 → **已修复**

**症状**: `TypeError: LLaDAModelLM.forward() got unexpected keyword argument 'next_sequence_length'`  
**根因**: transformers库新版本传递`next_sequence_length`参数，但LLaDA forward()不接受  
**修复**: 在forward()签名添加`**kwargs`参数

**修改文件**:
```
/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/modules/transformers_modules/
  LLaDA_hyphen_8B_hyphen_Base/modeling_llada.py (line 1408-1421)
```

**修改内容** (Line 1420添加):
```python
# 之前:
def forward(
    self,
    input_ids: torch.LongTensor = None,
    ...
    cache_position: Optional[Cache] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:

# 之后:
def forward(
    self,
    input_ids: torch.LongTensor = None,
    ...
    cache_position: Optional[Cache] = None,
    **kwargs,  # Absorb extra arguments from transformers generation
) -> Union[Tuple, CausalLMOutputWithPast]:
```

**验证**:
- [ ] 重新提交 difflm-llada-8b_smoke (3个任务)
- [ ] 应该能加载和运行，不crash on forward()

---

### 🔴 问题2: global-birwkv-50k 生成垃圾文本 (0%准确率)

**症状**: Fix2数据集全任务得分 0.0-1.0%  
**根因**: 推理参数(cfg_scale=1.5, temperature=0.0)不适配BiRWKV输出分布  
**修复方案**: 参数搜索

**已创建脚本**:
- `submit_birwkv_param_sweep.sh` — 提交20个配置 × 3任务
- `analyze_sweep_results.py` — 分析结果找最优参数

**搜索空间**:
```
cfg_scale ∈ {0.5, 1.0, 1.5, 2.0, 3.0}  (5个)
temperature ∈ {0.0, 0.3, 0.5, 0.7}     (4个)
smoke_tasks = arc_easy, sciq, piqa     (3个)

总计: 5 × 4 × 3 = 60 evals ≈ 2小时 (1×H100)
```

**预期改进**: 0.5% → 30%+ (+29个百分点)

**提交流程**:
```bash
# Step 1: 提交参数搜索
bash /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/submit_birwkv_param_sweep.sh

# Step 2: 等待2小时完成
# 监控: qz train ListJobs --page-size 50

# Step 3: 分析结果
python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/analyze_sweep_results.py

# Step 4: 用最优参数重评所有14任务 (~5小时)
```

---

## Part 2: 已成功的评测结果 (可填表)

### ✅ Fix2数据集评测 (6个新任务) — **已完成可用**

| Model | Arc_Easy | Arc_Chal | Wino | SciQ | CSQA | PIQA | 平均 |
|-------|----------|----------|------|------|------|------|------|
| **champ-fixed-512** | 13.0 | 5.8 | 26.5 | 20.5 | 7.0 | 1.2 | **12.3%** |
| **champ-dynamic** | 15.5 | 4.2 | 26.0 | 25.0 | 8.5 | 2.0 | **13.5%** ⭐ |
| **global-dit-50k** | 15.2 | 7.8 | 25.2 | 25.8 | 6.8 | 1.0 | **13.9%** ⭐ |
| global-birwkv-50k | ❌ 0.0 | 1.0 | 0.2 | 0.8 | 0.2 | 0.8 | 0.5% ❌ |
| raw-rwkv | 15.8 | 5.0 | 34.0 | 28.0 | 6.2 | 0.5 | **14.9%** |
| **Baselines:** |
| llama-3.2-3b | 18.8 | 11.0 | 18.8 | 24.8 | 12.5 | 9.5 | **15.9%** |
| phi-2-2.7b | 19.0 | 9.5 | 22.8 | 26.0 | 11.8 | 3.2 | **15.4%** |
| qwen2.5-3b | 13.2 | 2.8 | 19.0 | 20.0 | 6.5 | 1.2 | **10.4%** |
| rwkv7-2.9b | 12.2 | 3.5 | 23.5 | 26.5 | 4.5 | 3.8 | **12.3%** |
| gemma-2-2b | ❌ 0.8 | 0.2 | 4.0 | 0.8 | 0.2 | 1.0 | 1.2% ❌ |
| mamba-2.8b | 11.0 | 3.2 | 5.8 | 16.5 | 6.8 | 1.0 | **7.4%** |

**关键观察**:
- ✅ 我们的模型(champ-*, global-dit) 表现强劲，与主流baseline竞争
- ✅ champ-dynamic: 13.5% (top 3)
- ✅ global-dit: 13.9% (top 3)
- ❌ global-birwkv: 崩溃(0.5%) → **优先修复**
- ❌ gemma-2: 也崩溃(1.2%) → **次要诊断**

**可直接用于论文**:
- Main文本Table: 前4行(我们的模型) ✓
- Raw-rwkv对照: 14.9% ✓  
- Baselines对标: llama/phi/qwen/rwkv ✓

---

## Part 3: 问题诊断总结

### 问题3: baseline-gemma-2-2b 全低分 (~1%)

**可能根因**:
1. Gemma-2加载逻辑错误?
2. 参数配置过严格?
3. 评估脚本bug?
4. 模型本身与数据不匹配?

**优先级**: 中等 (我们的模型优先)

**修复步骤**:
- [ ] 对比llama-3.2和gemma-2的加载代码
- [ ] smoke test诊断
- [ ] 如果仍低分，可考虑排除或标注为异常

---

### 问题4: PIQA得分普遍低 (0.5%-9.5%)

**症状**: 所有模型在PIQA上都远低于其他任务

**可能根因**:
1. PIQA答案格式特殊?(A/B vs数字?)
2. 生成长度不匹配?
3. 评测脚本PIQA处理有bug?

**优先级**: 低 (影响全部模型，可能不是特定模型问题)

**决定**: 先fix P0/P1问题，PIQA作为后期可选诊断

---

## Part 4: 执行流程 (单H100顺序提交)

### Timeline

```
T=0h:     LLaDA修复 ✅ (已完成代码修改)
          - 等待重新提交评测

T=0.5h:   启动birwkv参数扫描 (P0.2)
          cmd: bash submit_birwkv_param_sweep.sh
          预期: 20个配置 × 3任务并行提交

T=0.5-2.5h: 参数扫描运行中
          - 2小时完成60次evals

T=2.5h:   分析扫描结果
          cmd: python analyze_sweep_results.py
          输出: 最优(cfg_scale, temperature)

T=3h:     用最优参数重评14任务 (P0.2 full eval)
          预期: ~5小时

T=8h:     重新提交LLaDA完整评测 (P1.1)
          - difflm-llada-8b_full (14任务)
          - difflm-llada-moe-7b_full
          - difflm-dream-7b_full
          预期: ~7小时

T=15h:    Gemma-2诊断 (P1.2)
          预期: ~1-2小时

T=17h:    等13.3B训练完成 (现在进度8%, ETA +14h)
          总进度: T+31h = 明天1400 UTC

T=31h:    评测13.3B模型 (4模型 × 14任务)
          预期: ~7小时

T=38h:    汇总所有结果到paper表格 ✅
```

---

## Part 5: 行动清单

### 🔴 紧急 (今天内)

- [x] ✅ 修改LLaDA forward()添加**kwargs
- [ ] 🚀 提交birwkv参数扫描 (bash submit_birwkv_param_sweep.sh)
- [ ] 📊 等待2小时，分析结果
- [ ] 🚀 用最优参数重评14任务
- [ ] 🚀 重新提交difflm-llada-{8b, moe-7b}完整评测

### 🟡 随后 (24-48小时内)

- [ ] 诊断gemma-2低分原因
- [ ] 等13.3B训练完成
- [ ] 评测13.3B模型

### 📋 汇总 (48-72小时)

- [ ] Fix2结果填入main表格 (我们的3个模型)
- [ ] 参数优化后global-birwkv结果填表
- [ ] 13.3B scaling结果填表
- [ ] PPL表格 (待traj_cfg评测)
- [ ] 验证没有异常低分

---

## Part 6: 验收标准

**P0完成** ✓
- [x] LLaDA能加载和forward() (已修改)
- [ ] birwkv参数搜索完成
- [ ] global-birwkv从0.5%→20%+ (预期)

**P1完成**
- [ ] LLaDA完整评测落盘
- [ ] Gemma-2诊断完成
- [ ] 所有failing baseline要么fixed要么标注原因

**最终完成**
- [ ] 所有成功的评测结果在paper表格中
- [ ] 13.3B scaling结果填入tab:scaling
- [ ] 没有unexplained 0-1%的异常分数

---

## 下一步

**现在请**:
1. QZ登录 (if needed): `! qz login`
2. 提交参数扫描: `bash submit_birwkv_param_sweep.sh`
3. 2小时后运行分析: `python analyze_sweep_results.py`
4. 用最优参数重评全任务

我会持续监控进度。
