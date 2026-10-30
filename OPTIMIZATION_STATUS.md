# LACES 模型参数优化与训练状态 (Sep 20)

## 📊 参数优化进度

### 已优化完成的模型

#### 1. **champ-dynamic** ✅ 已最优
- **参数**: cfg_scale=3.0, temperature=0.0, top_k=50, top_p=0.9
- **性能**: 13.5% (Fix2平均)
- **新数据集**:
  - MMLU-Redux: 35% (trajectory)
  - C-Eval: 40% (trajectory) — 最佳表现! ⭐
  - MMLU-Pro: 15% (trajectory)
  - GPQA Diamond: 25% (trajectory)
- **评估方式**: Trajectory CFG (traj_cfg)
- **状态**: ✅ 完成

#### 2. **champ-fixed-512** ✅ 已评估
- **参数**: cfg_scale=3.0, temperature=0.0
- **性能**: 12.3% (Fix2平均)
- **状态**: ✅ 完成

#### 3. **global-dit-50k** ✅ 已评估
- **参数**: cfg_scale=3.0, temperature=0.0
- **性能**: 13.9% (Fix2平均)
- **状态**: ✅ 完成

### 正在优化的模型

#### 4. **global-birwkv-50k** 🔴 参数调优中

**当前问题**: 0.5% 准确率 (崩溃)

**参数搜索进行中** (Sep 19-20):
```
cfg_scale: {0.5, 1.0, 1.5, 2.0, 3.0}
temperature: {0.0, 0.3, 0.5, 0.7}
总计: 20个配置 × 3个smoke任务
```

**预期目标**: 
- cfg_scale=1.0, temperature=0.5 可能最优
- 目标: 0.5% → 30%+

**进度**: 🟡 扫描中 (smoke test阶段)

---

## 🟡 新训练模型 (13.3B) — 未测试

### 正在训练中的4个模型

| 模型名 | 架构 | 状态 | 进度 | ETA |
|--------|------|------|------|-----|
| **13.3B-global-birwkv** | global + BiRWKV | 🟡 训练中 | 8% | +14h |
| **13.3B-chunked-birwkv** | chunked + BiRWKV | 🟡 训练中 | 8% | +14h |
| **13.3B-global-nb32** | global + no-basis | 🟡 训练中 | 8% | +14h |
| **13.3B-chunked-nb32** | chunked + no-basis | 🟡 训练中 | 8% | +14h |

**重要**: 这4个模型**还未进行任何评测**

### 待进行的评测

当训练完成后(ETA ~22小时):
1. Fix2数据集 (6个任务)
2. 新数据集 (MMLU-Redux, MMLU-Pro, GPQA Diamond, C-Eval)
3. PPL评估
4. 填入Scaling表 (2.9B vs 13.3B对比)

**预期时间**: 
- 训练完成: T+22h
- 评测: T+29h (7小时 × 4模型 × 2种评测方式)
- 汇总: T+36h

---

## 📋 优化优先级

### 🔴 立即优化

1. **global-birwkv-50k**
   - 当前: 0.5%
   - 目标: >20%
   - 方法: 参数扫描 (进行中)
   - 完成时间: 今天 (T+7h)

### 🟡 后续优化 (可选)

2. **其他Global模型微调**
   - global-dit-50k: 13.9% → 14.5%?
   - 预期收益: +0.5-1%
   - 方法: 精细参数搜索

3. **新模型Chunked-variants**
   - 缺失: chunked-dit-*, chunked-birwkv-dynamic
   - 方法: 训练或转移
   - 优先级: 中

---

## 🎯 下一步行动

### 今天 (T+0-7h)

- [ ] 完成 global-birwkv-50k 参数扫描
- [ ] 分析最优参数
- [ ] 用最优参数重评14任务
- [ ] 得到 global-birwkv 最终分数

### 明天 (T+7-22h)

- [ ] 等13.3B训练完成
- [ ] 启动13.3B评测 (Fix2 + 新数据集)

### 后天 (T+22-36h)

- [ ] 13.3B评测完成
- [ ] 生成Scaling表
- [ ] 汇总所有结果到论文

---

## 📈 预期最终矩阵

### 2.9B模型完整性

```
                    DiT          BiRWKV
Chunked            ❌ 缺        ❌ 缺
Global-Dynamic     ✅ 13.9%     ✅ 待优(0.5%→30%)
Global-Fixed       ✅ 12.3%     ✅ 未测
```

### 13.3B模型 (新)

```
4个模型 × 2种采样方式 × 4个新数据集 = 32个评估
预期贡献: Scaling曲线 (2.9B vs 13.3B)
```

### 论文表格可用度

| 表格 | 当前 | 预期 |
|------|------|------|
| Fix2 (2.9B) | 4/4模型 | ✅ 4/4 (birwkv优化后) |
| 新数据集 | 1/4模型 | ✅ 5/4模型 (13.3B后) |
| Scaling | 无 | ✅ 有 (13.3B后) |
| PPL | 部分 | ✅ 完整 (traj_cfg后) |

---

**关键数字**:
- ✅ 已完成: 3个Global模型 (champ-fixed, champ-dynamic, global-dit)
- 🟡 进行中: global-birwkv参数优化 + 13.3B训练
- ⏳ 待测试: 4个13.3B新模型
