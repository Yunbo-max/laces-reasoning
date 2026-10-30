# LACES 8个模型训练状态 (2.9B完整矩阵)

**日期**: 2026-09-20 06:20 UTC

---

## 📊 2.9B模型矩阵 (2×2×2)

```
                   DiT          BiRWKV
                   ───          ──────
Chunked-Dynamic    ❌ 无         ❌ 无 (20k/50k)
Chunked-Fixed      ❌ 无         ❌ 无

Global-Dynamic     ✅ 50k        ✅ 50k
Global-Fixed       ✅ 完成       ✅ 50k
```

---

## ✅ 已完成的模型 (3个)

### 1. **Global-DiT-Dynamic** ✅
- 路径: `outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k`
- 进度: **100% (50000/50000 steps)**
- 参数: cfg_scale=3.0, temperature=0.0
- 评测状态: ✅ Fix2完成
- 新数据集: ❌ 未测
- 可用于论文: ✅

### 2. **Global-BiRWKV-Dynamic** ✅
- 路径: `outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k`
- 进度: **100% (50000/50000 steps)**
- 参数: cfg_scale=3.0, temperature=0.0 (但性能差 0.5%)
- 评测状态: ✅ Fix2完成 (⚠️ 异常低)
- 新数据集: ❌ 未测
- 优化状态: 🔴 参数扫描无结果
- 可用于论文: ❌ 需修复

### 3. **Global-BiRWKV-Fixed** ✅
- 路径: `outputs_dynamic_basis/laces-2.9B-fixed-r32-global-birwkv768x8-b16-50k`
- 进度: **100% (50000/50000 steps)**
- 参数: cfg_scale=3.0, temperature=0.0
- 评测状态: ✅ Fix2完成
- 新数据集: ❌ 未测
- 可用于论文: ✅

### 4. **Global-DiT-S0Frozen** (变体，60%)
- 路径: `outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03`
- 进度: **60% (30000/50000 steps)**
- 评测状态: ✅ Fix2完成 + 5数据集完成
- 备注: 这个是s0frozen版本，不在8模型矩阵内
- 可用于论文: ✅

---

## ❌ 缺失的模型 (4个)

### 1. **Chunked-DiT-Dynamic** ❌
- 状态: 完全不存在
- 需要: 训练 (~50小时)

### 2. **Chunked-DiT-Fixed** ❌
- 状态: 完全不存在
- 需要: 训练 (~50小时)

### 3. **Chunked-BiRWKV-Dynamic** ❌
- 状态: 不完整 (DiffRwkv有20k版本)
- 位置: `DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-xchunk`
- 进度: 20000/50000 (40%)
- 选项: 继续训 + 30k or 重新训

### 4. **Chunked-BiRWKV-Fixed** ❌
- 状态: 可能在DiffRwkv (需找)
- 备注: 之前提到过 `traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch`

### 5. **Global-DiT-Fixed** ❌
- 状态: 不存在
- 需要: 训练 (~50小时)

---

## 🎯 现实的8模型矩阵情况

```
✅ 能做评测的:     3个Global模型 (都已50k)
❌ 无法做评测的:   4个Chunked模型 (全不存在或不完整)

当前可用于论文:    3个模型 (缺失4个Chunked的对比)
```

---

## 📊 13.3B模型训练状态

| 模型 | 进度 | ETA |
|------|------|-----|
| 13.3B-global-birwkv | 30.6% (15300/50000) | Sep 24 |
| 13.3B-chunked-birwkv | 30.6% (15300/50000) | Sep 24 |
| 13.3B-global-nb32 | 30.6% (15300/50000) | Sep 24 |
| 13.3B-chunked-nb32 | 30.6% (15300/50000) | Sep 24 |

---

## 🔴 关键问题总结

### 问题1: 8个2.9B模型的完整性
```
目标: 2×2×2 = 8个模型的消融
现状: 只有3个Global模型 + 1个s0frozen变体
缺失: 4个Chunked模型

解决方案:
  选项A (快速): 放弃Chunked，只用4个Global (但破坏矩阵)
  选项B (完整): 补训4个Chunked (需 200+ GPU小时)
  选项C (混合): 只用现有的3个完整模型 + 继续13.3B训练
```

### 问题2: global-birwkv-50k性能异常
```
现象: 0.5% 准确率 (应该15%+)
原因: 参数不适配
进展: Sep 19提交的参数扫描无结果
状态: 🔴 未修复
```

### 问题3: 评测不完整
```
Fix2数据集:
  ✅ 4个模型都有 (包括s0frozen)

5个新数据集:
  ✅ champ-dynamic (s0frozen) 完整
  ❌ 其他3个模型: 0/5

GSM8K:
  ❌ 所有模型都没测
```

---

## 💡 建议的行动方案

### 立即 (能快速完成)

1. **补评测新数据集** (3个模型 × 5数据集)
   - 时间: 8-10小时
   - 产出: 完整的4模型×5数据集对标表

2. **修复global-birwkv参数**
   - 时间: 已提交参数扫描，等结果
   - 或手动尝试cfg=1.0, temp=0.3等

3. **评测GSM8K**
   - 时间: 4小时
   - 用4个已训好的模型

### 后续 (Sep 24后)

4. **13.3B模型评测**
   - Fix2 + 新数据集 + GSM8K
   - 生成Scaling表

### 可选 (需要时间)

5. **补训Chunked模型**
   - 200+ GPU小时
   - 完成8模型矩阵

---

## 📋 论文现在能填什么

| 内容 | 完成度 | 备注 |
|------|--------|------|
| 2.9B Fix2表 | 80% | 缺global-birwkv修复版 |
| 2.9B新数据集 | 25% | 只有1/4模型 |
| Scaling表 | 0% | 等13.3B Sep 24 |
| GSM8K表 | 0% | 全部待测 |

---

## 你的决定

**当前最关键的**:
1. **放弃Chunked矩阵** (只用3个Global) 还是 **等待补训**?
2. **立即补评测** (3个模型×5数据集) 还是 **等13.3B先**?
3. **修复birwkv** 还是 **标注为异常值**?

