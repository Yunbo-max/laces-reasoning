# LACES 模型训练真实状态 (2026-09-20 07:16 UTC)

## ✅ 现在**正在活跃训练**的模型

### 13.3B 模型 (4个，都在训)

| 模型 | 进度 | 最新更新 | 状态 |
|------|------|---------|------|
| **13.3B-global-birwkv** | 31.2% (15620/50k) | 07:15:08 | 🟢 训中 |
| **13.3B-chunked-birwkv** | 31.9% (15990/50k) | 07:14:45 | 🟢 训中 |
| **13.3B-global-nb32** | 31.7% (15850/50k) | 07:15:30 | 🟢 训中 |
| **13.3B-chunked-nb32** | 31.1% (15580/50k) | 07:14:34 | 🟢 训中 |

**ETA**: Sep 24 (还需 ~118小时)

---

## ✅ 已完成的2.9B模型

### Global 模型 (3个完成)

| 模型 | 完成度 | 状态 |
|------|--------|------|
| **global-dit-dynamic** | 100% (50k) | ✅ 完成 |
| **global-birwkv-dynamic** | 100% (50k) | ✅ 完成 |
| **global-birwkv-fixed** | 100% (50k) | ✅ 完成 |

---

## 🔴 **停止训练的2.9B Chunked模型** (在DiffRwkv)

| 模型 | 完成度 | 最后更新 | 路径 |
|------|--------|---------|------|
| chunked-birwkv-dynamic (xchunk) | 40% (20k) | 2026-07-05 | `DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-xchunk` |
| chunked-birwkv-fixed (coadapt) | 24% (12k) | 2026-07-06 | `DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-condadaln-coadapt-fixed` |
| chunked-dit-? | - | - | 不存在 |
| chunked-dit-? | - | - | 不存在 |

**状态**: 🔴 **已停止** (7月份就没更新了)

---

## 📊 完整的8模型矩阵 (2.9B)

```
                   DiT          BiRWKV
                   ───          ──────
Chunked-Dynamic    ❌ 无         🔴 20k/50k (停了)
Chunked-Fixed      ❌ 无         🔴 12k/50k (停了)

Global-Dynamic     ✅ 50k        ✅ 50k
Global-Fixed       ❓ 不确定     ✅ 50k
```

**关键发现**:
- ✅ 3个Global模型完成
- 🔴 2个Chunked模型**停止训练**，远未完成 (40%和24%)
- ❌ 2个Chunked-DiT模型**完全不存在**

---

## 💡 现状分析

### 2.9B Chunked模型为什么停了?

**可能的原因**:
1. 实验失败被放弃
2. 转向13.3B模型优先
3. 资源被13.3B训练占用
4. 计划改变 (只关注Global)

**是否需要恢复**?
- 如果要完整的8模型消融矩阵: 需要续训 + 新训2个DiT
- 如果只用Global对比: 现在就够了

---

## 🎯 **你的选择**

### 方案A: 只用3个Global模型 (快)
- 立即评测新数据集 (3个模型 × 5数据集)
- GSM8K评测
- Sep 24完整13.3B后就能出论文

### 方案B: 恢复Chunked模型 (慢)
- 续训 chunked-birwkv (30k到50k)
- 新训 2个 chunked-dit (0到50k)
- 需要 ~200+小时额外训练

### 方案C: 混合 (平衡)
- 先用3个Global完成论文
- Chunked作为可选补充/future work

---

## 📋 **建议的立即行动**

**今天**:
1. 补评测: 3个Global模型 × 5新数据集 (8-10小时)
2. GSM8K评测: 4个模型 (4小时)
3. 修复 global-birwkv (还是0.5%)

**Sep 24**:
4. 13.3B评测完成 → Scaling表生成

**论文成稿**:
- 完整的数据就有了

---

**你想怎么处理这2.9B的chunked?** 
- 放弃 → 用现有3个Global
- 恢复 → 需要多等几天
- 其他选择?

