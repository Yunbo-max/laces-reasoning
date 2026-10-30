# LACES 2.9B 模型全清单 (8个)

## ✅ 已有的4个 Global 模型

| 名称 | 类型 | Checkpoint | 状态 |
|-----|------|-----------|------|
| **global-dit-dynamic** | global + DiT + Dynamic | `laces-2.9B-dynlowrank-r32-global-joint-b16-50k` / step_00050000 | ✅ 已评估 |
| **global-birwkv-dynamic** | global + BiRWKV + Dynamic | `laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k` / step_00050000 | ✅ 已评估(坏) |
| **global-birwkv-fixed** | global + BiRWKV + Fixed | `laces-2.9B-fixed-r32-global-birwkv768x8-b16-50k` / step_00050000 | ✅ 已评估 |
| **global-dit-dynamic-s0frozen** | global + DiT + Dynamic (s0冻结) | `laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03` / step_00030000 | ✅ 已评估 |

---

## ❌ 缺少的4个 Chunked 模型

| 名称 | 类型 | 位置 | 需要找到 |
|-----|------|------|--------|
| **chunked-dit-dynamic** | chunked + DiT + Dynamic | ❌ 不存在 | **需要训练或找到** |
| **chunked-dit-fixed** | chunked + DiT + Fixed | ❌ 不存在 | **需要训练或找到** |
| **chunked-birwkv-dynamic** | chunked + BiRWKV + Dynamic | ❌ 不存在 | **需要训练或找到** |
| **chunked-birwkv-fixed** | chunked + BiRWKV + Fixed | `/DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch` / step_00026000 | ⏳ 需要转移到laces-reasoning |

---

## 🔍 在DiffRwkv中找到的Chunked候选

```
traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000
  ↑ 这个是 chunked-birwkv-fixed (32×16 = 512 tokens/chunk)
  
还需要找:
  - chunked-dit variants
  - chunked-birwkv-dynamic (可能是 traj64x64 的?)
```

---

## 现在的状态

✅ **Can evaluate now:**
- global-dit-dynamic (步长50k)
- global-birwkv-dynamic (步长50k) — 需要参数调优
- global-birwkv-fixed (步长50k)
- global-dit-s0frozen (步长30k)

❌ **Cannot evaluate without:**
- 4个Chunked模型 (全部缺失或未转移)

---

## 行动项

1. **找到或训练缺失的Chunked模型：**
   - 搜索DiffRwkv/outputs_relay中是否有其他traj32x16或traj64x64的chunked模型
   - 如果没有，需要在laces-reasoning中**重新训练**这4个模型

2. **转移已有的Chunked-BiRWKV-Fixed：**
   ```bash
   # Copy from DiffRwkv to laces-reasoning
   cp -r /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch \
       /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_fixed_basis/
   ```

3. **训练缺失的模型** (如果需要完整的8个):
   ```
   - chunked-dit-dynamic: 16h per GPU
   - chunked-dit-fixed: 16h per GPU
   - chunked-birwkv-dynamic: 16h per GPU
   - chunked-birwkv-fixed: (如果DiffRwkv中的不适用)
   ```

---

## 问题

**为什么要8个模型？** 
- **Chunked vs Global**: 执行范式的对比 (并行 vs 串行)
- **DiT vs BiRWKV**: Denoiser架构的对比
- **Dynamic vs Fixed**: Writer类型的对比
- 完整的2×2×2 = 8个矩阵，用于消融

**目前只有Global的4个，缺Chunked的4个。**

你想:
1. **立即用现有4个Global模型做参数优化?** (推荐，快速见效)
2. **先完成缺失的Chunked模型?** (慢，但完整)
3. **同时进行?** (最全，但时间长)
