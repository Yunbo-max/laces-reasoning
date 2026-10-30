# 明天的评测优化流程

## 优先级 1: 等参数搜索完成 ✅

### BiRWKV 参数搜索 (进行中)
- 搜索空间: cfg ∈ [0.3-3.0] × temp ∈ [0.0-1.0] (60个组合)
- 当前: 33/60 任务成功
- 预期完成: 几小时内
- **关键**: 找到最优 (cfg, temp) 替换掉 cfg=3.0, temp=0.0

### DiT 参数搜索 (进行中)
- 搜索空间: cfg ∈ [0.5-5.0] × temp ∈ [0.0-1.0] (60个组合)
- 当前: 30/60 任务成功
- 预期完成: 几小时内
- **关键**: 验证 cfg=3.0 是否最优，或找到更好的参数

---

## 优先级 2: 等 DiT + fixed 50k 训完 ✅

- 预期完成: ~9小时（今晚/明天早上）
- 训完后立即用**最优参数** eval (不是 cfg=3.0, temp=0.0!)

---

## 优先级 3: 用最优参数重新评测所有4个模型

**当参数搜索完成后**，提取最优参数：

### 4个模型重新eval流程:
```
for model in [global-dit-50k, global-birwkv-50k, global-birwkv-fixed-50k, dit-fixed-50k]:
    best_cfg, best_temp = 从参数搜索结果提取最优参数
    eval(model, cfg=best_cfg, temp=best_temp)  # 用最优参数，不是默认的!
```

### 具体的model对应的最优参数来源:
- **global-dit-50k** → 从 `results/benchmarks/global-dit-fullsearch/` 找最优
- **global-birwkv-50k** → 从 `results/benchmarks/global-birwkv-fullsearch/` 找最优
- **global-birwkv-fixed-50k** → 用birwkv搜索的最优参数 (architecture相同)
- **dit-fixed-50k** → 用dit搜索的最优参数 (architecture相同)

---

## ⚠️ 关键检查清单

- [ ] 参数搜索结果出现在 `results/benchmarks/global-*-fullsearch/`
- [ ] 解析每个模型的最优参数 (不是手动设置 cfg=3.0, temp=0.0)
- [ ] **确保每个模型都用对应的最优参数 eval**
- [ ] 不要偷懒用默认参数!

---

## 预期时间表

| 任务 | 预期完成 | 优先级 |
|------|---------|--------|
| 参数搜索 (birwkv + dit) | 今晚 ~3-5h | 🔴 立即 |
| DiT + fixed 50k 训练 | 明天早上 | 🔴 立即 |
| 用最优参数重eval所有4个模型 | 明天下午 | 🟠 高 |
| 生成最终论文表格 | 明天傍晚 | 🟢 完成 |
