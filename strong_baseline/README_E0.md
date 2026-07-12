# E0 Strong Baseline Patch

## 用法

1. 在仓库根目录复制原版本：

```bash
cp -r sample2048 exp3_strong_baseline
```

2. 用本压缩包中的 `dataset.py` 和 `train.py` 覆盖：

```text
exp3_strong_baseline/dataset.py
exp3_strong_baseline/train.py
```

3. `model.py` 保持原样，不要修改。

4. 先做冒烟测试：

```bash
python exp3_strong_baseline/train.py \
  --seeds 2026 \
  --max-steps 20 \
  --eval-interval 10 \
  --output-dir exp3_strong_baseline/smoke_test
```

5. 冒烟测试正常后运行完整实验：

```bash
python exp3_strong_baseline/train.py
```

默认设置：

- 15000 个优化步
- 每 500 步验证一次
- 随机种子 2026、2027、2028
- batch size 8
- 路径 × 噪声严格均衡
- 路径 × 固定时间窗口网格评估
- 保存 `best.pt` 与 `last.pt`
- 汇总最差路径、负 NR 数量和三随机种子均值/标准差

## 主要输出

```text
exp3_strong_baseline/outputs_e0/
├── run_config.json
├── multi_seed_summary.csv
├── multi_seed_aggregate.json
├── seed_2026/
│   ├── checkpoints/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── evaluations/
│   │   ├── validation_best_*.csv/png/json
│   │   ├── final_unseen_noise_seen_paths_*.csv/png/json
│   │   └── final_unseen_noise_unseen_paths_*.csv/png/json
│   ├── validation_history.csv
│   ├── data_split.json
│   └── seed_summary.json
└── seed_2027/...
```
