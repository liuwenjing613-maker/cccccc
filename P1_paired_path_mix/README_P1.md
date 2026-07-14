# P1 Paired Path Mix

## 1. 实验定位

P1 只改变训练数据生成方式：

- 模型结构不变
- 损失函数不变
- 训练步数不变
- 验证与最终测试不变
- 只在训练阶段加入“成对路径插值”

因此，P1 与 E0 的结果差异可以归因于路径插值，而不是其他模块。

## 2. 创建实验目录

在仓库根目录执行：

```bash
cp -r strong_baseline P1_paired_path_mix
```

将本补丁中的两个文件覆盖到：

```text
P1_paired_path_mix/dataset.py
P1_paired_path_mix/train.py
```

`model.py` 保持与 `strong_baseline/model.py` 完全一致。

检查：

```bash
diff strong_baseline/model.py P1_paired_path_mix/model.py
```

没有输出才正确。

## 3. 默认路径插值参数

```text
path_mix_probability = 0.5
path_mix_alpha_min = 0.2
path_mix_alpha_max = 0.8
```

在每个完整训练数据周期里：

- 50% 为真实原始路径样本
- 50% 为两条训练路径的插值样本
- 相同噪声、相同起点、相同 alpha 同时用于 d 和 sh
- 未见路径不会参与训练或插值

## 4. 冒烟测试

```bash
python P1_paired_path_mix/train.py \
  --seeds 2026 \
  --max-steps 20 \
  --eval-interval 10 \
  --output-dir P1_paired_path_mix/smoke_test
```

## 5. 完整实验

```bash
python P1_paired_path_mix/train.py
```

默认仍然是：

- seeds: 2026, 2027, 2028
- max_steps: 15000
- batch_size: 8
- eval_interval: 500

## 6. 关闭路径插值以做回归检查

```bash
python P1_paired_path_mix/train.py \
  --seeds 2026 \
  --path-mix-probability 0 \
  --output-dir P1_paired_path_mix/regression_no_mix
```

此时训练行为应与 E0 基本一致，用于确认修改没有破坏原流程。

## 7. 输出

默认输出：

```text
P1_paired_path_mix/outputs_p1/
```

重点比较：

```text
multi_seed_aggregate.json
multi_seed_summary.csv
seed_*/evaluations/final_unseen_noise_seen_paths_paths.csv
seed_*/evaluations/final_unseen_noise_unseen_paths_paths.csv
```

## 8. 成功标准

相对 E0：

- Seen 平均 NR 不下降超过 0.3 dB
- Unseen 平均 NR 提升至少 0.5 dB
- Seen 最差路径高于 0.77 dB
- Unseen 最差路径高于 1.47 dB
- 负 NR 样本仍为 0

如果平均值提升但最差路径下降，不能判定成功。
