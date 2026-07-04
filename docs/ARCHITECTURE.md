# CCF_DEEPANC_2026 基线系统架构说明

> 本文档面向**想读懂并改动基线代码的开发者 / 参赛选手**,在 `README.md`(赛题背景与基线性能)之上,逐文件、逐函数讲清**数据集格式与划分、Dataloader、Loss、评分脚本、模型架构、训练/推理流程**,并附张量形状、设计动机与可改进项。所有结论均按当前源码逐行核实,文末附 `文件:行号` 速查索引。
>
> 适用代码版本:`model.py` / `dataset.py` / `train.py` / `README.md` 四文件基线(含官方 **2026-06-27 因果性修正**:`dataset.py` 已移除局部动态归一化,详见 §4.4)。数据集 `餐厅` 命名已修复为小写(详见 §2.1);§7 已补**官方评分规则**(§7.4)、§5 已补**模型复杂度**(§5.4),§7.3 基线性能改用**本地 60 epoch 实跑结果**。

## 目录

1. [系统概览](#1-系统概览)
2. [数据集格式](#2-数据集格式)
3. [数据集划分(train / test)](#3-数据集划分train--test)
4. [Dataloader 详解](#4-dataloader-详解)
5. [模型架构 TimeDomainANC](#5-模型架构-timedomainancmodelpy)
6. [Loss 设计](#6-loss-设计trainpy171-178)
7. [评分脚本与评估指标](#7-评分脚本与评估指标trainpy12-111)
8. [训练与推理流程](#8-训练与推理流程)
9. [已知设计权衡与可改进项](#9-已知设计权衡与可改进项)
10. [关键文件与代码位置速查](#10-关键文件与代码位置速查)

---

## 1. 系统概览

### 1.1 任务定位
**Deep ANC(深度学习主动降噪)**:神经网络在**时域**直接从参考噪声 `x(n)` 预测反相驱动信号 `y(n)`,目标是让 `y(n)` 经过**次级声学路径** `S(z)` 后,在误差麦克风处抵消**期望噪声** `d(n)`。

物理模型(`README.md` 第 29 行):

$$e(n) = d(n) - \hat{d}(n) = \underbrace{[x(n) * P(z)]}_{d(n),\ 已离线给定} - \underbrace{[y(n) * S(z)]}_{a(n),\ 代码现场卷积}$$

- **初级路径 $P(z)$** 被组委会**物理隐藏**,不提供;`d(n)` 已离线算好。
- 选手只能拿到三样东西:参考噪声 `x(n)`、期望噪声 `d(n)`、次级路径 `S(z)`。
- 网络必须**纯数据驱动**地隐式学习 `P(z)` 与 `S(z)` 的规律。

### 1.2 四文件职责

| 文件 | 行数 | 职责 |
|---|---|---|
| `model.py` | 93 | 定义 `TimeDomainANC`(因果空洞卷积 + 残差块的纯时域 1D-CNN)|
| `dataset.py` | 141 | `PreconvolutedANCDataset`(数据集)+ `apply_dynamic_path`(次级路径物理卷积)|
| `train.py` | 200 | 数据集划分、DataLoader、训练循环、`evaluate_and_plot`(评分 + 可视化)、入口 |
| `README.md` | 129 | 赛题背景、物理模型、基线性能参考 |

### 1.3 端到端数据流

```
                ┌─────────────┐   y_t           ┌──────────────────┐  a_t
 x_t [B,48000]→ │ TimeDomainANC│ ─────────────→ │ apply_dynamic_path│ ──┐
 (参考噪声)      └─────────────┘  [B,48000]      │  (× 次级路径 sh)   │   │
                                                 └──────────────────┘   │
 sh [B,1967] ───────────────────────────────────────────┘              │
                                                                        ▼
 d_t [B,48000] (期望噪声) ───────────────────────────────→  e_t = d_t − a_t  [B,48000]
                                                                        │
                              训练: loss = mean(e_t²)  ◄────────────────┤
                              评估: NR = 10·log10(Σd² / Σe²) ◄──────────┘
```

三个核心张量(Dataset 返回的三元组)贯穿全程:`x_t`(输入)、`sh`(物理路径)、`d_t`(目标)。

---

## 2. 数据集格式

数据放在项目根目录 `dataset/` 下,由三部分组成:

| 数据 | 路径 | 格式 / 形状 | 物理含义 |
|---|---|---|---|
| 参考噪声 `x(n)` | `dataset/NOISE/{噪声名}.wav` | **16-bit PCM** WAV,48 kHz,单声道,**8 个文件** | 参考麦克风采集的原始环境宽带噪声 |
| 期望噪声 `d(n)` | `dataset/EXPECTED_NOISE/{噪声名}_scene_{NN}.wav` | **32-bit IEEE float** WAV,48 kHz,单声道,**80 个文件**(噪声 × 10 路径) | 噪声经隐藏初级路径 $P(z)$ 衰减后到达误差麦克风的目标声 |
| 次级路径 `S(z)` | `dataset/sh.npy` | NumPy `float64` 数组 | 扬声器 → 误差麦克风的物理冲激响应 |

> ⚠️ **位深差异(实测)**:`NOISE/` 是 **16-bit PCM 整数**,而 `EXPECTED_NOISE/` 是 **32-bit IEEE 浮点**,两者并不相同。代码统一用 `sf.read(..., dtype='float32')` 读取(`dataset.py:73`),`soundfile` 会把 16-bit PCM 自动解码缩放到 `[−1, 1)` 浮点,浮点文件则原样读出,故上层无需关心源位深。

### 2.1 噪声种类(8 个物理文件)
`车载`、`公交`、`地铁`、`火车`、`餐厅`、`KTV`、`厨房`、`步行街`,**均为小写 `.wav` 扩展名**,8 个全部被基线识别。

### 2.2 期望噪声命名规则
`{噪声名}_scene_{path_idx+1:02d}.wav`,其中 `path_idx` 是 0-based 路径下标(`dataset.py:90`):

| path_idx (代码内) | 文件后缀 |
|---|---|
| 0 | `_scene_01.wav` |
| 1 | `_scene_02.wav` |
| … | … |
| 9 | `_scene_10.wav` |

例:`KTV_scene_01.wav` … `KTV_scene_10.wav`。每个噪声对应 10 条声学路径(scene),共 8×10 = 80 个期望文件。

### 2.3 次级路径 `sh.npy`
- **磁盘上实际形状**:`(1967, 10)`,`float64`,即 `(Length, Num_Paths)`。
- 代码装载时**做了转置**:`self.sh_paths = np.load('sh.npy').T`(`dataset.py:60`)→ 逻辑形状变为 `(10, 1967)` = `(Num_Paths, Length)`。
- `self.sh_paths[path_idx]` 取出第 `path_idx` 条路径的**冲激响应向量**,长度 `L = 1967` 个样本(@48 kHz ≈ **41 ms**)。
- 取出后转成 `float32`(`dataset.py:140`,在返回三元组时统一转换)。

> 因此 README 里写的"维度 `(Num_Paths, Length)`"指的是**转置后**的逻辑形状,而非磁盘存储顺序——务必记得代码里的 `.T`。

---

## 3. 数据集划分(train / test)

划分逻辑全部在 `train.py:122-149`,分**噪声**和**声学路径**两个独立维度。**没有独立验证集**,训练完直接跑两个测试集。

### 3.1 噪声维度划分
```python
all_noise_names = sorted([splitext(f)[0] for f in os.listdir(noise_dir) if f.endswith('.wav')])
train_noises = all_noise_names[:-2]   # 除最后 2 个外
test_noises  = all_noise_names[-2:]   # 最后 2 个(用于测试拼接)
```

8 个噪声**全部识别**,`sorted()` **按 Unicode 码点排序(非拼音)**——`KTV` 是 ASCII、排在所有汉字之前,其余汉字按码点序。最终顺序为
`['KTV', '公交', '厨房', '地铁', '步行街', '火车', '车载', '餐厅']`,取后 2 个作测试:

| 识别(8 个,按码点序) | 划分 |
|---|---|
| `KTV`、`公交`、`厨房`、`地铁`、`步行街`、`火车` | **训练噪声(6 个)** |
| `车载`、`餐厅` | **测试噪声(2 个,未见)** |

> 与 README"8 类噪声 / 后 2 个测试"一致:**训练 6 / 测试 2**。注意"后 2 个"由**码点排序**决定(`车载`/`餐厅`),改文件名会改变划分。

### 3.2 声学路径维度划分(`train.py:139-143`)
```python
num_paths = 10
train_count = int(num_paths * 0.8)            # = 8
train_path_indices = list(range(0, 8))        # [0..7]  → _scene_01..08
test_path_indices  = list(range(8, 10))       # [8, 9]  → _scene_09, _scene_10
```
- 训练路径:8 条(idx 0–7)
- 测试路径:2 条(idx 8–9,未见)

### 3.3 三个数据集 / 两类测试场景

| 数据集对象 | 噪声 | 路径 | `is_train` | 用途 |
|---|---|---|---|---|
| `train_dataset` | 训练噪声(6)| 训练路径(0–7)| `True` | 训练 |
| `test_dataset_seen_paths`(**Test 1**)| 测试噪声(2)| **训练路径(0–7)** | `False` | 未见噪声 + 已见路径 → 噪声泛化(8 个场景)|
| `test_dataset_unseen_paths`(**Test 2**)| 测试噪声(2)| **测试路径(8–9)** | `False` | 未见噪声 + 未见路径 → 跨物理空间双重泛化(2 个场景)|

Test 1 有 8 个评估场景、Test 2 有 2 个,与 README 基线性能表给出的 NR 数量(8 个 / 2 个)一致。

---

## 4. Dataloader 详解

核心类 `PreconvolutedANCDataset`(`dataset.py:44-141`)+ 物理卷积函数 `apply_dynamic_path`(`dataset.py:13-38`),框架为原生 PyTorch。

### 4.1 返回的三元组
每个样本返回严格对齐的三元组(`dataset.py:139-141`),均为 `float32`:

| 张量 | 符号 | 形状(单样本)| 含义 |
|---|---|---|---|
| `seg_raw` | `x_t` | `(48000,)` | 参考噪声,1 秒 @48 kHz;幅值落在 `[−1,1)`(16-bit PCM 经 `soundfile` 解码所致,**非**逐片段动态归一化,见 §4.4)|
| `sh` | `sh` | `(1967,)` | 该样本对应路径的次级路径冲激响应 |
| `seg_exp` | `d_t` | `(48000,)` | 期望目标噪声(32-bit 浮点、已离线预归一化),与 `x_t` 严格对齐,**不再做任何现场缩放** |

`segment_length = int(1.0 × 48000) = 48000`(`dataset.py:55`)——**所有样本定长**。

### 4.2 `__len__`:每个 epoch 的样本数(关键)
```python
def __len__(self):
    return len(self.path_indices)        # dataset.py:65-66
```
- **数据集长度 = 路径数,而非音频时长**。训练集 `path_indices=[0..7]` ⇒ **`len=8`**。
- 噪声种类与音频起点是在 `__getitem__` 内**随机抽取**的(见下),路径下标只决定用哪条 `sh` 和哪个 `_scene_NN` 期望文件。
- 含义:`batch_size=8` 时,**每个 epoch 仅 1 个 batch、仅 8 个随机 1 秒片段**。这是数据曝光偏低的根源,详见 §9。

### 4.3 `__getitem__`:训练随机采样 vs 测试确定性拼接

**读取工具** `_fast_read_slice`(`dataset.py:68-77`):用 `soundfile` 指针级切片读取(`sf.read(start=, frames=)`),**不全量载入**长音频;多声道时取均值转单声道。

**训练分支** `is_train=True`(`dataset.py:86-100`):
1. `random.choice(noise_names)` 随机选一种训练噪声;
2. 用 `sf.info(...).frames` 取总帧数,`np.random.randint` 在 `[20s, 末尾]` 区间随机选起点(跳过前 20 秒瞬态);
3. 读取该噪声的 `raw`(NOISE)与 `exp`(EXPECTED_NOISE,后缀按 `path_idx+1`)各 1 秒。

**测试分支** `is_train=False`(`dataset.py:102-131`):**中点拼接两种测试噪声**模拟环境突变——
- 前半秒取自 `test_noises[0]`(`车载`),后半秒取自 `test_noises[1]`(`餐厅`);
- 起点随 `idx` 递进(`skip_samples + idx*half_len`),并对超界做取模回绕保护;
- `raw` 与 `exp` 都按同样方式拼接,保证对齐。

**尾部零填充**(`dataset.py:133-137`):片段不足 `segment_length` 时右侧补零。

### 4.4 关于归一化(官方因果性修正:已移除局部动态归一化)
> 本节描述的是一处**已被官方删除**的旧逻辑,保留说明是为了帮助读旧代码 / 旧资料的人对齐。

**修正前**,`__getitem__` 末尾曾有 3 行逐片段动态归一化:
```python
# —— 旧代码,已于 2026-06-27 官方修正中删除 ——
norm_factor = np.max(np.abs(seg_raw)) + 1e-8
seg_exp = seg_exp / norm_factor
seg_raw = seg_raw / norm_factor
```

**为什么删?——它是非因果(non-causal)的**:`np.max(np.abs(seg_raw))` 取的是**整段 1 秒片段的全局峰值**,相对于片段内"当前样本"而言,这个峰值可能落在**未来**。真实 ANC 是逐点实时 DSP,只能拿到过去与当前样本、无法预知未来 1 秒内的局部极值;用一个"未来峰值"去缩放当前样本,破坏了数据流转的严格因果性(官方公告见 `README.md` 顶部,落款 2026-06-27)。

**为什么删了影响很小**:所有数据在上传前已做过**离线归一化**,官方回归测试显示该 3 行删除后**基线 NR 与收敛表现基本不变**。

**删除后 `x→d` 的增益比仍然保留**:现在对 `x_t` / `d_t` **完全不再做任何现场缩放**,二者直接沿用源文件(离线预归一化)中的相对幅值关系,因此隐藏初级路径 $P(z)$ 的物理幅值/能量比依旧成立,抵消所需的相对能量不受影响。当前输入落在 `[−1,1)` 仅仅是 16-bit PCM 经 `soundfile` 解码到浮点的自然结果(见 §2 位深注记)。

### 4.5 DataLoader 参数(`train.py:151-153`)

| Loader | batch_size | shuffle | collate_fn | num_workers | sampler |
|---|---|---|---|---|---|
| `train_loader` | 8 | True | 默认 | 0(默认)| 无 |
| `test_loader_seen`(Test 1)| 1 | False | 默认 | 0 | 无 |
| `test_loader_unseen`(Test 2)| 1 | False | 默认 | 0 | 无 |

- **无自定义 `collate_fn`**:样本定长,PyTorch 默认 `collate` 直接堆叠为 `[B, 48000]` / `[B, 1967]`,无需 padding/分桶。
- 未设 `num_workers`/`pin_memory`/`drop_last`(取默认)。

### 4.6 次级路径物理卷积 `apply_dynamic_path`(`dataset.py:13-38`)
把网络输出的反噪声 `y_t` 经过次级路径,得到误差麦克风处的抵消信号 `a_t`:
```python
def apply_dynamic_path(signal_batch, path_batch):   # [B,T], [B,L] -> [B,T]
    B, T = signal_batch.shape; L = path_batch.shape[1]
    signal_reshaped = signal_batch.view(1, B, T)
    path_flipped = torch.flip(path_batch, dims=[1])  # ① 时域翻转
    path_reshaped = path_flipped.view(B, 1, L)
    signal_padded = F.pad(signal_reshaped, (L-1, 0)) # ② 左 pad 保证因果
    output = F.conv1d(signal_padded, path_reshaped, groups=B)  # ③ 分组卷积
    return output.squeeze(0)
```
三个设计要点:
1. **时域翻转**:`F.conv1d` 底层是互相关,翻转冲激响应后才等价于真实物理**卷积**(前向因果传播);
2. **左 pad `L−1`**:只用过去样本,保证因果、输出长度仍为 `T`;
3. **`groups=B`**:让**每个样本走自己的那条路径**(batch 内每个样本的 `sh` 不同)。

---

## 5. 模型架构 `TimeDomainANC`(`model.py`)

纯时域端到端 1D-CNN,`forward: [B, T] → [B, T]`,直接在原始波形上逐点处理,**无 STFT 分帧、无帧缓冲延迟**,面向实时低延迟 ANC。

### 5.1 逐层结构与张量流转(`model.py:60-93`)

| 阶段 | 模块 | 形状变化 |
|---|---|---|
| 输入对齐 | `unsqueeze(1)`(若输入 2D)| `[B, 48000]` → `[B, 1, 48000]` |
| 输入投影 | `input_conv`(Conv1d 1×1,1→32)| `[B, 1, T]` → `[B, 32, T]` |
| 主干 | **10 × `ResidualBlock`**(dilation = 2⁰…2⁹)| `[B, 32, T]`(保持)|
| 输出投影 1 | `output_conv1`(1×1,32→32)+ `PReLU` | `[B, 32, T]` |
| 输出投影 2 | `output_conv2`(1×1,32→1)| `[B, 32, T]` → `[B, 1, T]` |
| 输出 | `squeeze(1)` | `[B, 1, T]` → `[B, 48000]` |

超参(`train.py:156`):`in_channels=1, out_channels=1, hidden_channels=32, num_layers=10`。

### 5.2 残差块 `ResidualBlock`(`model.py:33-50`)
`out = CausalDilatedConv1d(x) → Conv1d(1×1) → + x`(残差跳连促进梯度流,支撑深层堆叠)。

### 5.3 因果空洞卷积 `CausalDilatedConv1d`(`model.py:4-31`)
```python
self.padding = (kernel_size - 1) * dilation        # 对称 pad 量
x = self.conv(x)
if self.padding > 0:
    x = x[:, :, :-self.padding]                    # 裁掉右端 → 严格因果
return self.prelu(x)
```
- `kernel_size=3`,**膨胀率指数递增** `2⁰, 2¹, …, 2⁹ = 1,2,4,…,512`;
- 通过"pad 后裁右端"保证**因果性**(不泄露未来样本);
- 激活 `PReLU`。

**感受野**:$RF = 1 + \sum_{i=0}^{9}(k-1)\cdot 2^{i} = 1 + 2\,(2^{10}-1) = \mathbf{2047}$ 个样本 ≈ **42.6 ms** @48 kHz。这与次级路径长度(1967 ≈ 41 ms)同量级,足以覆盖低频噪声的长时相关与路径时延。

### 5.4 模型复杂度(参数量 / FLOPs)

> 实测自 `model.py` 默认超参(`hidden_channels=32, num_layers=10, kernel_size=3`),对照 §7.4 的复杂度计分档位。

**参数量:`sum(p.numel())` = 42,764 ≈ 0.0428 M**(逐模块):

| 模块 | 参数量 |
|---|---|
| `input_conv`(1→32, k1)| 64 |
| `res_blocks`(10 × `ResidualBlock`)| 41,610(每块 4,161 = 空洞卷积 3,105 + 1×1 卷积 1,056)|
| `output_conv1`(32→32, k1)| 1,056 |
| `prelu` | 1 |
| `output_conv2`(32→1, k1)| 33 |
| **合计** | **42,764(≈ 0.043 M)** |

→ 远 < 5M,落在**参数量满分档(10 分)**。

**FLOPs(以 1 秒 = 48000 样本为单位)**:卷积 MACs = `C_in × C_out × K × T`,逐层累加 ≈ **2018 MMac**(主要来自 10 层空洞卷积 ~1475 MMac + 10 层 1×1 卷积 ~492 MMac);若按 1 MAC = 2 FLOP 计 ≈ **4037 MFlops**;摊到每样本 ≈ 42k MACs/sample。

> ⚠️ **口径说明**:FLOPs 数值随**计数约定**变化很大——是否把 1 MAC 记成 2 FLOP、以及按「每样本 / 每秒 / 整段测试音频」哪种为单位,可相差一个数量级。上面按「每 1 秒片段、MAC 口径」给出 ~2018 MMac。无论哪种约定,本基线都落在 **2000–5000 MFlops 档(6 分)**、**并非满分**——因为它要对 48000 个样本逐点跑 32 通道 × 10 层卷积。提交前应以官方指定口径复核。

**结论**:**参数量是基线天然优势**(0.043M ≪ 5M);**FLOPs 才是主要可优化方向**——降通道数 / 减层数 / 改用更高效结构(分组或深度可分离卷积、下采样-上采样)都能直接抬高复杂度得分,且呼应官方「轻量级 + 离线推断」要求(§7.4)。

---

## 6. Loss 设计(`train.py:171-178`)

**单一损失:残余噪声能量的均方误差(MSE)**。
```python
y_t = model(x_t)                    # 反噪声
a_t = apply_dynamic_path(y_t, sh)   # 经次级路径
e_t = d_t - a_t                     # 残余误差
loss = torch.mean(e_t ** 2)         # MSE
```

$$\mathcal{L} = \frac{1}{N}\sum_{n=1}^{N} e(n)^2 = \frac{1}{N}\sum_{n=1}^{N}\big[d(n) - (y(n) * S(z))\big]^2$$

- 物理意义:**端到端最小化误差麦克风处的残余噪声能量**,直接对应 ANC 系统的优化目标。
- 无多任务、无加权项、无频域/感知项(可改进方向见 §9)。

---

## 7. 评分脚本与评估指标(`train.py:12-111`)

评分函数 `evaluate_and_plot`,**唯一指标:降噪量 Noise Reduction(NR, dB)**,纯 NumPy 自实现(无 PESQ/STOI/SI-SDR 等)。

### 7.1 NR 计算(`train.py:42-44`)
```python
energy_d = np.sum(d_np ** 2)
energy_e = np.sum(e_np ** 2)
nr_db = 10 * np.log10(energy_d / (energy_e + 1e-12))
```

$$NR = 10\log_{10}\!\left(\frac{\sum_n d(n)^2}{\sum_n e(n)^2 + \epsilon}\right)\ \text{(dB)}$$

- **NR > 0**:降噪成功,**越大越好**;NR < 0 表示网络反而放大了噪声。
- 逐场景打印 NR,并输出平均 NR(`train.py:47-49`)。

### 7.2 可视化(每个测试场景两图)

| 图 | 内容 | 文件名 |
|---|---|---|
| 时域波形 | `d(t)` vs `e(t)` 对比(取前 `plot_duration=0.9` s = **900 ms**)| `anc_{prefix}_time_result.png` |
| 频域 PSD | 降噪前后功率谱,对数频轴 20 Hz–24 kHz,NFFT=1024 | `anc_{prefix}_freq_result.png` |

`prefix` 为 `seen_paths`(Test 1)/ `unseen_paths`(Test 2),故共生成 4 张 PNG。

> 注:时域子图标题里写的是 "First 50ms",但实际绘制的是 900 ms(见 §9 文案瑕疵)。

### 7.3 基线性能参考(本地 60 epoch 实跑)

> 下列为在本机(`cuda`)按基线默认配置(已含 2026-06-27 因果性修正)训练 **60 epoch** 的实跑结果,并列 README 组委会参考值作对照。
> 数据源:`results_causal_60ep/train_log_causal_60ep.txt`;图不在文中嵌入,见该目录下 `anc_{seen,unseen}_paths_{time,freq}_result.png`。

| 运行 | 训练噪声数 | 训练末段 loss | Test 1 平均 NR | Test 2 平均 NR |
|---|---|---|---|---|
| 本地实跑(`results_causal_60ep`)| 6 | ~`5e-5` | **5.74 dB** | **3.71 dB** |
| README 组委会参考 | — | — | 5.53 dB | 3.93 dB |

逐场景 NR(dB):
- Test 1(8 场景):`7.94, 8.11, 7.87, 5.84, 5.47, 4.28, 0.36, 6.04`
- Test 2(2 场景):`6.33, 1.09`

**三点洞察:**
1. **低 loss ≠ 高 NR / 强泛化**:训练末段 loss 已收敛到 ~`5e-5`(很小),但平均 NR 仅 ~5.7 dB,且换到未见路径(Test 2)进一步降到 3.71 dB。说明 MSE 训练 loss 与最终降噪量/泛化并非线性挂钩——优化时不能只盯训练 loss(也呼应 §7.4:训练 loss 与官方评分口径并不一致)。
2. **存在「盲区」场景**:Test 1 里有一个场景仅 0.36 dB,与 README 报告的 0.43 dB 一致。说明基础因果卷积对某类突变/相位跳跃存在固定跟踪盲区,是明确的改进靶点。
3. **跨未见路径衰减(核心难点)**:从 Test 1 到 Test 2(换到未见物理路径)平均 NR 明显下降(5.74 → 3.71 dB)。路径泛化是本赛题最大挑战,呼应 §3.2 的路径划分设计。

### 7.4 官方评分规则(与代码内 NR 的差异)

> 来源:`docs/official/CCF-赛题2-主动降噪.md`。**代码里的 NR 只是本地自测代理,并非官方评分口径**,二者差异很大,优化时务必对齐官方指标。

**总分构成(满分 100):**

| 维度 | 指标 | 口径 / 档位 | 分值 |
|---|---|---|---|
| 客观 · 降噪 | ANC 降噪量 | **50 Hz–5 kHz** 的 **1/3 倍频程**统计,占客观 70% | 28 |
| 客观 · 反弹 | ANC **反弹**(rebound)| **1 k–8 kHz** 的 **1/3 倍频程**统计,占客观 30% | 12 |
| 客观 · 复杂度 | 参数量 + FLOPs | 见下方两张档位表 | 20 |
| 主观 · 创新性 | ANC 架构 / 自适应 / 滤波器结构等 | 原创进展 20–30;基于现有方法优化 0–20 | 30 |
| 主观 · 听感 | Overall MOS | 四级:1 级 10 / 2 级 8 / 3 级 6 / 4 级 4 | 10 |

**复杂度档位(共 20 分,参数量与 FLOPs 各成一档):**

| 参数量(M)| 分 | 复杂度(MFlops)| 分 |
|---|---|---|---|
| < 5M | 10 | < 500 | 10 |
| 5–10M | 8 | 500–2000 | 8 |
| 10–15M | 6 | 2000–5000 | 6 |
| 15–20M | 5 | 5000–10000 | 5 |
| > 20M | 4 | > 10000 | 4 |

**硬约束**:模型必须**离线推断**——实际运行时**不允许在线训练或在线参数更新**;并要求**轻量级**(直接体现在上面的复杂度计分)。

**与代码内 NR 的关键差异**(优化时极易踩坑):

| 维度 | 代码内 `evaluate_and_plot` | 官方评分 |
|---|---|---|
| 频带 | **全频带** `Σd²/Σe²`(整段能量比)| **分 1/3 倍频程**(50 Hz–5 kHz)|
| 反弹 | 无 | **单列 1 k–8 kHz 反弹**(高频被放大要扣分)|
| 复杂度 | 不计 | 参数量 + FLOPs **直接占 20 分** |
| 听感 | 无 | Overall **MOS** 10 分 |

> 含义:本地 NR 高 ≠ 官方分高。① 官方按 1/3 倍频程加权,**低频(<5 kHz)降噪权重大**;② **高频反弹是独立扣分项**——网络若在 1 k–8 kHz 放大了噪声,即使全频带 NR 为正也会失分(对应 §7.1 里 NR<0 的局部放大);③ 复杂度直接计分,详见 §5.4。

---

## 8. 训练与推理流程

### 8.1 超参与配置

| 项 | 值 | 位置 |
|---|---|---|
| 设备 | 自动 `cuda` / `cpu` | `train.py:115` |
| 优化器 | `Adam(lr=0.001, amsgrad=True)` | `train.py:157` |
| 学习率调度 | **无** | — |
| Epochs | **20**(代码默认;README 基线按 60)| `train.py:158` |
| 采样率 | 48 kHz,片段 1 s | `train.py:147-149` |
| checkpoint | **无保存** | — |

### 8.2 训练循环(`train.py:161-184`)
```text
for epoch in range(epochs):
    for x_t, sh, d_t in train_loader:        # 每 epoch 仅 1 个 batch(见 §4.2)
        y_t = model(x_t)                     # 前向
        a_t = apply_dynamic_path(y_t, sh)    # 过次级路径
        e_t = d_t - a_t
        loss = mean(e_t**2); loss.backward(); optimizer.step()
    print(epoch 平均 loss)
```

### 8.3 推理 / 评估(`evaluate_and_plot:23-45`)
推理 forward 与训练**完全一致**,只是包在 `torch.no_grad()` 里、`batch_size=1` 逐场景跑,并计算 NR、画图。训练结束后依次评估 Test 1、Test 2(`train.py:187-195`)。

### 8.4 运行方式
```bash
pip install torch torchvision torchaudio numpy matplotlib scipy librosa soundfile
python train.py        # 一键训练 + 评估 + 出图
```
> 强依赖 `soundfile` 实现长音频指针级快速读取。

---

## 9. 已知设计权衡与可改进项

> 以下均为按当前源码核实的客观行为,供优化基线时参考。**按重要性排序,靠前者更直接影响得分与模型质量。**

1. **本地评估未对齐官方评分口径(最该先解决)**
   - `evaluate_and_plot` 只算**全频带** NR(`Σd²/Σe²`,`train.py:42-44`);而官方按 **50 Hz–5 kHz 的 1/3 倍频程**计降噪量(占客观 70%)、并单独统计 **1 k–8 kHz 的反弹**(占 30%)(详见 §7.4)。本地 NR 涨 ≠ 官方分涨,且**高频反弹完全没被监控**。
   - 建议:自建一个对齐官方口径的本地评分器(按 1/3 倍频程计降噪 + 反弹),用它做模型选择,并随时盯住 1 k–8 kHz 是否被放大。

2. **单一损失:未含反弹 / 分频带(训练侧的同一问题)**
   - 仅 MSE + 仅全频带 NR。可探索 SI-SDR、频域(STFT 幅度/复数谱)、感知损失;**关键是加抗反弹 / 分频带加权损失以对齐官方口径**(配合第 1 项的评分器),并辅以多指标(分频带 NR、收敛速度、突变后恢复时间)评估。

3. **每个 epoch 仅 8 个样本(数据曝光极低)**
   - `__len__` 返回 `len(path_indices)=8`,配合 `batch_size=8` ⇒ 每 epoch 仅 1 个 batch。这是严重欠训练、对性能影响最大的工程项。
   - 建议:把 `__len__` 改为可配置的 `steps_per_epoch`(如几千),`__getitem__` 内随机选路径,大幅提升数据多样性与训练充分度。

4. **测试集过小、评估高方差**
   - Test 1 仅 8 个场景、Test 2 仅 2 个场景、未见路径也只有 2 条(§3.3、§4.2)。2 个样本求平均极不稳,一个 1.09 dB 的坏场景就把 Test 2 均值拉低,容易误导模型选择。
   - 建议:每条路径多采几个起点以扩大场景数、多留几条未见路径,并报告标准差 / 区间。

5. **训练无随机种子 → 结果不可复现**
   - 选噪声 `random.choice`(`dataset.py:88`)、选起点 `np.random.randint`(`dataset.py:97`)、`main()` 未设 `torch`/`numpy`/`random` 种子。每次运行的数据采样与权重初始化都不同——这也是本地 5.74/3.71 与 README 5.53/3.93 略有出入的原因之一。
   - 建议:入口处统一设种子(必要时 `cudnn.deterministic`);报告结果时给多次平均更稳。

6. **固定学习率、无验证集、无早停**:`lr=0.001` 全程不变,直接用测试集观察,缺少验证/早停/调度。可加 `ReduceLROnPlateau`/`CosineAnnealing` 与独立验证划分。

7. **测试「环境突变」场景单一**
   - 测试分支恒为 `车载`(前半秒)+ `餐厅`(后半秒)、确定性起点(`dataset.py:102-131`,§4.3),只覆盖一种突变与一个方向,而赛题核心正是环境突变下的稳定性。
   - 建议:扩展为多对噪声、双向(A→B / B→A)、随机起点的拼接,更全面地考察突变跟踪与恢复。

8. **未保存模型 checkpoint(提交必需)**:训练脚本无 `torch.save`,跑完即丢、无法复用或提交权重。建议补 `torch.save(model.state_dict(), ...)` 及最优权重保存(初赛需提交可执行模型)。

9. **文案小瑕疵**:时域子图标题写 "First 50ms"(`train.py:69`),实际 `plot_duration=0.9` 绘制 900 ms;不影响功能。

---

## 10. 关键文件与代码位置速查

| 主题 | 位置 |
|---|---|
| 物理模型方程 | `README.md:29` |
| 噪声/路径划分 | `train.py:122-143` |
| 三个 Dataset / 三类 Loader | `train.py:147-153` |
| `apply_dynamic_path`(时域翻转+因果卷积)| `dataset.py:13-38` |
| `sh.npy` 装载与转置 | `dataset.py:60` |
| `__len__` | `dataset.py:65-66` |
| `_fast_read_slice`(soundfile 切片)| `dataset.py:68-77` |
| 训练随机采样 | `dataset.py:86-100` |
| 测试中点拼接 | `dataset.py:102-131` |
| ~~相对幅值归一化~~(已随官方 2026-06-27 修正移除)| 见 §4.4 历史说明 |
| 返回三元组 | `dataset.py:139-141` |
| 因果空洞卷积 | `model.py:4-31` |
| 残差块 | `model.py:33-50` |
| `TimeDomainANC.forward` | `model.py:78-93` |
| 模型复杂度(参数量/FLOPs)| §5.4(实测 0.043M / ~2018 MMac)|
| NR 计算 | `train.py:42-44` |
| 可视化(时域/频域)| `train.py:55-111` |
| 官方评分规则(1/3 倍频程 / 反弹 / 复杂度 / MOS)| §7.4 · `docs/official/CCF-赛题2-主动降噪.md` |
| 本地 60ep 实跑结果 | §7.3 · `results_causal_60ep/` |
| 训练循环 | `train.py:161-184` |
| Loss(MSE)| `train.py:177-178` |
| 优化器 / epochs | `train.py:157-158` |
| 入口 | `train.py:199-200` |
