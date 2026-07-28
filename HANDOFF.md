# CoDA 复现与消融 — 交接文档

> 面向不了解上下文的读者，内容自足。
> 时间：2026-07-25 ~ 07-27，单张 RTX 3090。
> 论文：CoDA: From Text-to-Image Diffusion Models to Training-Free Dataset Distillation (ICLR 2026)

---

## 0. TL;DR

复现目标是论文 Table 2 中 **ImageWoof / IPC=10 / ResNetAP-10 = 39.2±0.7** 这一格。

- 按仓库发布的超参原样跑，只能到 **35.0**，差 4.2 个点。
- 排查后发现真正的瓶颈**不在超参、不在数据集划分，而在聚类所用的距离度量**。
- 把 VAE latent 在聚类前做一次**空间平均池化**（3 行代码），再用最朴素的 KMeans 选点，
  G 达到 **39.0**，基本追平论文；代表样本集 R 从 29.4 提到 **35.6**。
- 论文主打的 Distribution Discovery（UMAP + HDBSCAN + 三重后处理）相对朴素 KMeans **没有优势**，
  但相对随机选点有巨大优势（35.4 vs 28.4）——选点很重要，但那套繁复机制不是收益来源。

---

## 1. 方法与背景

CoDA 用现成的 SDXL（不在目标数据集上预训练）做数据集蒸馏，两个阶段：

- **Distribution Discovery**：把目标数据集用 SDXL 的 VAE 编码到 latent，聚类找出每类的
  IPC 个"代表样本"，构成集合 **R**。
- **Distribution Alignment**：用 R 中的第 j 个样本引导 SDXL 生成第 j 张图，构成集合 **G**。
  G 是最终的蒸馏数据集。

论文对前人（D4M、MGD³）的批评：他们直接在 VAE latent 上跑 KMeans 并强令 k=IPC，
而 KMeans 假设簇是凸的且各向同性、对离群点敏感，会为离群点建立伪簇，
因此"找不到真正的高密度区域"。CoDA 改用 UMAP 降维 + HDBSCAN 密度聚类 + 三重后处理对齐 IPC。

评测：用蒸馏出的 10 类 × 10 张图训练 ResNetAP-10（256×256, 2000 epochs），在验证集上报 top-1。

---

## 2. 环境与数据

### 两份都叫 "ImageWoof" 的数据集（重要）

| | fastai imagewoof2 | ImageNet-1K 子集 |
|---|---|---|
| 出处 | Jeremy Howard 的 fastai 仓库，2019-12-06 起改为 70/30 划分 | IDC (ICML 2022) 起的蒸馏文献惯例：类别列表套完整 ImageNet |
| 训练 | 9,025 | 12,454 |
| 验证 | 3,929 | 500（50/类） |
| 本地 | `/media/lm/_dde_data/data/imagewoof` | `/media/lm/_dde_data/data/imagewoof_in1k`（软链构建） |

**两份的图片总数都是 12,954**：fastai 就是把这 10 类的 ImageNet train(12,454) + val(500)
合并后重切。证据：fastai 的 `train/` 里含 `ILSVRC2012_val_*.JPEG` 文件；
每类总数恰好 = 1300+50 = 1350（英国猎狐犬 754+50 = 804）。
副作用：fastai 的验证集里约 96% 是 ImageNet **训练集**的图。

ManifoldGD 引用的是 fastai 仓库。CoDA 的代码是 IDC 血统（`misc/class_woof.txt` + `find_subclasses`），
但两种目录结构它都能读。**论文未注明用哪一份**——这是复现的第一个不确定性，已排除（见 §5.4）。

### 模型

SDXL base 1.0 + refiner 1.0，fp16，本地 `model/SDXL-Refiner/`（12.4 GB），首次运行自动下载。
注意 `DF=1.0` 时 refiner 完全不参与。

### 机器

单张 RTX 3090（24 GB），16 核，62 GB 内存。

---

## 3. 流水线与耗时

| 阶段 | 做什么 | 耗时 |
|---|---|---|
| 1 特征提取 | 全部训练图缩到 1024²，过 SDXL VAE，latent 展平成 65536 维 | 38.5 min（每数据集一次，可缓存复用） |
| 2 选点 | 聚类得到每类 IPC 个代表样本 → **R** | 4.5 min（CPU） |
| 3 生成 | 用 R 引导 SDXL → **G** | 14.3 min（8.5 s/张） |
| 4 评测 | ResNetAP-10, 2000 epochs | 6 min/次 |

复用特征缓存后，一组完整消融约 **25 分钟**。

---

## 4. 完整实验结果

除注明外均为：ImageNet-1K 子集、官方 val 500 张、IPC=10、GTP=0.9、γ=0.05、
sample_step=25、DF=1.0、单卡单种子。

### 4.1 选点方式（引导设置固定）

| 选点方式 | 聚类所用特征 | 维度 | **R** | **G** |
|---|---|---|---|---|
| KMeans + 8×8 池化 | 4×16×16 | 1024 | 32.6 | **39.0** ★ |
| KMeans + 16×16 池化 | 4×8×8 | 256 | 32.8 | 38.4 |
| KMeans + 32×32 池化 | 4×4×4 | 64 | **35.6** ★ | 37.8 |
| KMeans（原始 latent） | 4×128×128 展平 | 65536 | 32.4 | 35.4 |
| **CoDA 原版**（UMAP+HDBSCAN+后处理, mcs=55） | 65536→UMAP 50 | 50 | 29.4 | 35.0 |
| CoDA, mcs=35 | | | 30.2 | 34.6 |
| CoDA, mcs=25 | | | 29.8 | 33.6 |
| KMeans + 通道统计量（均值+方差） | 每通道 2 个数 | 8 | 30.8 | 30.0 |
| 随机选 10 张 | — | — | 27.4 | 28.4 |
| 最远点采样 k-center greedy | 65536 | 65536 | 24.8 | 23.6 |
| CoDA 配额版（只用真实模式 M=3~5，有重复） | | | 21.0 | 24.4 |
| **无引导（γ=0）** | — | — | — | **32.4** |
| 论文 CoDA | | | | 39.2±0.7 |
| 论文 Random / Herding / DiT / IDC-1 / MGD³ | | | 29.4 / 32.0 / 34.7 / 39.1 / 40.4 | |
| 论文 Full（全量上限） | | | | 87.5 |

### 4.2 引导时间窗（选点固定为 CoDA mcs=55）

`guideTPercent`(GTP) 控制引导覆盖去噪轨迹的前多大比例（高噪声端）。

| GTP | 引导覆盖 | G |
|---|---|---|
| —（γ=0） | 0 步 | 32.4 |
| 0.2 | 前 5/25 步 | 32.6 |
| 0.5 | 前 12/25 步 | 34.4 |
| 0.9（脚本默认） | 前 22/25 步 | 35.0 |
| 1.0 | 前 24/25 步 | **36.2** |

单调上升。但与池化选点**不可加**：256 维池化下 GTP=0.9 得 38.4，GTP=1.0 反而降到 37.2。

### 4.3 数据集划分对照

| 划分 | R | G |
|---|---|---|
| ImageNet 官方（12,454 / 500） | 29.4 | 35.0 |
| fastai imagewoof2（9,025 / 3,929） | 27.9 | 30.85 |

（两个 G 在不同验证集上评测，不能直接比大小；能比的是"都到不了 39.2"。）

---

## 5. 主要发现

### 5.1 瓶颈是距离度量，不是聚类算法 ★核心结论

VAE latent 是 4×128×128 的**空间张量**，代码（`get_features.py:110`）把它展平成 65536 维再算欧氏距离。
展平后的 L2 是**逐格子对比**——A 图 (17,42) 位置的格子减 B 图 (17,42) 位置的格子。
它量的是**构图相似度**而非语义：同一只狗居中 vs 偏左，距离可能大于两个不同犬种；
两只不同的狗都居中配草地背景，距离反而很小。

三条独立证据：

1. **算法换遍了结果不动**：HDBSCAN 密度聚类、KMeans、三重后处理，选出的 R 全部落在
   29.4–32.4，即论文 Random(29.4) 到 Herding(32.0) 这个区间。
2. **HDBSCAN 把每类 28%~57%（平均约 45%）的样本判成噪声**——密度模型认为近一半数据
   不属于任何高密度区域，说明这个空间里没有清晰的密度结构可找。
   这也解释了 CoDA 的"在噪声点上跑 KMeans"策略为何 10/10 类都触发：噪声池永远"充足"。
3. **压掉空间维度后立刻见效**：8×8 平均池化（1024 维）+ 最朴素的 KMeans，G 从 35.4 → **39.0**。

修复只需 3 行（`select_variants.py` 的 `transform`）：

```
L = X.reshape(N, 4, 128, 128)
s = 128 // f                                   # f 是池化窗口
Z = L.reshape(N, 4, s, f, s, f).mean(axis=(3,5)).reshape(N, -1)
```

不引入任何外部模型，不违反 training-free 原则。

**池化粒度有甜蜜点**，而且 R 和 G 的最优点不同：

| 输出维度 | 65536 | 1024 | 256 | 64 | 8 |
|---|---|---|---|---|---|
| R | 32.4 | 32.6 | 32.8 | **35.6** | 30.8 |
| G | 35.4 | **39.0** | 38.4 | 37.8 | 30.0 |

压到 8 维（完全平移不变的通道统计量）信息毁太多，两个指标都崩。

### 5.2 CoDA 的 Discovery 模块相对朴素 KMeans 没有优势

保持生成端完全不变，只把选点换成 MGD³ 式的朴素 KMeans（无 UMAP、无 HDBSCAN、无后处理，
质心吸附到最近真实样本）：

- G 打平：35.4 vs 35.0（单种子下 0.4 无意义）
- R 高 3 个点：32.4 vs 29.4

顺带在数据上确认了论文对 KMeans 的批评属实——KMeans 确实产生只有 1~2 个点的伪簇
（例：`[376, 290, 247, 139, 129, 112, 2, 2, 2, 1]`）——但**这些伪簇非但没有伤害，
选出的样本反而更有用**。

**注意措辞**：不能说"Discovery 阶段没用"。加上随机基线后（见 5.3），选点这一步非常重要，
只是 CoDA 那套繁复机制不是收益的来源。

### 5.3 锚点质量决定引导是帮忙还是帮倒忙

```
最远点采样锚点   23.6   ← 比不引导差 8.8
随机锚点         28.4   ← 比不引导差 4.0
无引导 γ=0       32.4
CoDA 锚点        35.0
KMeans 锚点      35.4
池化 KMeans      39.0   ← 比不引导好 6.6
```

**锚点选得不好，引导会起反作用。** 这也说明 CoDA 的引导机制本身是有效的、能真正把目标信息
传进生成结果——否则锚点质量不该造成 15 个点的跨度。

### 5.4 数据集划分不是差距的原因

fastai imagewoof2 那份跑出来 30.85，比官方划分的 35.0 **更低**，两者都够不到 39.2。
这条线可以排除（fastai 那版验证集 3,929 张，单种子噪声小得多，30.85 其实比 35.0 更可信）。

### 5.5 引导目标的"数量"是主导因素

去掉 CoDA 的补齐逻辑、只用 HDBSCAN 找到的 M=3~5 个真实模式、按簇大小分配生成配额
（同一模式引导多张图，靠不同噪声种子拉开差异），G 从 35.0 **崩到 24.4**。

结论：

- G 对**用哪些** mode 不敏感（33.6 ~ 39.0，取决于度量）
- G 对**有几个不同的** mode 极其敏感（10 个 → 35.0，3~5 个 → 24.4）

推论：CoDA 的三重后处理不是可有可无的 hack，它是必需的——但真实作用比论文声称的朴素：
不是"找到内在核心分布"，而是"凑够足够多的互不相同的锚点"。

---

## 6. 论文与代码的不一致（两处）

### 6.1 引导项多了一个 σ_t

论文 §3.2 推导 `Δẑ₀ = γ(s_j − ẑ₀)`，转到噪声空间应为

```
Δε_t = − γ · sqrt( ᾱ_t / (1−ᾱ_t) ) · ( s_j − ẑ₀ )
```

而 `CoDA_SDXLBasePipeline.py:265` 实际是

```
pix_guide_mark = pix_diff * γ * self.scheduler.sigmas[i]      # 多出来的 σ_t
Δε_t           = − sqrt( ᾱ_t/(1−ᾱ_t) ) * pix_guide_mark
```

Karras 调度下 σ_t 从约 14 单调降到 0，等于**给引导强度加了一条随时间退火的曲线**，
论文里没有这个设计。

### 6.2 `guideTPercent` 的语义

`CoDA_main.py` 注释写的是
`PIS = sample_step × DF × (1−GTP) + sample_step × (1−DF)`，
但代码（`CoDA_SDXLBasePipeline.py:218,244`）是
`stop_idx = int(len(timesteps) × GTP)`，引导在 `t > timesteps[stop_idx]` 时生效。
timesteps 降序，所以 **GTP=0.9 表示引导开在前 22/25 步，最后 3 步关闭**——
GTP 是"引导覆盖比例"，不是"留给尾部的步数"。

---

## 7. 已被证伪的假设（避免重走）

| 假设 | 做法 | 结果 |
|---|---|---|
| `min_cluster_size` 过大导致兜底过多，是性能瓶颈 | 扫 55/35/25 | 兜底从 10/10 类降到 6/10 类，G 反而从 35.0 降到 33.6。**证伪** |
| 论文用的是 fastai 划分，基准不对齐 | 完整跑 fastai 那份 | 30.85，更低。**证伪** |
| 低 IPC 下蒸馏集要的是覆盖而非代表性 | k-center greedy 最远点采样 | 23.6，所有方法里最差。**证伪** |
| 解耦 M 与 IPC，不编造模式 | 按簇大小分配配额 | 24.4。**证伪**，但揭示了 5.5 |
| γ=0.05 太弱，引导是空转 | Idea 3 + γ=0 对照 | 锚点质量造成 15 点跨度，引导显然有效。**证伪** |

---

## 8. 未验证：Idea 4 — 引导目标沿时间轴由粗到细

### 动机

代码已经让引导**强度**随 t 退火（§6.1 的 σ_t），但**目标 s_j 全程是同一张具体图片的完整 latent**。

在高噪声步，模型只能分辨 σ_t 尺度以上的结构（扩散本来就是先定构图、后填细节）。
此时把 ẑ₀ 拉向一张具体图片的全部细节，超出当前可分辨尺度的那部分要么被后续步骤洗掉（浪费），
要么过早锁死轨迹（伤多样性）。作者用 σ_t 打折的是**幅度**，没动**目标粒度**。

### 改法

把 `s_j` 换成 `s_j(t)`，其余不变：

```
现状:   所有 t ────► s_j (一张具体图片，恒定)
Idea4:  t 大   ────► 粗粒度目标（类均值 / 父节点）
        t 中   ────► x_j 邻域的局部均值
        t 小   ────► x_j
```

三种实例化：

**(a) 硬切换**（需层级树，多一个超参 τ）

```
s_j(t) = μ_p(j)   if t > τ
       = x_j      otherwise
```

**(b) 线性插值**（无额外超参）

```
s_j(t) = λ_t · μ_p(j) + (1 − λ_t) · x_j ,    λ_t = sqrt(1 − ᾱ_t)
```

**(c) 核平滑**（最有依据，不需要树）

```
            Σ_k  x_k · w_t(k)
s_j(t) = ───────────────────────      k 遍历该类全部真实样本
              Σ_k  w_t(k)

w_t(k) = exp( − ‖x_k − x_j‖² / (2 h_t²) )
h_t²   = (1 − ᾱ_t) / ᾱ_t              带宽由调度器给定，不是新超参
```

**(c) 的理论依据**：前向过程在 t 时刻把数据分布卷积成 `p_t = p_data ∗ N(0, h_t² I)`。
对**经验分布**而言，t 时刻的最优去噪器恰好就是这个核加权均值（Tweedie / Nadaraya-Watson）。
所以 (c) 等价于：**把引导目标对齐到当前噪声水平下的最优经验去噪目标**，而非一个与 t 无关的固定样本。
极限行为天然 coarse-to-fine：t→T 时 h_t→∞ 趋近类均值；t→0 时 h_t→0 趋近 x_j。

**风险**：(c) 让引导逼近经验 score，理论上会推向记忆训练样本。
需检查生成图到最近训练图的 latent 距离分布是否明显左移。

### 前提已验证

GTP 扫描（§4.2）显示 32.4 → 32.6 → 34.4 → 35.0 → 36.2 单调上升，时间轴上确实有信号。

### 最有说服力的验证方案

用 (c) 重跑 §5.5 的配额版（M=3~5 个真实模式，10 张图）。
若能把 **24.4 拉回 34+**，就同时证明"沿轨迹分化能恢复多样性"与"不必编造模式"。

### 实现位置

`CoDA_SDXLBasePipeline.py:244-272` 的循环内，把 `represent_latent` 换成随 t 计算的目标；
`generated.py:85` 处需要多传粗粒度目标或整个类的 latent 矩阵。

---

## 9. 下一步建议（按优先级）

1. **补齐种子**。当前全部是单卡单种子，`±0.00` 不是真方差。用 `--repeat 3` 复跑
   {CoDA 原版, KMeans, 池化 1024 维} 三组，确认 39.0 vs 35.0 的差距不是噪声。
   成本：3 组 × 18 min。**这是所有结论的前提，应最先做。**
2. **池化 × GTP 的交互**。两个增益不可加（256 维下 GTP=1.0 反而更差），需要二维小网格
   （池化 ∈ {1024, 256}，GTP ∈ {0.9, 1.0}）。成本 4 组 × 20 min。
3. **换数据集验证泛化**。目前只在 woof 上验证了池化的效果，应在 nette 或 imageA 上复现。
   需重跑特征提取（38.5 min）+ 一组消融。
4. **Idea 4**（§8）。锚点这一侧已接近论文水平（39.0 vs 39.2），下一个增量更可能在引导机制。
5. **修 `get_features.py` 的缓存膨胀**（§11），跑 ImageNet-1K 时这是硬约束。

---

## 10. 复现方式

### 脚本清单

| 文件 | 用途 |
|---|---|
| `scripts/CoDA_woof.sh` | 主运行脚本；阶段开关与超参走环境变量 |
| `scripts/CoDA_woof_fastai.sh` | 同上，数据指向 fastai 那份 |
| `select_variants.py` | 池化 / 最远点采样等选点变体 ★核心 |
| `kmeans_baseline.py` | MGD³ 式朴素 KMeans 选点 |
| `random_baseline.py` | 随机选点基线 |
| `quota_baseline.py` | 配额版（不补齐模式） |

`scripts/CoDA_woof.sh` 的环境变量：
`STEP1 / FEATURES / CLUSTER / GENERATE / STEP2 / REAL_IMAGES`、`IPC / N_NEIGHBORS / SIZE_MIN`。

### 分阶段跑

```bash
cd /media/lm/_dde_data/code/CoDA
# 阶段 1：特征提取（每数据集只需一次）
STEP1=true FEATURES=true CLUSTER=false GENERATE=false STEP2=false bash scripts/CoDA_woof.sh
# 阶段 2：CoDA 原版聚类
SIZE_MIN=55 STEP1=true FEATURES=false CLUSTER=true GENERATE=false STEP2=false bash scripts/CoDA_woof.sh
# 阶段 3/4：生成 + 评测
SIZE_MIN=55 STEP1=true FEATURES=false CLUSTER=false GENERATE=true STEP2=false bash scripts/CoDA_woof.sh
SIZE_MIN=55 STEP1=false STEP2=true REAL_IMAGES=false bash scripts/CoDA_woof.sh   # 评 G
SIZE_MIN=55 STEP1=false STEP2=true REAL_IMAGES=true  bash scripts/CoDA_woof.sh   # 评 R
```

### 跑池化变体（复现 39.0）

```bash
python select_variants.py --cluster_dir results/clusterfile/woof_in1k \
  --out_root results/woof_in1k --method kmeans --pool p8 --path_tag n_4_s_0 --nclass 10 --IPC 10
python CoDA_main.py --program_path _in1k_run --dataset_dir /media/lm/_dde_data/data/imagewoof_in1k \
  --local_model_path model/SDXL-Refiner --spec woof --IPC 10 \
  --n_neighbors 4 --min_cluster_size 0 \
  --sample_step 25 --denoising_factor 1.0 --guideTPercent 0.9 --CoDA_guidance_scale 0.05 --generate_images
PYTHONPATH=$PWD python ./test/train.py \
  --dataset_dir results/woof_in1k/Step-25/IPC-10/DF-1.0-GTP-0.9-gamma-0.05/n_4_s_0/generated_images \
                /media/lm/_dde_data/data/imagewoof_in1k/validation \
  -d imagenet --spec woof --nclass 10 --size 256 --ipc 10 -n resnet_ap --depth 10 \
  --save-dir trained_results/woof_in1k/gen/n_4_s_0-resnet_ap --workers 8 \
  --n_neighbors 85 --min_cluster_size 55 --tag test
```

### 路径标记对照

结果目录由 `n_{n_neighbors}_s_{min_cluster_size}` 拼成，被复用为实验标记：

| 标记 | 含义 |
|---|---|
| `n_85_s_55/35/25` | CoDA 原版，不同 `min_cluster_size` |
| `n_85_s_155` | 配额版（不补齐） |
| `n_0_s_0` | 朴素 KMeans（原始 latent） |
| `n_0_s_1` | 随机选点 |
| `n_1_s_0` / `n_4_s_0` / `n_5_s_0` | KMeans + 16×16 / 8×8 / 32×32 池化 |
| `n_2_s_0` | KMeans + 通道统计量（8 维） |
| `n_3_s_0` | 最远点采样 |

ImageNet-1K 那一轮的结果归档在 `results/woof_in1k`、`trained_results/woof_in1k`、
`results/clusterfile/woof_in1k`；`_in1k_run/` 是一组软链，让 `--program_path` 能指回这些目录。

---

## 11. 必须知道的 caveat

- **全部结果都是单卡单种子**。`test/train.py:102` 按 GPU 数起进程、每卡一个种子，
  单卡 + `repeat=1` 只有一个样本，输出里的 `±0.00` **不是真方差**。
  1~2 个点的差异不应当作结论，但 39.0 vs 35.0（4 个点）和 23.6 vs 39.0（15 个点）应该是真的。
- **特征缓存膨胀 8 倍**：`get_features.py:111-118` 存的是整个 batch 张量的**视图**，
  pickle 会连底层 storage 一起存，`batch_size=8` 正好放大 8 倍（13 GB vs 理论 1.6 GB）。
  改成 `.clone()` 即可。跑 ImageNet-1K 时这是硬约束：1.28M 张 × 1 MB = 1.3 TB。
- **超参来自 ImageNet-A**：`n_neighbors=85, min_cluster_size=55` 等是仓库给 ImageNet-A 的默认值，
  作者未公布 woof 专用配置。已就此向作者发信询问，尚未收到回复。
- **孤儿进程**：kill 驱动脚本不会杀掉已派生的生成子进程，它会继续占 19 GB 显存导致连锁 OOM。
  后续脚本已加显存守卫。
- 论文 Table 2 注明 "All the results are reproduced by us"，前人结果引自 MGD³。
  CoDA 在 IPC=10/20 落后于当时 SOTA，IPC≥50 才反超。
