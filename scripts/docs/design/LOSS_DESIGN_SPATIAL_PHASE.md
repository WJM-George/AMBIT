# VAE 相位 + 空间 loss 设计稿 (v2, 已定稿并实现)

> v2 变更: (1) novelty 检索完成, SCM 拆分加权版确认为我们的 idea, 已实现
> (`FOASpatialCovarianceLoss`, `FrequencyGatedIFGDPhaseLoss`, 见 §0.7);
> (2) ablation 定为两波 4+4 GPU, 从 800k ckpt resume (§5);
> (3) (C) band 曲线本轮不上, 只进论文公式 (§7)。

> 目标: 在现有 4ch FOA VAE (ds1024_z64, 正在跑的 hf_overshoot_decay 实验) 基础上,
> 加 (1) 频率加权的相位 loss、(2) FOA 空间一致性 loss、(3) 把高频 band 权重改成参数化曲线,
> resume 续训少量 steps 作为 ablation。参考: Stable Audio 3 / SAME、Qwen-Music、εar-VAE。

![weight curves](loss_design_curves.png)

---

## 0. 三篇参考到底做了什么 (精确版)

| | SAME (SA3 的 AE) | Qwen-Music Spec-VAE | εar-VAE (两家共同引用) |
|---|---|---|---|
| 谱 loss | 7 分辨率 MRSTFT (32..2048, 75% overlap), K-weighting 预滤波, spectral contrast + 自适应 log-mag | MRSTFT + SpectroStream mixed-scale + SAME 的自适应 log-mag + K-weighting | EnCodec 式多尺度 log-mag |
| 相位 loss | IFGD: 相邻帧/相邻 bin 的**单位相量** (免 unwrap), 余弦距离, 能量加权 + 归一化复距离项; 权重与谱项等权, **无频率依赖** | 直接用 εar 的 IF/GD (λ=0.1, 且只在 Stage-3 refiner 阶段开) | IF/GD 有限差分 L1 (mod 2π) + Correlation Loss (逐 bin 相位差余弦) |
| 空间处理 | mid/side + left/right 各算一遍 MRSTFT; 另有 latent 上的 ILD 线性回归头 | MSLR 分解 (抄 εar); Band-Mode Refiner **架构**: 低频只修相位 / 中频都修 / 高频只修幅度 | 幅度用 MSLR 四路监督, **相位只用 LR** (M/S 相位会破坏 IPD, 有 ablation 证据) |
| 频率分 band | 无 (K-weighting 即全部) | 分 band 做在 refiner 架构里, 不在 loss 里 | 无 |

对我们的直接启示:
1. 相位 loss 用 **SAME 的相量形式** (免 unwrap, 我们代码已有), 不用 εar 的 mod-2π 差分。
2. εar 的结论"相位只在物理通道上监督"移植到 FOA: 相位/空间项在 W/Y/Z/X 原始通道上算, 不做虚拟 downmix。
3. 谁都没做频率依赖的 loss 权重曲线 (Qwen 只做在架构里) → 这是我们的差异点之一。
4. 谁都没做 FOA; 方向/空间 loss 是我们的差异点之二。

## 0.5 我们代码库现状 (fork 已含 SA3 更新)

- `losses/auraloss.py` 已有: `SpectralContrastLoss`、自适应 log-mag (`log(x/σ+1)`)、
  `if_gd_loss` + `normalized_complex_distance_loss` (= SAME 的 L_IF+L_GD+L_cd, 走 `w_phs`,
  **目前没有任何 config 启用**)、K-weighting FIR (`perceptual_weighting: true`, 已启用)。
- 4ch 路径: `sdstft = MultiResolutionSTFTLoss` 直接把 4 通道 fold 进 batch → 开 `w_phs`
  即是"逐通道相位 loss", 正好符合 εar 的 LR-only 原则 (FOA 版: 物理通道 only)。
- 已有 `FOASpatialConsistencyLoss` (DirAC intensity 方向余弦 + dir/omni ratio, 未在当前实验启用),
  `BandWeightedMelSpectrogramLoss` (5 个离散 band, 在跑), `HighFrequencySpectralOvershootLoss`
  (在跑, 350k→650k cosine decay 0.2→0.05), per-loss schedule + resume 时 loss-state 对账机制都已就绪。

## 0.6 SCM loss 的 novelty 检索 (2026-07-19)

结论: **时频复数 SCM 匹配 + 极分解 + 感知频率门控没有先例, 是我们的 idea**。必须引用的近邻:

| 工作 | 做了什么 | 和我们的差别 |
|---|---|---|
| Hirvonen & Namazi, Samsung, "Compression of HOA with Multichannel RVQGAN" (arXiv:2411.12008, ICASSP'25) | 16ch HOA codec + covariance loss: **时域宽带**实值 Pearson 相关矩阵的 L1, 权重 1.0 | 无时频分辨率、无相位/IPD、无能量/感知加权; 原文明确把 frequency-dependent 版本留作未做项 ("We evaluate the covariance broadband, but it is also possible to have the measure be frequency-dependent") |
| Qiao et al., Tencent, "Neural Ambisonic Encoding" (arXiv:2409.06954) | 麦阵→SOA 编码; SCM 只作**输入特征**; loss 是 beamformed spatial power map 的 KL | loss 不在 SCM 上, 只约束功率分布 (实部投影), 无 IPD 项; 任务不是 codec/VAE 重建 |
| Xu et al., "SpatialCodec" (arXiv:2309.07432) | 多通道语音 codec, SCM 作**输入特征**, 输出 complex ratio filter | loss 在信号/CRF 上, 非 SCM 匹配 |
| FOA Tokenizer (arXiv:2510.22241) | intensity 方向余弦 + (1-diffuseness) 加权 | = 我们 SCM 第一行实部的特例 (repo 已有移植, 即 `foa_spatial`) |
| Tokala et al. (binaural 语音增强) / spatial-loss 综述 (arXiv:2506.19404) | ILD+IPD 单 cue 保持项; 综述确认全场无 TF-SCM 训练 loss | 单 cue、stereo、增强任务; 综述里最接近的 BINAQUAL 只是 metric 非 loss |

论文定位一句话: 把 Hirvonen 的宽带实值 covariance loss 推广到时频复数域,
用极分解把 level/coherence/IPD 三种感知量解耦, IPD 项用 duplex-theory 门控
λ_IPD(f) 和参考相干度软掩码加权; intensity-vector loss 是其严格特例。

## 0.7 实现与验证状态 (已完成)

- `losses/semantic.py`: 新增 `_logistic_frequency_gate`、`FrequencyGatedIFGDPhaseLoss`
  (A)、`FOASpatialCovarianceLoss` (B_scm)。
- `training/autoencoders.py`: 新增 `phase_ifgd` / `foa_scm` config 块 (schedule、
  per-step ramp、wandb 子项日志、resume 对账前缀均已接)。
- 单测 `scripts/vae/loss/test_new_spatial_losses.py` 全过: identity≈0、梯度有限、
  **rot90(FOA 旋转) 使 SCM loss 3 倍于噪声基线**、单通道 3-sample 延迟
  IF/GD 不响应 (设计如此, 相位导数对常数延迟不变) 而 SCM-IPD 项正确捕捉、
  相位抖动由 GD/CD 项捕捉。
- 权重标定 `scripts/vae/loss/calibrate_new_loss_weights.py` (800k EMA ckpt + 6 条真实
  FOA, CPU): mrstft=0.255, phase_ifgd=0.263, foa_scm=1.350, foa_spatial=0.272
  → 按 ~8% 贡献定权: **phase 0.08, scm 0.015, intensity 0.075**。
- 两个 arm config 已生成并通过"真 ckpt resume + 调度 + 前向"端到端验证:
  `.../autoencoders/ablation_arms/stable_audio_4ch_vae_ds1024_z64_phase_scm.json`
  `.../autoencoders/stable_audio_4ch_vae_ds1024_z64_phase_int.json` (已弃用)

---

## 1. 设计总览

在总 loss 里新增/修改三处 (其余 GAN/KL/overshoot 不动):

\[
\mathcal{L} = \underbrace{\mathcal{L}_{\text{MRSTFT}}^{K\text{-w}}}_{\text{已有}}
+ w_{\phi}\,\underbrace{\mathcal{L}_{\text{IFGD}}^{\lambda(f)}}_{\text{(A) 新: 频率加权相位}}
+ w_{\text{sp}}\,\underbrace{\mathcal{L}_{\text{SCM}}}_{\text{(B) 新: FOA 空间协方差}}
+ w_{\text{bm}}\,\underbrace{\mathcal{L}_{\text{bandmel}}^{w_{\text{mag}}(f)}}_{\text{(C) 权重改为曲线}}
+ \text{GAN} + \text{KL} + \text{overshoot}
\]

三处共用同一个**门控曲线族** (log-频率上的 logistic, 也是整个设计的"一个公式讲完"卖点):

\[
g(f;\,f_c,\beta,\rho) \;=\; \rho + \frac{1-\rho}{1+(f/f_c)^{\beta}}
\qquad\in[\rho,\,1]
\]

单调、有界、光滑、三参数各有物理含义 (转折频率 / 陡度 / 高频保底), 不同用途只换 \((f_c,\beta,\rho)\)。

---

## 2. (A) 频率加权相位 loss — \(\mathcal{L}_{\text{IFGD}}^{\lambda(f)}\)

沿用 SAME 相量形式 (repo 已有实现)。对每个 STFT 分辨率 r、每个物理通道:

\[
U_t(f,t)=\frac{S(f,t)\,\overline{S(f,t-1)}}{|S(f,t)||S(f,t-1)|+\epsilon},\qquad
U_f(f,t)=\frac{S(f,t)\,\overline{S(f-1,t)}}{|S(f,t)||S(f-1,t)|+\epsilon}
\]

\[
\mathcal{L}_{\text{IF}}=\frac{\sum_{f,t}\lambda_{\text{IF}}(f)\,w_t(f,t)\,\big(1-\mathrm{Re}[U_t^{\text{pred}}\overline{U_t^{\text{ref}}}]\big)}{\sum_{f,t}\lambda_{\text{IF}}(f)\,w_t(f,t)}
,\qquad \mathcal{L}_{\text{GD}} \text{ 同理用 } \lambda_{\text{GD}}
\]

- \(w_t\) = 现有的 detach 几何平均能量权重 (聚焦有能量的 bin, 空 band 自动无梯度,
  与 overshoot loss 的"防高频幻觉"叙事自洽)。
- **新增**: \(\lambda_{\text{IF}}(f)=g(f;\,2\,\text{kHz},\,2,\,0.1)\) — 依据听神经 phase-locking
  在 ~1.5–4 kHz 以上衰减 (Palmer & Russell 1986), 精细结构相位只在中低频被感知编码。
- \(\lambda_{\text{GD}}(f)=g(f;\,8\,\text{kHz},\,2,\,0.25)\) — group delay 可听阈值到 8 kHz 仍有效
  (Blauert & Laws 1978), 瞬态对齐需要更宽的频带, 所以 GD 门限放高、保底放大。
- 归一化用**加权平均** (λ 乘进权重后再除以权重和), 这样加曲线不改变 loss 量级,
  续训时不会突然改变梯度 scale。
- \(\mathcal{L}_{\text{cd}}\) (归一化复距离) 保持不变, 不加曲线 (它主要是幅相耦合的正则)。

与参考的差异: SAME/Qwen 的 IFGD 是全频等权; 我们是"感知门控"版本。公式只多一个 g(f)。

## 3. (B) FOA 空间协方差 loss — \(\mathcal{L}_{\text{SCM}}\) (本次的"空间 loss"主角)

**原理**: 对 FOA 信号 \(\mathbf{s}(f,t)=[S_W,S_Y,S_Z,S_X]^\top\in\mathbb{C}^4\),
任何线性空间渲染 (binaural、扬声器阵列、rotation) \(\mathbf{y}=A\mathbf{s}\) 的二阶统计量
只通过**空间协方差矩阵** \(C=\mathbb{E}_\tau[\mathbf{s}\mathbf{s}^{H}]\) 依赖于信号
(\(C_y = A C A^H\))。所以在每个感知时频块上匹配 SCM ⇒ 匹配任何下游渲染器能呈现的
方向 (主特征向量)、扩散度 (特征值谱)、通道电平比 (对角)、通道间相位差 IPD (非对角辐角)。
这是比"intensity 方向余弦"更完备的一个量, 现有 `FOASpatialConsistencyLoss` 的
intensity \(I=\mathrm{Re}\{\overline{W}\cdot[X,Y,Z]\}\) 恰好是 SCM 第一行的实部 —— 即其特例。

**定义** (多分辨率 r ∈ {2048, 512}, 时间平滑 ~5 帧, 与现有 foa_spatial 同一套 STFT 设施):

\[
\tilde C(f,t)=\frac{C(f,t)}{\operatorname{tr}C(f,t)+\varepsilon},\qquad
\mathcal{L}_{\text{SCM}}
=\frac{\sum_{f,t} w_E(f,t)\,\Big[
\underbrace{\|\Delta_{\text{mag}}\|^2}_{\text{电平+相干度}}
+\lambda_{\text{IPD}}(f)\,\underbrace{\|\Delta_{\text{phase}}\|^2}_{\text{通道间相位}}
\Big]}{\sum_{f,t} w_E(f,t)}
\]

其中把非对角复误差按极分解拆开 (对角项天然属于 mag 部):

- \(\Delta_{\text{mag}}\): 对角能量占比误差 + 非对角**模长** (相干度/扩散度) 误差;
- \(\Delta_{\text{phase}}\): 非对角**单位相量**误差
  \(\big(1-\mathrm{Re}[U_{ij}^{\text{pred}}\overline{U_{ij}^{\text{ref}}}]\big)\),
  \(U_{ij}=C_{ij}/|C_{ij}|\), 用参考相干度 \(|\tilde C_{ij}^{\text{ref}}|\) 作权
  (扩散 bin 的 IPD 是噪声, 自动降权 —— 接管了旧 loss 的 diffuseness mask, 但是软的);
- \(\lambda_{\text{IPD}}(f)=g(f;\,1.5\,\text{kHz},\,2,\,0.1)\) — duplex theory (Rayleigh):
  精细结构 ITD/IPD 线索只在 ~1.5 kHz 以下起作用, 高频空间感知靠电平差和包络
  → 高频"只管幅度、少管相位", 低频"相位重点管"。
  这个"随频率从相位主导连续过渡到电平主导"的结构与 Qwen 的 Band-Mode Refiner
  动机同源, 但 (i) 做在 loss 不做在架构, (ii) 连续曲线不分硬 band, (iii) FOA 不是 stereo —— 不会"一模一样"。
- \(w_E\) = 参考 trace 能量, per-item 归一化 (与现有实现同款)。
- trace 归一化 ⇒ 整体 scale-invariant 且有界 (trace-1 PSD 的 Frobenius 距离 ≤ 2),
  对 GAN 训练友好。

**与旧 foa_spatial 的关系**: SCM loss 严格包含 intensity 方向项 (Re 部)、dir/omni ratio 项
(对角占比), 还额外覆盖 reactive intensity (Im 部)、XYZ 两两关系与扩散度。
建议做成新 module `foa_scm`, 与旧 loss 并存可配置, ablation 里可以对比 "intensity 版 vs SCM 版"。

**简化选项** (如果想先跑最小版): 直接
\(\mathcal{L}_{\text{SCM}}=\mathbb{E}[w_E\|\tilde C^{\text{pred}}-\tilde C^{\text{ref}}\|_F^2]\)
一条复 Frobenius, 不拆 mag/phase、不加 λ 曲线。少 3 个超参, 但丢掉"频率-感知"卖点。

## 4. (C) band_mel 离散权重 → 参数化曲线 \(w_{\text{mag}}(f)\)

现在 5 个 band 权重 (0.85 / 0.9 / 1.25 / 1.0 / 0.25) 改为从闭式曲线取值:

\[
w_{\text{mag}}(f) \;=\; b\cdot\Big[1 + a\,e^{-\ln^2(f/f_0)/(2\sigma^2)}\Big]\cdot g(f;\,f_r,\kappa,\rho)
\]

拟合现有权重的参数: \(b{=}0.85,\ a{=}0.47,\ f_0{=}4.5\,\text{kHz},\ \sigma{=}0.75,\ f_r{=}15\,\text{kHz},\ \kappa{=}6,\ \rho{=}0.12\)
(见图右; 5 条红线是现值, 蓝线即曲线)。

- 第一因子 = presence 区敏感度 bump: 等响曲线 (ISO 226) 在 2–5 kHz 最敏感、耳道共振 ~3 kHz;
- 第二因子 = 高频置信度 roll-off: >14 kHz 听觉敏感度下降 + 训练集 upsample/带限内容不可靠
  (与 overshoot loss 同一动机, 叙事闭环);
- 论文里可把"有效权重"写成 \(K(f)\cdot w_{\text{mag}}(f)\) 一条总曲线 (K-weighting 已在 MRSTFT 内)。

**实现零风险方案**: 保留 `BandWeightedMelSpectrogramLoss` 机制, band 数从 5 加密到 ~10–12,
每个 band 权重 = 曲线在 band 几何中心的取值。公式进论文, 代码只改 config。
(可选 plus: 在 loss 里支持 per-mel-bin 权重, 工作量也不大。)

---

## 5. Ablation / 续训方案 (定稿: 两波 4+4 GPU)

所有 arm 从 `epoch=12-step=800000.ckpt` resume, `--checkpoint-every 10000`,
`--max-steps 900000` (+100k)。旧 8 卡续训 (wandb `cr8wgqb6`) 已在 23:10 存下
`epoch=13-step=850000.ckpt` —— 它就是免费的 **base_cont 对照臂** (800k→850k
无新 loss 纯续训), 停掉它只损失 ~850k 之后的几千步。

**Wave 1 (先跑, 回答"SCM vs intensity" + "A+B vs base"):**

| GPU | tag | loss 变更 (相对 base) | config |
|---|---|---|---|
| 0-3 | `abl_phase_scm` | +phase_ifgd(0.08) +foa_scm(0.015) | `..._ds1024_z64_phase_scm.json` |
| 4-7 | `abl_phase_int` | +phase_ifgd(0.08) +foa_spatial(0.075) | ~~`..._phase_int.json`~~ **已弃用** |

> **2026-07-20 更新**: 850k held-out 三方对比后 **弃用 phase+INT**
> (`dir_energy_ratio_err` / elevation 崩坏)。已停训并删除
> `/mnt/sdc/ckpts/vae_abl_phase_int` 全部 ckpt。后续只保留 **phase+SCM**
> (目标 900k 后 vs base_cont)。launch 脚本已改为 8 卡。

两臂共享 A (相位项) → 单因子差异, 直接隔离空间 loss 形式; 生产配置也一定含 A,
所以"A 在场时哪个空间项更好"才是运营上要回答的问题。对照点:
- **850k 配对比较**: 两臂@850k vs base_cont@850k (同步数, 排除"多训 50k"混杂);
- **900k 终点**: SCM vs base_cont (INT 已弃)。

**Wave 2 (可选, wave 1 出结果后 ~1 天, 补归因):**

| GPU | tag | loss 变更 |
|---|---|---|
| 0-3 | `abl_phase_only` | 只 +phase_ifgd |
| 4-7 | `abl_spatial_only` | 只 +B_winner (wave 1 胜者) |

加上 base_cont 与 wave 1 即凑齐 2×2 因子表: A 的边际 = (A+B)−(B), B 的边际 =
(A+B)−(A), 单项 vs base。论文 ablation 表由此完整。

**权重/调度 (已标定, 见 §0.7)**: phase 0.08 / scm 0.015 / intensity 0.075,
cosine ramp 800k→815k 从 0 升到目标, 避免打崩 GAN 平衡。三个新项各贡献
总重建项的 ~8% (anchor = mrstft×1.0 = 0.255)。

**batch**: 旧 8 卡跑 batch 2/GPU (有效 16)。4 卡臂优先尝试 `--batch-size 4`
(有效 16, 与 base_cont 可比); OOM 再降 2 (两臂必须一致, 且注明与 base 的
有效 batch 差异为 caveat)。

**launch (待确认后执行):**

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT="/mnt/sdc/ckpts/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/checkpoints/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/cx9iuuoa/checkpoints/epoch=12-step=800000.ckpt"
DATA=/mnt/sdc/ckpts/vae_ds1024_z64_hf_overshoot_decay_350k_8gpu/configs/dataset_frozen_1018957.json
A=stable_audio_tools/configs/model_configs/autoencoders

# 0) 停旧续训 (已有 850k ckpt, 建议现在停)
#    tmux/进程: uv run ... train_4ch.py --name vae_ds1024_z64_hf_overshoot_decay_350k_8gpu

mkdir -p /mnt/sdc/ckpts/vae_abl_phase_scm /mnt/sdc/ckpts/vae_abl_phase_int
tmux new-session -d -s abl_scm "CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train_4ch.py --model-config $A/ablation_arms/stable_audio_4ch_vae_ds1024_z64_phase_scm.json --dataset-config $DATA --ckpt-path \"$CKPT\" --name vae_abl_phase_scm --num-gpus 4 --batch-size 4 --num-workers 8 --precision bf16-mixed --strategy ddp_find_unused_parameters_true --save-dir /mnt/sdc/ckpts/vae_abl_phase_scm --checkpoint-every 10000 --max-steps 900000 --logger wandb --seed 42 2>&1 | tee /mnt/sdc/ckpts/vae_abl_phase_scm/train.log"
tmux new-session -d -s abl_int "CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train_4ch.py --model-config $A/stable_audio_4ch_vae_ds1024_z64_phase_int.json --dataset-config $DATA --ckpt-path \"$CKPT\" --name vae_abl_phase_int --num-gpus 4 --batch-size 4 --num-workers 8 --precision bf16-mixed --strategy ddp_find_unused_parameters_true --save-dir /mnt/sdc/ckpts/vae_abl_phase_int --checkpoint-every 10000 --max-steps 900000 --logger wandb --seed 42 2>&1 | tee /mnt/sdc/ckpts/vae_abl_phase_int/train.log"
```

前 15k steps 观察 wandb: `train/foa_scm_weight`、`train/phase_ifgd_weight`
按 ramp 上升; `train/loss_adv`、`feature_matching` 无突变; 子项
`foa_scm_{level,coherence,ipd}_raw`、`phase_ifgd_{if,gd,cd}_raw` 下降。

**评测**: `eval_vae_recon` (lsd_db, doa_az/el_err, dir_energy_ratio_err,
ic_corr_err) 在 {base@800k, base_cont@850k, 两臂@850k, 两臂@900k}; 建议补
**CCPC/ICPC** (εar 定义, SAME/Qwen 都报, 论文可横向对比)。预期: phase 项动
ICPC/CCPC 与瞬态; 空间项动 doa/dir_ratio/ic_corr。磁盘: 每 ckpt 2.5GB ×
10 × 2 臂 = 50GB, /mnt/sdc 余 492GB, 够。

- 资源估计: 旧 run 800k→850k 用时 8.5h (~1.6 it/s, batch 2); batch 4 慢一些,
  100k steps 预计 ~20-28h/臂 (并行, 共 1 天多)。

## 6. 差异化声明 (回应"别和 qwen music 一模一样")

1. **频率门控相位 loss**: SAME/Qwen/εar 的 IF/GD 都是全频等权; 我们引入 psychoacoustic
   门控曲线 g(f) (phase-locking / duplex / GD 阈值三组文献支撑), 且是连续曲线非硬分带。
2. **FOA SCM loss**: 三家都是 stereo (M/S + ILD 回归最多); 我们在 FOA 上用
   "SCM 完备性" (任何线性渲染只依赖 SCM) 这一论证给出统一的空间目标,
   intensity-vector loss 是它的特例。Qwen 的 band-mode 思想被吸收为 loss 内的
   连续 λ_IPD(f) 权重, 而非 refiner 架构。
3. 高频处理: Qwen 用 refiner 网络修, 我们用 (C) 曲线加权 + 已有 overshoot 非对称项,
   全部在 loss 侧, 不加参数、不改推理路径。

## 7. 拍板记录 (2026-07-20)

1. ✅ SCM 用拆分加权版 —— novelty 检索确认是我们的 idea (§0.6), 已实现。
2. ✅ intensity 不删, 作为对比臂: GPU0-3 SCM / GPU4-7 intensity, 均从 800k
   resume, 每 10k 存 ckpt, 跑到 900k (§5 wave 1)。
3. ✅ (C) band 曲线本轮不上 (避免三因子混杂), 公式进论文; 待 ablation 结束后
   若需要再单独开一臂。
4. ⏳ 待确认后执行: 停旧 8 卡续训 (850k ckpt 已落盘, 现在停几乎无损失) →
   按 §5 launch。wave 2 (单项归因臂) 看 wave 1 结果再定。
5. TODO: eval 脚本补 CCPC/ICPC 指标; 论文相关工作补引 Hirvonen & Namazi
   (2411.12008)、Qiao et al. (2409.06954)、FOA Tokenizer (2510.22241)。
