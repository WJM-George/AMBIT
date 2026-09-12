# AMBIT

这是从独立 staging 副本整合到 AMBIT 的源码目录，包含数据构造、生成/编辑 AR 与 DiT、CLAP 及共用训练和评估代码。原工作区和 staging 均保留；模型、数据和环境不随代码复制。尚未完成跨机器运行验证。

已完成一轮功能去重：编辑数据读取、内存音频/RIR 辅助逻辑、评估聚合和 VAE 解码统一实现，旧入口保留兼容；清除了孤立本机运维脚本及 retired P10 的历史实现。详见 [整合记录](docs/CLEANUP.md)。

本次合入又统一了 artifact I/O、caption 模型加载、数据审计配额及评估元数据配置。可从 [功能入口](docs/ENTRYPOINTS.md) 查找实现，迁移范围和验证见 [本次合入记录](docs/INTEGRATION.md)。原 Python 包名保持 `stable_audio_tools`，以兼容现有 import 和检查点。

## 目录

| 目录 | 内容 |
| --- | --- |
| `stable_audio_tools/` | 共用模型、数据读取、生成/编辑 AR 与 DiT、CLAP、训练和推理实现 |
| `dataset/` | 数据索引、caption、FOA 合成及语音处理 |
| `scripts/t2a/data/` | ScenePlan 与编辑数据构造 |
| `scripts/t2a/inference/` | 生成与编辑推理入口 |
| `scripts/t2a/train/` | AR、DiT、CLAP 训练入口 |
| `scripts/t2a/experiments/` | AR/CLAP 各版本研究实现；不代表全部推荐使用 |
| `data_download/` | 独立的数据下载包及脚本 |
| `research/` | 新增 500k 数据与 1.75M 续训代码快照，尚需路径适配 |
| `tests/` | 原有测试源码 |
| `docs/SOURCE_MANIFEST.json` | 初次复制时每个文件的原路径、大小和 SHA-256（历史记录） |
| `docs/CURRENT_SOURCE_AUDIT.json` | 整合后的 Python/Shell 语法、哈希和绝对路径检查 |
| `docs/CLEANUP_MANIFEST.json` | 此次清理的文件变更与前后哈希 |
| `docs/INTEGRATION_MANIFEST.json` | staging → AMBIT 的文件差异和来源一致性检查 |

保留原包名与相对结构以减少 import 破坏。未复制模型权重、实际数据、音视频、缓存、虚拟环境、训练日志或原 `.git`。共用源码及脚本中仍包含 OPSD 和历史实验；这些不视为已完成的发布功能，也没有自动执行。

## 状态边界

- 生成 AR→DiT：已有 8k 评估和 CLI 冒烟记录；原记录仍有人工听评和自由补全合理性审查未完成。
- Dataset：最新编辑队列 RESULT 已完成，新增空间/多声源 500k train 数据已有 DATA_READY；训练 pair 总量可达 1.75M。
- 编辑：既有 AR20k 音频及论文评估完成，DiT50k/65k 结果已有记录；新结构 AR/DiT 联训和 1.75M 数据续训不可据此宣称最终完成。
- CLAP：新增 50k 训练和五检查点 20k 选型完成，新增 10k 为 AR 接入主候选；不代表后续所有 AR 质量验收完成。
- OPSD：仍为研究开发内容。

依据来自原工作区 `reports/editing_pipeline_20260912/RESULT.json`、`reports/generation_ar_CURRENT.json`、`reports/editing_dit_mixed1750k_30k_20260912/README.md`，以及原数据盘 AR/CLAP 的 CURRENT 与 README。部分交接文档落后于后续 RESULT；此处不以旧状态推断实时训练步数。

## 发布前尚需完成

1. 修复或完成 GitHub 仓库连接。合入前 AMBIT 只有不完整的 `.git`，Git 无法识别为仓库；本次保留它，没有声称 clone 成功，也没有 commit/push。
2. 选择正式发布的训练/推理版本；实际重复实现已合并，相互依赖的历史实验仍保留。
3. 将检查报告列出的本机绝对路径替换为明确的输入参数或配置，核对动态加载及外部资源依赖。不要直接运行历史 launcher，它们可能操作原数据目录。
4. 在隔离环境验证安装、正式 CLI 和代表性 CPU/GPU 工作流。本次在 AMBIT 下通过 35 项 CPU 测试、静态检查和提取函数一致性检查；staging 阶段另有 16 条真实数据读取对照。这不是完整训练/推理复现。

上游 `LICENSE`、`LICENSES/` 和作者信息原样保留；此快照没有重新声明全部代码归属。
