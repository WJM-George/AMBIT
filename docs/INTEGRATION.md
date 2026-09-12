# staging → AMBIT 合入记录

2026-09-12：从 `/home/tanhe/dataset_storage/AMBIT-code-staging` 复制已清理的源码到 `/home/tanhe/dataset_storage/AMBIT`。合入前目标没有业务文件，只有不能被 Git 识别的不完整 `.git`。未改动其内容，未执行 commit、push 或启动任何训练。

## 本次继续合并

- 两个研究工作流的 canonical JSON、SHA-256、读写、zlib 编解码和只读 SQLite 入口收拢到 `stable_audio_tools/data/artifact_io.py`。两个 `common.py` 保留运行配置以及原导出名称；保留模块导入，避免破坏历史 `from common import *` 消费方。
- 内存音频缓存、RIR 和评估报告复用以上 I/O。冻结研究记录继续拒绝 NaN/Infinity；旧评估哈希显式保持原来的非有限值序列化规则，没有因合并改变已有报告哈希协议。
- 音频、视频 caption 共用一个延迟加载的 Qwen 模型入口，仍支持直接脚本和 Python 包导入。
- manifest/materialized 两种数据审计共用配额计算；DiT 与 Spatial-CoT 评估共用元数据 provider 配置。
- 删除一次性的 `tools/export_sources.py`，避免从旧机器路径再次导入清理前的代码。原脚本仍在 staging 中。

staging 阶段已经完成的数据读取器、VAE 解码、RIR、内存音频和评估聚合去重均已带入。没有再次复制原工作区的旧实现来覆盖这些改动。

## 保留的相似代码

`DUPLICATION_AUDIT.json` 列出剩余较长函数的精确 AST 相同项及保留理由。文本相同不意味着在当前全局配置下行为相同：两套 GPU runtime 使用不同的选卡范围、资源锁和旧服务检查；审计 main 调用不同审计逻辑；冻结 renderer 与 model-facing ScenePlan 保持协议边界。未把这些实现强行合并。

本次没有按 v1/v2/v3 文件名删除研究代码。较新 AR/CLAP 实验存在对较早模块的引用；仍需要按最终模型版本逐项判断哪些可从正式发布中排除。也没有把不同采样、loss、评估分母的实现仅因用途相近而归为重复。

## 验证与范围

- 在 AMBIT 下运行 35 项 CPU 测试通过：17 项编辑计划、10 项已有整合回归、6 项 artifact I/O、2 项 SpatialFamilyMetadata。
- artifact 测试覆盖 Unicode/数值序列化、压缩字节、报告非有限值政策、失败写入时原文件保留及只读数据库拒绝写入/创建。
- 新提取的模型加载、配额、metadata provider 与来源函数 AST 相同；caption 导入身份、配额输入用例及两个 caption CLI 的 `--help` 通过。见 `INTEGRATION_EXTRACTION_CHECK.json`。
- 全部 Python/Shell 语法检查和当前源码哈希见 `CURRENT_SOURCE_AUDIT.json`。
- staging 的文件哈希未变；目标差异见 `INTEGRATION_MANIFEST.json`。`INTEGRATION_BASELINE.json` 是合入时的完整来源清单。

`CLEANUP.md`、`CLEANUP_*`、`SOURCE_MANIFEST.json`、`SNAPSHOT_CHECK.json` 保留上一轮历史证据，不应当作 AMBIT 当前文件哈希。该轮 16 条真实读取对照没有在本轮重复运行。

仍有本机路径与外部动态加载资源未迁移，未验证全量训练、模型恢复或跨机器部署。没有把目录拷贝完成等同于 GitHub clone 成功。
