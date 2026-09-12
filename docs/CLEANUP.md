# 源码整合记录 — 2026-09-12

本次只修改 AMBIT-code-staging。原工作区、原训练任务、数据和权重未操作。

## 已合并

| 功能 | 唯一实现 | 兼容方式 |
| --- | --- | --- |
| 编辑 latent 读取、完整性检查、ScenePlan 条件编译 | `stable_audio_tools/data/sceneplan_transfusion_editing_dataset.py` | 多声源契约在 `sceneplan_compound_editing_dataset.py` 中以子类声明；两份研究读取器改为转发入口 |
| 内存音频读取和源音频缓存 | `stable_audio_tools/data/editing_memory_io.py` | 两份研究 `memory_io.py` 转发同一实现 |
| 冻结 RIR 扩展加载与线程控制 | `stable_audio_tools/data/editing_rir_pool.py` | 各研究入口分别绑定前序资源位置；保留扩展哈希检查、128 个逻辑归约块及上下文退出恢复 |
| P11 challenge 指标聚合 | `stable_audio_tools/data/p11_challenge_metrics.py` | 单次评估和分片合并共用 `_mean`、`_aggregate`；原函数名称仍可导入 |
| 四通道 VAE 解码 | `scripts/vae/eval/decode_latents_4ch.py` | 下载目录的旧入口转发，默认继续输出试听 WAV；正式入口默认不输出试听 WAV |

解码统一入口同时支持 `--decode-all`、`--listen-wav`、`--no-listen-wav`。声学输出、采样方式及已有参数的默认值沿用各自原入口。

原始数据读取器仍拒绝多声源 v2 契约；多声源子类仍拒绝原始 v1 契约。没有把两个数据版本的校验条件简单放宽，也没有取消 latent、元数据、源/目标数量检查。

两个研究 `common.py` 的 REPO 改为从本副本推导，确保引用本次合并代码。研究运行目录、原生扩展、数据盘和调度配置仍需另行适配，不是跨机器运行验证。

## 已清理

- 删除没有其他源码引用的本机运维脚本：`scripts/system/execute_sceneplan_v2_deletion_manifest.py`、`scripts/system/install_log_budget_10g.sh`、`scripts/utils/migrate_vggsound_to_sdc.sh`。
- 11 个原本标记为 retired 的 P10 launcher 移除历史实现，原命令路径只输出退役提示并以 64 退出；即使设置旧的 `P10_ALLOW_RETIRED_ROUTE=1` 也不会启动。历史复现实现在原工作区保留。
- 删除以上 launcher 不再使用的 `_retired_p10_route_guard.sh`。

没有按文件名批量删除 AR/CLAP 的 v1/v2/v3。部分较新实验继承较早版本，训练状态/动态导入还可能绑定旧模块；这些不是可直接删除的重复副本。通用 checkpoint prune 工具也保留，它不等同于绑定本机的清盘脚本。

## 验证

- 原有编辑计划测试：17 项通过。
- 新增整合回归测试：10 项通过，覆盖数据契约隔离、未完成数据拒绝、缓存副本独立性、内存 FLAC 20 条回放、RIR 恢复/逻辑线程数/哈希拒绝、两条解码 CLI 的默认值与覆盖参数。
- 真实原始 TRAIN 和多声源 TRAIN 各 8 条：用同一个实际本地 Qwen tokenizer 调用合并前后读取逻辑，张量及全部返回元数据逐项相同。只读取现有数据，没有训练、生成、GPU 计算或写回数据。详见 `CLEANUP_READER_PARITY.json`；这不是全量数据验收。
- 两个评估入口的聚合函数与提取后的函数 AST 完全相同。
- 全部 Python/Shell 语法检查见 `CURRENT_SOURCE_AUDIT.json`。可用 `python tools/audit_sources.py` 重跑静态检查。

测试命令（使用已具备依赖的 Python 环境）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider tests/test_source_consolidation.py tests/test_sceneplan_transfusion_editing_plan.py
python tools/audit_sources.py
```

`SOURCE_MANIFEST.json` 和 `SNAPSHOT_CHECK.json` 是初始复制时的历史记录，不是修改后文件的哈希或当前检查。`CLEANUP_MANIFEST.json` 记录此次增加、删除、修改和前后哈希；`CLEANUP_BASELINE.json` 指向本机临时完整备份，临时目录可能被系统清理。长期历史来源仍见初始来源清单。

本次没有运行完整训练、验证所有 checkpoint 恢复路径或完成跨机器部署，也没有 clone/push GitHub。
