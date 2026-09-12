# 功能入口

优先从这里定位代码。路径均相对于 AMBIT 根目录。运行模型入口前仍需提供对应的数据、权重与依赖；这些资源不在源码包内。

| 功能 | 入口或主要实现 |
| --- | --- |
| 数据下载 | `data_download/scripts/download_dataset.py`、`data_download/scripts/download_all.py` |
| FOA 数据构造 | `dataset/synthesis/`、`scripts/t2a/data/` |
| 音频/视频 caption | `dataset/captioning/caption_audio.py`、`caption_sphere360.py`，共用 `qwen_model.py` |
| 生成 AR → ScenePlan → FOA | `scripts/t2a/inference/generate_foa_from_raw_english.py`、`run_generation_ar_bundle.py` |
| 编辑 AR/CLAP → FOA | `scripts/t2a/inference/edit_foa_with_clap44.py` |
| AR、DiT、CLAP 训练 | `scripts/t2a/train/`；不同实验入口在 `scripts/t2a/experiments/`，不把版本号最大者自动视为发布版本 |
| 原始编辑数据读取 | `stable_audio_tools/data/sceneplan_transfusion_editing_dataset.py` |
| 多声源编辑数据读取 | `stable_audio_tools/data/sceneplan_compound_editing_dataset.py`，继承原始读取逻辑，单独声明契约 |
| VAE latent 解码/试听 | `scripts/vae/eval/decode_latents_4ch.py`；旧下载目录入口仅转发 |
| 评估 | `scripts/t2a/eval/`、`scripts/evaluation/` |
| 共用 artifact I/O | `stable_audio_tools/data/artifact_io.py` |
| 共用内存音频与 RIR | `stable_audio_tools/data/editing_memory_io.py`、`editing_rir_pool.py` |
| 扩充数据与续训研究 | `research/`，仍有环境和外部运行资源依赖 |

现有 `pyproject.toml` 与 `uv.lock` 保留，避免在代码整合时顺带改变模型依赖版本。数据下载包保留独立安装配置，允许不安装训练依赖而使用下载功能。
