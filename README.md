# ad-pipeline-independent

一个从零独立实现的 AI 视频广告生成 pipeline。核心脚本（`extract_product_views.py` /
`ad_director.py` / `render_pipeline.py` / `finalize_ad.py`）均为全新设计编写，不参考、
不复制任何既有代码库的源码；`ad_tracker.py`、`lib/story_reel/` 与
`scripts/run_full_pipeline.sh` 为搬运自本人原创的既有文件（后者已调整路径引用以适配本仓库结构）。

## 目录结构

```
ad-pipeline-independent/
├── README.md
├── requirements.txt
├── config/settings.example.json   # 复制为 settings.json 并填入真实密钥（已 gitignore）
├── scripts/
│   ├── extract_product_views.py   # 产品多角度图片 -> assets/<brand>/<view>.png
│   ├── ad_director.py             # LLM 生成分镜脚本 -> workflows/<title>.json
│   ├── render_pipeline.py         # spec -> 静音渲染视频（按 animate_backend 选后端）
│   ├── finalize_ad.py             # 静音视频 -> 配音/字幕/BGM -> outputs/<title>_full.mp4
│   ├── ad_tracker.py              # 发布记录与效果追踪（CSV/xlsx）
│   └── run_full_pipeline.sh       # 串联以上全部步骤的一键脚本
├── lib/story_reel/                # 渲染依赖抽象层（stub，供 render_pipeline.py 复用）
├── workflows/                     # ad_director.py 生成的 spec JSON
├── outputs/                       # 渲染/成片/追踪表（已 gitignore 大文件）
└── tests/
```

## 流程

1. `extract_product_views.py` — 将产品原始图片按角度分类整理成 `assets/<brand>/<view>.png`。
2. `ad_director.py` — 调用 LLM，根据产品视图（+ 可选 brief）生成分镜脚本 spec JSON，
   写入 `workflows/<title>.json`。spec 中**必须**包含 `animate_backend` 字段
   （`wan_flf` = 付费云端渲染，`interp` = 免费本地占位/插值渲染）。
3. `render_pipeline.py` — 读取 spec，按 `animate_backend` 选择渲染后端，产出静音视频
   `outputs/<title>_silent.mp4`。两种后端都基于 `lib/story_reel/` 这层抽象实现，
   因此本脚本本身不依赖任何特定的本地渲染环境。
4. `finalize_ad.py` — 给静音视频加配音（TTS）、字幕、背景音乐，输出成片
   `outputs/<title>_full.mp4`。
5. `ad_tracker.py register` — 登记成片，后续用 `publish` / `metrics` / `report` /
   `export` 追踪多平台发布效果。

一键跑通：

```bash
./scripts/run_full_pipeline.sh <raw_photo.png> <product_name> ["可选 brief"]
```

## 当前状态

四个核心脚本目前只有函数签名 + docstring + `TODO`（`NotImplementedError`），设计方向
确认后再逐个填充具体实现。`ad_tracker.py`、`lib/story_reel/*`、
`scripts/run_full_pipeline.sh` 已是可直接使用的完整实现。

## 安装

```bash
pip install -r requirements.txt
cp config/settings.example.json config/settings.json  # 然后填入真实密钥
```

需要 Python 3.10+ 与已加入 `PATH` 的 `ffmpeg`。
