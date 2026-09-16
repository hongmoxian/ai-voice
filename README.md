# ai-voice · 本地声音克隆语音助手

用**你自己的声音**说话的 AI 语音助手。声音克隆跑在本地，只有对话大脑走云端 API。

> **状态：方案调研完成，等待在 Apple Silicon 上实测。**
> 在 Windows CPU 上已完成一轮完整实测（数据见下），
> 但调研发现 **Apple Silicon + MLX 可以快 14~20 倍**，因此运行目标改为 Mac。
> 下一步是在 Mac 上量出候选方案的真实速度。

---

## 一、核心结论

### 两条路线的实测/公开数据对比

| | Audio8 0.6B INT4<br>（Windows CPU，**我们实测**） | GPT-SoVITS<br>（Apple M4 + MLX，**官方基准**） | 差距 |
|---|---|---|---|
| **RTF** | 2.32 ~ 3.17 | **0.16** | **快 14~20 倍** |
| **首块延迟 / TTFT** | 2.4 ~ 2.6 秒 | **0.85 秒** | 快 3 倍 |
| **连续流式播放** | ❌ 必然断流（生成只有播放速度的 43%） | ✅ 6.25× 实时 | **质变** |
| 硬件 | 无 GPU 的 x86 | Apple Silicon | — |

> RTF = 生成时间 ÷ 音频时长。**> 1 表示比实时慢。**

**这个差距改变了整个项目的设计空间。** 如果 Mac 上能拿到 RTF < 1，那么
「限制回复长度」「短开场白」「分句播放」这些妥协**全都不需要**，可以直接做流畅的流式对话。

### 运行目标

**Mac 作为运行目标**，原因不是 Mac 更便宜，而是 **MLX 是真正能用上 Apple GPU 的原生路径**。

⚠️ 注意区分：**ONNX Runtime / PyTorch-MPS 路线在 Mac 上确实用不到 GPU**
（XTTS 还有明确的 MPS 报错），这也是本 README 早期版本误判「Mac 帮不上忙」的原因。
**MLX 是另一条路，生态已经成熟。**

### 待确认

- **目标 Mac 的具体型号与内存**（M1 与 M4 差距约 2 倍，内存决定可用模型规模）
- **mlx-audio 的克隆模型没有公开速度数据**（只找到 Kokoro 的 22×，但它不能克隆）

结论：**必须先实测，不要凭公开数据下结论。**

---

## 二、方案调研

### 2.1 候选方案

#### A. `mlx-audio` · MIT · **首选**

> 作者 Prince Canuma，基于 Apple MLX。`pip install mlx-audio`
> 仓库 / 文档：<https://github.com/Blaizzy/mlx-audio> · <https://blaizzy.github.io/mlx-audio/>

- **13 个模型支持声音克隆**，全部使用统一 API（`ref_audio` + `ref_text`）
- 针对本项目场景的最佳选择：
  | 模型 | 规模 | 语言 | 克隆 | 流式 |
  |---|---|---|---|---|
  | **Qwen3-TTS** | 0.6B / 1.7B | **中英日韩** | ✅ | ✅ |
  | **OmniVoice** | 0.6B | 646+ | ✅（零样本） | ✗ |
  | Chatterbox | 0.5B | 23 种 | ✅ | ✗ |
  | VoxCPM2 | 2B | 30 种 | ✅ | ✗ |
  | Higgs Audio v3 | 4B | 100 种 | ✅ | ✗ |
- **自带 OpenAI 兼容服务**：
  - `/v1/audio/speech` —— TTS + 克隆
  - `/v1/audio/transcriptions` —— **STT**（Whisper / Qwen3-ASR / Parakeet）
- **一箭双雕**：STT 也在同一个库里，**不需要单独再装 faster-whisper**
- 接口形态与已有的 Audio8 客户端**几乎一致**，后端改动很小

**优点**：架构最简单（TTS + STT 一个服务）、接口与我们已有代码贴合、MIT 可商用
**风险**：克隆模型的速度**无公开数据**，必须实测

#### B. `GPT-SoVITS` + MLX · 社区最成熟

> 53k stars，中文效果公认最好

官方 README 的设备基准表（**唯一有公开 M 系列数据的方案**）：

| 设备 | RTF | TTFT | 后端 |
|---|---|---|---|
| RTX 5090 | 0.05 | 150 ms | CUDA |
| **Apple M4** | **0.16** | **850 ms** | MLX Varlen |
| i7-12700K | 0.28 | — | Torch（x86 CPU） |

- 支持零样本 + 少样本微调（1 分钟数据即可微调）
- ⚠️ **许可证待确认**

**优点**：有公开实测数据、中文效果最好、社区最大
**代价**：是完整的训练/推理工具链，集成需用其 `api_v2.py`，比 mlx-audio 重得多

#### C. `OminiX-MLX` 的 GPT-SoVITS 纯 Rust + MLX 移植

- 文档称「Apple Silicon 上 4× 实时」（<https://mintlify.wiki/OminiX-ai/OminiX-MLX/tts/gpt-sovits>）
- 生态小、成熟度低，作为备选

#### D. 相关项目（供参考）

- **`soniqo/speech-swift`** —— Apple Silicon 语音工具包（ASR / TTS / VAD / 说话人分离），MLX + Swift
- **腾讯 AuK** —— 1.5B 语音生成 + 编辑统一模型，有 MLX 移植

### 2.2 已排除的方案（**不要再重新评估**）

| 方案 | 排除原因 |
|---|---|
| **XTTS-v2** | CPU 上 RTF 5.31（21 秒一句话）；M 系列有明确 MPS 报错，Mac 也救不了；许可 **CPML 非商用** |
| **Audio8 0.1B INT8** | **反直觉重点**：参数少 6 倍，实测**反而更慢**（RTF 2.69~4.12，首块 5.2~5.9 秒）。根因是其 README 写明 prefill **一次一个位置**，而 0.6B 是批量 prefill。**参数少 ≠ 快** |
| **Kokoro-82M** | 极快（M4 Max 上 RTF 22×），但**不能克隆特定人声**，只能选预设音色 |
| **GPT-SoVITS 在 x86 CPU 上** | RTF 0.28 看着能跑，但那是**批量 40** 的数据；单条实时性差，且中文场景仍需 MLX 才划算 |
| **Audio8 0.6B INT4 作为最终方案** | 不是不能用——**Windows 上它是唯一可行方案**（见下方实测）。但在 Mac 上比 MLX 路线慢一个数量级，**降级为 Windows 备选** |

> ⚠️ 本 README 早期版本曾断言「Apple GPU 对这些模型用不上」，
> 该结论**仅对 ONNX Runtime / PyTorch-MPS 路线成立**，对 MLX 不成立。已更正。

---

## 三、架构

### 目标架构（Mac）

```
浏览器（按住说话）
   │ WebSocket
   ▼
FastAPI 后端 ── 我们写的部分，只做编排 + 网页
   ├─ 听 ┐
   ├─ 想 │ 云端 LLM API（DeepSeek / OpenAI）
   └─ 说 ┘
   │
   ▼
mlx-audio 服务（本地，端口 8000）
   ├─ POST /v1/audio/speech          TTS + 声音克隆
   └─ POST /v1/audio/transcriptions  STT
```

**我们自己的项目完全不需要 torch / transformers / coqui-tts**，依赖约 200 MB。

### Windows 备选架构

Windows 上没有 MLX，回退到 **Audio8 0.6B INT4 的 ONNX Runtime CPU 服务**
（自带 `/api/tts`、`/api/tts/stream`、`/api/tts/cancel`、`/api/voices/register`），
由 `scripts/setup_audio8.py` 拉取。性能见下节，足以跑通链路，但长回复会断流。

---

## 四、实测数据存档

### 4.1 Audio8 0.6B INT4 · 纯 CPU（**我们实测**）

测试机：Windows，**12 逻辑核心，无 GPU**，ONNX Runtime 5 线程。

| 文本 | 长度 | 音频时长 | 整段耗时 | RTF | 首块延迟 |
|---|---|---|---|---|---|
| 你好，我在。 | 6 字 | 1.25 s | 3.98 s | 3.17 | **2.35 s** |
| 好的，我帮你查一下今天的天气，稍等一下。 | 20 字 | 4.32 s | 12.19 s | 2.82 | **2.56 s** |
| 这个问题的答案是这样的：… | 60 字 | 12.96 s | 30.09 s | 2.32 | **2.49 s** |

模型加载 4.57 s、音色注册 10.65 s，均为一次性开销。

**线程数扫描**（同一条 20 字文本）：

| 线程 | 4 | **6** | 8 | 12 |
|---|---|---|---|---|
| RTF | 3.31 | **2.52** | 3.01 | 7.05 |

→ **用物理核数，不要用逻辑核数。** 12 线程比 6 线程慢 2.8 倍（超线程争抢）。

**结论**：首块延迟 2.4~2.6 秒可用（且与文本长度无关），但 **RTF > 1 意味着长回复无法连续播放**。

### 4.2 第三方公开数据（引用来源）

| 来源 | 数据 |
|---|---|
| GPT-SoVITS 官方 README | Apple M4 + MLX：**RTF 0.16 / TTFT 850 ms**；i7-12700K：RTF 0.28 |
| [vllm-mlx 音频基准](https://vllm-mlx.is-a.dev/benchmarks/audio/) | M4 Max：Kokoro-82M **RTF 22×**（0.045 s/秒音频）；M1 Max：11.9~15.5×；whisper-large-v3-turbo STT RTF 55× |
| [CPU 声音克隆横评](https://heyneo.com/blog/voice-cloning-models-cpu-benchmark)（2026-08） | 8 核纯 CPU：Kokoro RTF 0.12 / Audio8 1.42 / XTTS **5.31** |
| [Zenodo 论文](https://doi.org/10.5281/zenodo.19458410)（2026-04） | M4 Pro 24GB 上 GPT-SoVITS 端到端延迟 **1.5 s**（37 分钟数据微调约 70 分钟）；识别并修复 7 项平台兼容问题，含 float16 精度错误 |

> ⚠️ 部分数据为非同一进程的 A/B 对比，方向可信、精度需自行验证。

### 4.3 已实测排除

| 方案 | 实测 |
|---|---|
| Audio8 0.1B INT8（本机） | RTF 2.69~4.12，首块 5.20~5.86 s —— **比 0.6B 更慢，已弃用** |

---

## 五、安装与复现

### 前置

- **Windows**：Python 3.11+
- **macOS**：Python 3.10+，且为 Apple Silicon（M1 或更新）
- 网络：若 `huggingface.co` 直连不通，脚本会自动降级
  **本地代理 → `hf-mirror.com`**

### 在 Mac 上（目标环境）

```bash
git clone <this-repo> ai-voice && cd ai-voice
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip

# mlx-audio 路线（推荐先测这个）
./.venv/bin/python -m pip install mlx-audio

# 音频工具
brew install ffmpeg
```

### 在 Windows 上（备选，跑通链路用）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 克隆 Audio8 仓库（钉 commit）+ 下载模型（约 968 MiB）
.\.venv\Scripts\python.exe scripts\setup_audio8.py

# 实测速度
.\.venv\Scripts\python.exe scripts\bench_audio8.py
.\.venv\Scripts\python.exe scripts\bench_audio8.py --sweep-threads
```

### 没有录音时怎么先测速度

`data/fixtures/reference_huihui.wav` 是用 Windows 内置中文语音合成的测试参考音频
（14.7 秒，54 字，**非真人**），配同名 `.txt` 作为精确转写。它让你**不必先录音**
就能跑通链路、量出速度。

换成自己的声音：录 **0.5~30 秒**清晰朗读 + **逐字一致的转写文本**。

---

## 六、已知的坑

1. **中文 Windows + Audio8 的编码 bug**
   `registration.py` 用 `Path.write_text()` 未指定 `encoding`，中文 Windows 默认 GBK，
   写含中文的 `meta.json` 会抛 `UnicodeEncodeError`。
   本项目用 **`PYTHONUTF8=1`**（PEP 540）兜住，脚本会自动重新 exec 自己，**不改第三方代码**。
   macOS 默认 UTF-8，不受影响。

2. **PowerShell 的 `Set-Content -Encoding UTF8` 会写 BOM**
   读参考文本必须用 `utf-8-sig` 剥掉，否则 BOM 混入文本导致注册失败。

3. **两套 Audio8 运行时不能共用 venv**
   0.6B 要求 `onnxruntime==1.24.4`，0.1B 要求 `>=1.22,<1.24`。

4. **`D:\` 根目录在 Windows 上不可写**（ACL 只给 `Authenticated Users` 只读）。

5. **换行符**：仓库带了 `.gitattributes`，`.sh` 若带 CRLF 会让 bash 直接报错。

---

## 七、路线图

- [x] 验证纯 CPU 声音克隆可行性，实测出硬约束
- [x] 调研 Apple Silicon / MLX 路线，发现快 14~20 倍的方案
- [ ] **在 Mac 上实测 mlx-audio + Qwen3-TTS 的真实 RTF 与首块延迟** ← 当前
- [ ] 对比 GPT-SoVITS MLX（唯一有公开 M 系列数据的方案）
- [ ] 定型方案，录用户自己的声音，确认克隆质量
- [ ] 接 STT：说话 → 文字
- [ ] 接云端 LLM：文字 → 回复
- [ ] 串成实时对话 + 网页界面（按住说话）
- [ ] 打断插话

---

## 八、目录结构

```
ai-voice/
├── app/                    后端（待开发）
├── web/                    前端（待开发）
├── scripts/
│   ├── setup_audio8.py     克隆 Audio8 仓库（钉 commit）+ 下载模型，含代理/镜像选路
│   └── bench_audio8.py     速度实测，产物可试听
├── data/fixtures/          测试参考音频（合成音，非真人）
├── data/bench/             实测产物（不入库）
├── vendor/Audio8_TTS/      第三方仓库 + 模型（不入库，由脚本拉取）
└── models/                 下载缓存（不入库）
```

---

## 九、隐私与伦理

- 克隆声音**须事先取得本人同意**，合成音频在适当场合应明确标注
- `voices/`、`data/recordings/`、`*.local.wav` 已在 `.gitignore` 中排除：
  音色档案只有几 KB，但它是**从录音提取的个人生物特征**，不应入库或分享

## 十、许可

- 本项目：待定
- 依赖方案：**mlx-audio（MIT）** · Audio8（Apache-2.0）· GPT-SoVITS（待确认）
