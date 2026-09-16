# ai-voice · 本地声音克隆语音助手

用**你自己的声音**说话的 AI 语音助手。声音克隆完全跑在本地 CPU 上，
只有对话大脑走云端 API。

> **状态：技术路线已验证，应用层待开发。**
> 声音克隆的 CPU 速度已经实测完成（见下方「实测数据」），
> 下一步是把它接上语音识别、大模型和网页界面。

---

## 为什么是这套技术栈

最初的计划是 GPT-SoVITS / CosyVoice 那类方案，但它们**为 GPU 设计**，
这台机器没有 NVIDIA 显卡。经过实测对比后改成了 Audio8：

| 方案 | CPU 上的 RTF | 一句 6 秒音频 | 能克隆 | 许可 |
|---|---|---|---|---|
| Kokoro-82M | 0.124 | 0.5 秒 | ❌ 固定音色 | Apache 2.0 |
| **Audio8 0.6B** | **2.32 ~ 3.17** | **4 ~ 30 秒** | ✅ | **Apache 2.0** |
| XTTS-v2 | 5.31 | 21 秒 | ✅ | CPML（**非商用**） |
| GPT-SoVITS / CosyVoice / Fish | — | 实际需要 GPU | ✅ | 各异 |

> RTF = 生成时间 ÷ 音频时长。**> 1 表示比实时慢。**
> Kokoro 虽然快 25 倍，但它不能克隆特定人声，只是固定音色的基线。

**关于 Mac**：Apple Silicon 的 GPU（MPS）对这类模型基本用不上——XTTS 在 M 系列上
有明确的 MPS 报错，Audio8 的 CPU 路径走的是 ONNX Runtime，也不碰 Apple GPU。
Mac 的独门优势是统一内存跑本地大模型，而本项目的 LLM 已经走云端。
值得在 Mac 上一试的理由只有一条：**Audio8 是逐 token 串行的，单核性能直接决定速度**，
M 系列的单核比普通 x86 强不少。

---

## 架构

```
浏览器（按住说话）
   │ WebSocket
   ▼
FastAPI 后端 ── 我们写的部分
   ├─ 听：faster-whisper（CPU，CTranslate2，不需要 torch）
   ├─ 想：云端 LLM API（DeepSeek / OpenAI）
   ├─ 说：调本地 Audio8 服务的 OpenAI 兼容接口
   └─ 网页
   │
   ▼
Audio8 ONNX 服务（127.0.0.1:8024）── 官方自带，不用改
   ├─ 音色注册  POST /api/voices/register
   ├─ 合成      POST /api/tts
   ├─ 流式 PCM  POST /api/tts/stream   （NDJSON + base64）
   └─ 取消生成  POST /api/tts/cancel
```

**Audio8 自带完整的本地服务**（含浏览器界面），所以我们的项目
**完全不需要 torch / transformers / coqui-tts**，依赖只有约 200 MB。

---

## 实测数据

测试机：Windows，**12 逻辑核心，无 GPU**，ONNX Runtime 5 线程，Audio8 0.6B INT4。

| 文本 | 长度 | 音频时长 | 整段耗时 | RTF | 首块延迟 |
|---|---|---|---|---|---|
| 你好，我在。 | 6 字 | 1.25 s | 3.98 s | 3.17 | **2.35 s** |
| 好的，我帮你查一下今天的天气，稍等一下。 | 20 字 | 4.32 s | 12.19 s | 2.82 | **2.56 s** |
| 这个问题的答案是这样的：… | 60 字 | 12.96 s | 30.09 s | 2.32 | **2.49 s** |

模型加载 4.57 s、音色注册 10.65 s，都是**一次性**开销。

**线程数扫描**（同一条 20 字文本）：

| 线程 | 4 | **6** | 8 | 12 |
|---|---|---|---|---|
| RTF | 3.31 | **2.52** | 3.01 | 7.05 |

**用物理核数，不要用逻辑核数。** 12 线程比 6 线程慢 2.8 倍（超线程争抢）。

### 结论

- ✅ **首块延迟 2.4~2.6 秒，且与文本长度无关**（流式一出前 12 帧就解码）—— 可以对话
- ❌ **RTF 2.3~3.2，比实时慢** —— 生成速度只有播放速度的 43%，**长回复连续播放必然断流**

### 由此推导出的设计约束

| 对策 | 效果 |
|---|---|
| 让 LLM **先说一句极短的开场白**（"好的，我看看。"） | 2.5 秒就能出声，体感立刻改善 |
| **限制回复长度**到 1~2 句 | 避开断流区间 |
| **先出文字再出声** | 等待期间屏幕有字，感知延迟下降 |
| **分句合成 + 分段播放** | 接受句间停顿，但不断流 |

### 已排除的方案

- **0.1B INT8 版本**：看着小 6 倍，实测**反而更慢**（RTF 2.69~4.12，首块 5.2~5.9 秒）。
  原因是它的 README 里写着 prefill 是**一次一个位置**做的，而 0.6B 是批量 prefill。
  参数少不等于快，瓶颈在串行 prefill 和每步开销。

---

## 目录结构

```
ai-voice/
├── app/                    后端（待开发）
├── web/                    前端（待开发）
├── scripts/
│   ├── setup_audio8.py     克隆第三方仓库 + 下载模型（含代理/镜像自动选路）
│   └── bench_audio8.py     两版模型的速度实测
├── data/
│   ├── fixtures/           测试参考音频（Windows 合成，非真人）
│   └── bench/              实测产物，可试听（不入库）
├── vendor/Audio8_TTS/      第三方仓库 + 模型（不入库，由脚本拉取）
└── models/                 下载缓存（不入库）
```

---

## 安装

### 前置

- **Python 3.11 或更高**
- 能访问 GitHub 与 Hugging Face。如果直连不通，脚本会自动尝试
  **本地代理**（默认探测 Clash 的 `7897` 端口）再降级到 **hf-mirror.com**

### Windows

```powershell
git clone <this-repo> ai-voice
cd ai-voice

# 1) 我们的后端环境（不含 torch）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 第三方仓库 + 模型
.\.venv\Scripts\python.exe scripts\setup_audio8.py

# 3) 实测速度
.\.venv\Scripts\python.exe scripts\bench_audio8.py
```

### macOS

```bash
git clone <this-repo> ai-voice
cd ai-voice

python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/setup_audio8.py
./.venv/bin/python scripts/bench_audio8.py

# 扫一遍线程数找 Mac 上的最优点
./.venv/bin/python scripts/bench_audio8.py --sweep-threads
```

> **Mac 上重点看两个数**：`首块延迟` 和 `RTF`。
> 首块 < 2s 且 RTF < 1 的话，长回复也能连续播放，体验是质变。

### 没有录音时怎么先测速度

`data/fixtures/reference_huihui.wav` 是用 Windows 内置中文语音合成的测试参考音频
（14.7 秒，54 字，非真人），配同名 `.txt` 作为精确转写。它让你**不必先录音**
就能跑通整条链路、量出速度。

要换成自己的声音：录 **0.5~30 秒**清晰朗读，写下**逐字一致的转写文本**，
然后注册（见下）。

---

## 音色注册

参考音频要求：**0.5 ~ 30 秒**，≤ 50 MiB，能被 libsndfile 读取，
自动转单声道 44.1 kHz。**转写文本必须与录音逐字一致**——
噪声大、过长或转写不准都会明显降低相似度。

```bash
# 启动 Audio8 服务（自带浏览器界面，端口 8024）
vendor/Audio8_TTS/onnx_runtime/start_server.sh        # macOS
vendor/Audio8_TTS/onnx_runtime/start_server.ps1       # Windows
```

或直接调 HTTP：

```bash
curl http://127.0.0.1:8024/api/voices/register \
  -F 'audio=@/绝对路径/my_voice.wav' \
  -F 'text=录音的逐字转写文本' \
  -F 'name=me' \
  -F 'overwrite=true'
```

注册产物极小（`codes.npy` 几 KB + `meta.json`），但它是从你的录音提取的
**个人生物特征**，所以已在 `.gitignore` 中排除，不会入库。

---

## 已知的坑

1. **中文 Windows + Audio8 的编码 bug**
   `registration.py` 用 `Path.write_text()` 没指定 `encoding`，在中文 Windows 上
   默认 GBK，写含中文的 `meta.json` 会抛 `UnicodeEncodeError`。
   本项目用 **`PYTHONUTF8=1`**（PEP 540）兜住，脚本会自动重新 exec 自己，
   不改第三方代码。macOS 默认 UTF-8，不受影响。

2. **PowerShell 的 `Set-Content -Encoding UTF8` 会写 BOM**
   读取参考文本时必须用 `utf-8-sig` 剥掉，否则 BOM 会混进文本。

3. **两套运行时不能共用 venv**
   0.6B 要求 `onnxruntime==1.24.4`，0.1B 要求 `>=1.22,<1.24`，版本冲突。
   各自有独立 venv（0.1B 的 `setup.ps1`/`setup.sh` 会自动创建）。

---

## 路线图

- [x] 验证 CPU 上声音克隆可行性，选定 Audio8 0.6B INT4
- [x] 实测速度，得出「首块 2.5s / RTF 2.5」的硬约束
- [ ] 录自己的声音，确认克隆质量
- [ ] 接 faster-whisper：说话 → 文字
- [ ] 接云端 LLM：文字 → 回复
- [ ] 串成实时对话 + 网页界面（按住说话）
- [ ] 打断插话（Audio8 已有 `/api/tts/cancel`）

---

## 许可与伦理

- Audio8 代码与权重：**Apache 2.0**（可商用）
- 克隆声音**须事先取得本人同意**，合成音频在适当场合应明确标注
- 本项目仅用于个人自用
