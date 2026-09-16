# ai-voice — 本地声音克隆语音助手

用你自己的声音说话的 AI 助手。目标运行环境是 **Apple Silicon Mac**。

**状态**：方案调研完成，**等待在 Mac 上实测**。
实测数据、调研来源、架构都在 @README.md，这里只写**给 agent 的约束与决策依据**。

---

## 当前决策状态（会变，改之前先看这里）

```
运行目标：Apple Silicon Mac（不是 Windows）
主推方案：mlx-audio + Qwen3-TTS —— 待实测确认
备选方案：GPT-SoVITS + MLX —— 唯一有公开 M 系列数据的方案
Windows 回退：Audio8 0.6B INT4（已实测，能跑但长回复断流）
```

**下一步是在 Mac 上实测 mlx-audio 的克隆模型速度。**
在拿到实测数字之前，不要开始写应用层（后端/前端/ASR/LLM 串联）——
因为 RTF 是否 < 1 会决定完全不同的交互设计。

---

## 选型决策与理由（不要再重复调研）

### 关键事实

**Apple Silicon + MLX 比纯 CPU 快 14~20 倍。** 这是整个项目的分水岭。

| | Audio8 0.6B（Windows CPU，实测） | GPT-SoVITS（M4 + MLX，公开基准） |
|---|---|---|
| RTF | 2.32 ~ 3.17 | **0.16** |
| 首块延迟 | 2.4 ~ 2.6 s | **0.85 s** |
| 连续流式 | ❌ 断流 | ✅ 6.25× 实时 |

### 曾经犯过的错（**写下来防止重犯**）

1. **断言「Apple GPU 对这些模型用不上」** —— 错。该结论**只对 ONNX Runtime /
   PyTorch-MPS 路线成立**（XTTS 还有明确的 MPS 报错），**对 MLX 不成立**。
   MLX 是 Apple 原生的真·GPU 路径，生态已成熟。
2. **把 GPT-SoVITS 列为「需要 GPU，放弃」** —— 错。那是在推理 x86 CPU + PyTorch 的情况。
   **在 M4 上用 MLX，它比实时快 6.25 倍。**

**教训：排除一个方案前，要确认是把「这条路走不通」还是「这类模型都走不通」搞混了。
换框架/换后端要重新验证，不能沿用旧结论。**

### 已确认排除（不必再试）

| 方案 | 原因 |
|---|---|
| XTTS-v2 | CPU RTF 5.31（21 秒/句）；M 系列 MPS 报错；许可 CPML 非商用 |
| Audio8 **0.1B** INT8 | **参数少 6 倍反而更慢**（RTF 2.69~4.12，首块 5.2~5.9 s）。根因：其 prefill 一次一个位置，0.6B 是批量 prefill。**参数少 ≠ 快** |
| Kokoro-82M | 极快（M4 Max RTF 22×）但**不能克隆**，只能选预设音色 |
| Audio8 0.6B 作为**最终**方案 | 降级为 **Windows 回退**，Mac 上比 MLX 慢一个数量级 |

### 候选方案

- **`mlx-audio`（MIT，首选）**：13 个克隆模型（Qwen3-TTS 中英日韩 + 克隆 + 流式、
  OmniVoice 646 语言、Chatterbox、VoxCPM2…），统一 `ref_audio`+`ref_text` API，
  **自带 OpenAI 兼容服务，且 TTS 与 STT 都在同一个服务里**（省掉单独的 faster-whisper）。
  ⚠️ **克隆模型无公开速度数据，必须实测**
- **`GPT-SoVITS` + MLX（备选）**：唯一有公开 M 系列数据；中文效果最好；
  但集成要拖一整套训练/推理工具链。⚠️ 许可证待确认
- **`OminiX-MLX`**：GPT-SoVITS 的纯 Rust + MLX 移植，生态小，第三备选

### 设计约束的来源

当前最优选（Audio8，Windows）的 **RTF 2.32~3.17 > 1**，意味着**长回复无法连续播放**。
若 Mac 实测能到 **RTF < 1**，则「限制回复长度 / 短开场白 / 分句播放」这些妥协
**全都不需要**，可以直接做流畅流式对话。
**所以不要在 Mac 实测出结果前，把妥协设计写进代码。**

---

## 环境事实

### 本机（Windows，开发用）

- **无 NVIDIA GPU**（`nvidia-smi` 不存在，torch 是 CPU 版）
- CPU 12 逻辑核（约 6 物理核）。**ONNX Runtime 用物理核数**——
  实测 12 线程比 6 线程慢 2.8 倍（超线程争抢）
- **`D:\` 根目录不可写**（ACL 只给 `Authenticated Users` 只读）→ 用 `E:\` 或 D 盘已有可写子目录
- 磁盘紧张：E 盘约 23 GB / C 盘约 12 GB / D 盘约 219 GB。
  **HF 与 pip 缓存默认写 C 盘，要显式改到项目盘**（`HF_HOME` / `PIP_CACHE_DIR`）
- 主解释器：`D:\keyan gongju\miniconda\python.exe`（**路径含空格，要引号**）

### 网络（中国大陆）

- **`huggingface.co` 直连不通** → 降级：本地代理 → `https://hf-mirror.com`
- `github.com` 通，但 **`api.github.com` 不稳**（TLS 超时，需走代理）
- 本地代理 **Clash Verge 在 `127.0.0.1:7897`**
- **`gh auth status` 会因 keyring 超时误报失败**，但不影响 `gh api` / `gh repo create`
- **GitHub 推送走 SSH**（`git@github.com`，密钥已配好），HTTPS 无 credential helper

### 目标机（Mac）

- ⚠️ **具体型号与内存未知——动手前必须问清楚。**
  M1 与 M4 差距约 2 倍（Kokoro：M1 Max RTF 11.9× vs M4 Max 22×），
  内存决定可用模型规模（1.7B/4B 模型对内存要求明显更高）

---

## 目录与命令

```
app/                  后端（待开发）
web/                  前端（待开发）
scripts/
  setup_audio8.py     克隆第三方仓库（钉 commit）+ 下载模型，含代理/镜像选路
  bench_audio8.py     速度实测
data/fixtures/        测试参考音频（Windows 合成，非真人）
vendor/Audio8_TTS/    第三方仓库 + 模型（不入库）
```

```bash
# Windows（备选路线：跑通链路）
.\.venv\Scripts\python.exe scripts\setup_audio8.py
.\.venv\Scripts\python.exe scripts\bench_audio8.py --sweep-threads
```

---

## 必须知道的坑

1. **中文 Windows 上 Audio8 会崩**：`registration.py` 用 `Path.write_text()` 未指定
   `encoding`，默认 GBK，写含中文的 `meta.json` 抛 `UnicodeEncodeError`。
   **对策：脚本里 `os.execv` 重开自己并设 `PYTHONUTF8=1`**（PEP 540）。
   **不要改第三方源码**——重新拉取会丢。macOS 默认 UTF-8，不受影响
2. **PowerShell `Set-Content -Encoding UTF8` 写 BOM**，读参考文本必须用 `utf-8-sig`
3. **两套 Audio8 运行时不能共用 venv**：0.6B 要 `onnxruntime==1.24.4`，0.1B 要 `<1.24`
4. **第三方仓库钉住 commit**（见 `scripts/setup_audio8.py` 里的 `REPO_COMMIT`），别跟 main
5. **先量后买**：任何"换硬件/换模型"的决定，先在现有机器上实测出数字

---

## 隐私红线

`voices/`、`data/recordings/`、`*.local.wav` 已在 `.gitignore` 中排除。
音色档案只有几 KB，但它是**从录音提取的个人生物特征**，绝不能入库或分享。
克隆声音须取得本人同意，合成音频在适当场合应明确标注。

> 本仓库**已转为公开**。提交前检查：不要提交音色档案、用户录音、API Key、.env。
> `data/fixtures/reference_huihui.wav` 是 Windows 合成的非真人音频，可以公开。
