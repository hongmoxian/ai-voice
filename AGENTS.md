# ai-voice — 本地声音克隆语音助手

用你自己的声音说话的 AI 助手。声音克隆跑在本地 CPU，只有对话大脑走云端 API。

**状态**：技术路线已验证，应用层未开始。详细实测数据见 @README.md。

---

## 技术选型结论（不要再重新评估）

选型过程本身就是这个项目最重要的产出。**被排除的方案同样重要**，写在这里避免重复调研。

### 为什么是 Audio8 0.6B INT4

| 候选 | CPU 实测 RTF | 能克隆 | 许可 | 结论 |
|---|---|---|---|---|
| **Audio8 0.6B INT4** | **2.32 ~ 3.17** | ✅ | Apache-2.0 | **采用** |
| Audio8 0.1B INT8 | 2.69 ~ 4.12 | ✅ | Apache-2.0 | ❌ 见下 |
| XTTS-v2 | 5.31 | ✅ | CPML 非商用 | ❌ 慢 2.3 倍 + 不能商用 |
| Kokoro-82M | 0.12 | ❌ 固定音色 | Apache-2.0 | ❌ 不能克隆人声 |
| GPT-SoVITS / CosyVoice / Fish | 需 GPU | ✅ | 各异 | ❌ 本机无 GPU |

- **XTTS 排除原因**：CPU 上 RTF 5.31（21 秒一句话），且 M 系列上有明确 MPS 报错，
  Mac 也救不了；许可还是非商用
- **0.1B 排除原因（反直觉，重点）**：参数少 6 倍，实测**反而更慢**，首块延迟还差 2.3 倍。
  根因是它的 prefill 是**一次一个位置**做的（见其 README），而 0.6B 是批量 prefill。
  **参数少 ≠ 快，瓶颈在串行 prefill 和每步开销。**
  想重新启用它之前，先读 `vendor/Audio8_TTS/onnx_runtime_0_1b_int8/README.md` 的 prefill 说明
- **Mac 不作加速方案**：Apple GPU（MPS）对这些模型基本不可用。Mac 的价值在统一内存跑
  本地大模型，而 LLM 已走云端。唯一值得一试的理由是 **Audio8 逐 token 串行，
  单核性能直接决定速度**，M 系列单核更强——但官方没公布速度，只能实测

### 由此推导出的硬约束（决定应用层设计）

```
首块延迟  2.4 ~ 2.6 秒，与文本长度无关（流式一出前 12 帧就解码）
RTF       2.32 ~ 3.17，比实时慢 → 生成速度只有播放速度的 43%
```

**因此长回复无法连续播放，必然断流。** 应用层必须做：

1. 让 LLM **先说一句极短开场白**（"好的，我看看。"）→ 2.5 秒出声，体感质变
2. **限制回复长度**到 1~2 句
3. **先出文字再出声**
4. **分句合成 + 分段播放**

**ONNX Runtime 线程数用 6（物理核），不要用 12（逻辑核）**——实测 12 线程 RTF 7.05，
比 6 线程慢 2.8 倍。

---

## 架构

```
浏览器 ──WebSocket──> FastAPI 后端（我们写的）
                        ├─ 听：faster-whisper
                        ├─ 想：云端 LLM API
                        └─ 说：调 Audio8 本地服务的 OpenAI 兼容接口
                                   │
                                   ▼
                        Audio8 ONNX 服务 (127.0.0.1:8024)  官方自带，不要改
                        注册 /api/voices/register · 合成 /api/tts
                        流式 /api/tts/stream · 取消 /api/tts/cancel
```

**关键设计：不在我们进程里嵌 TTS。** Audio8 自带完整的本地 HTTP 服务
（含浏览器界面、流式 PCM、取消生成），我们只当客户端。好处：
项目**完全不需要 torch / transformers / coqui-tts**，依赖从 4-6 GB 降到约 200 MB，
而且流式和打断插话是白送的。

---

## 目录

```
app/                  后端（待开发）
web/                  前端（待开发）
scripts/
  setup_audio8.py     克隆第三方仓库（钉 commit）+ 下载模型，含代理/镜像自动选路
  bench_audio8.py     速度实测，产物可试听
data/fixtures/        测试参考音频（Windows 合成，非真人）
data/bench/           实测产物（不入库）
vendor/Audio8_TTS/    第三方仓库 + 模型（不入库，由脚本拉取）
```

---

## 常用命令

```bash
# 安装（Windows 用 .\.venv\Scripts\python.exe 代替 ./.venv/bin/python）
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/setup_audio8.py

# 实测速度（首块延迟 + RTF 是关键指标）
./.venv/bin/python scripts/bench_audio8.py
./.venv/bin/python scripts/bench_audio8.py --sweep-threads
```

---

## 必须知道的坑

1. **中文 Windows 上 Audio8 会崩**：它的 `registration.py` 用 `Path.write_text()`
   没指定 `encoding`，默认 GBK，写含中文的 `meta.json` 抛 `UnicodeEncodeError`。
   **对策：脚本里 `os.execv` 自己重开一次并设 `PYTHONUTF8=1`**（PEP 540）。
   **不要改第三方源码**——重新拉取会丢。macOS 默认 UTF-8，不受影响
2. **PowerShell 的 `Set-Content -Encoding UTF8` 写 BOM**，读参考文本必须用
   `utf-8-sig`，否则 BOM 混进文本导致注册失败
3. **两套运行时不能共用 venv**：0.6B 要 `onnxruntime==1.24.4`，0.1B 要 `>=1.22,<1.24`
4. **先量后买**：任何"换硬件/换模型"的决定，都先在现有机器上实测出数字

---

## 隐私红线

`voices/`、`data/recordings/`、`*.local.wav` 已在 `.gitignore` 中排除。
音色档案只有几 KB，但它是**从录音提取的个人生物特征**，绝不能入库。
克隆声音须取得本人同意，合成音频在适当场合应明确标注。

---

## 路线图

- [x] 验证 CPU 声音克隆可行性，选定方案并实测出硬约束
- [x] 同步到 GitHub
- [ ] 在 Mac 上复测（重点看 RTF 是否 < 1）
- [ ] 录用户自己的声音，确认克隆质量
- [ ] 接 faster-whisper：说话 → 文字
- [ ] 接云端 LLM：文字 → 回复
- [ ] 串成实时对话 + 网页界面（按住说话）
- [ ] 打断插话（Audio8 已有 `/api/tts/cancel`）
