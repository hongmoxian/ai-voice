"""实测 Audio8 在纯 CPU 上的合成速度，用于回答「交互式对话能不能用」。

要回答的其实只有一个问题：**首块音频延迟**——按下说话到听见第一个字之间等多久。
整段合成的 RTF 再漂亮，首块延迟 20 秒也一样没法对话。

两个版本各有独立 venv（onnxruntime 版本要求冲突），所以要分别用各自的解释器跑：

    # 0.6B INT4（音色需自己注册）
    .venv\\Scripts\\python.exe scripts\\bench_audio8.py --variant 0.6b

    # 0.1B INT8（自带 default 音色，装完即测）
    vendor\\Audio8_TTS\\onnx_runtime_0_1b_int8\\.venv\\Scripts\\python.exe \
        scripts\\bench_audio8.py --variant 0.1b

    # 扫线程数找最优点
    ... --variant 0.1b --sweep-threads

产物（可试听）落在 data/bench/<variant>/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _ensure_utf8() -> None:
    """强制 UTF-8 模式（PEP 540）。

    中文 Windows 的默认编码是 GBK，而 Audio8 的 registration.py 用
    `Path.write_text()` 没指定 encoding，写含中文的 meta.json 时会直接抛
    UnicodeEncodeError。在进程内改 os.environ 是没用的——`sys.flags` 在解释器
    启动时就固定了——所以只能重新 exec 自己。
    """
    if sys.flags.utf8_mode or os.environ.get("PYTHONUTF8") == "1":
        return
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


_ensure_utf8()

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "Audio8_TTS"
FIXTURES = ROOT / "data" / "fixtures"

# 两套运行时是独立实现，图形互不兼容，onnxruntime 版本要求也冲突，各有各的 venv。
VARIANTS: dict[str, dict] = {
    "0.6b": {
        "dir": VENDOR / "onnx_runtime",
        "label": "0.6B INT4",
        "voice": "bench_voice",
        "auto_voice": False,  # 需要自己注册音色
    },
    "0.1b": {
        "dir": VENDOR / "onnx_runtime_0_1b_int8",
        "label": "0.1B INT8",
        "voice": "default",
        "auto_voice": True,  # 模型自带 reference_codes.npy
    },
}

# 三档长度，对应「一句短回复」到「一段长回复」
DEFAULT_TEXTS = [
    "你好，我在。",
    "好的，我帮你查一下今天的天气，稍等一下。",
    "这个问题的答案是这样的：首先需要确认你的网络连接正常，然后再检查一下配置文件"
    "的路径是否正确，如果还有问题可以随时再问我。",
]


def fmt(seconds: float) -> str:
    return f"{seconds:6.2f}s"


def load_reference(reference: Path, text_arg: str | None) -> tuple[bytes, str]:
    if not reference.is_file():
        raise SystemExit(
            f"找不到参考音频：{reference}\n"
            "先用 Windows 内置语音生成一个，或改用 --variant 0.1b（自带音色）"
        )
    text = text_arg
    if text is None:
        sidecar = reference.with_suffix(".txt")
        if not sidecar.is_file():
            raise SystemExit(f"参考文本缺失：{sidecar}（或用 --reference-text 指定）")
        # utf-8-sig：PowerShell 的 Set-Content -Encoding UTF8 会写 BOM，必须剥掉，
        # 否则 BOM 会混进参考文本，后面写 meta.json 时直接炸。
        text = sidecar.read_text(encoding="utf-8-sig")
    # 参考文本必须与录音逐字一致，任何首尾空白都会降低相似度。
    text = text.strip().lstrip("\ufeff").strip()
    if not text:
        raise SystemExit("参考文本不能为空——它必须与录音内容逐字一致")
    return reference.read_bytes(), text


def register_voice(model_dir: Path, voices_dir: Path, fingerprint: str,
                   data: bytes, filename: str, reference_text: str,
                   name: str) -> float:
    """注册音色，返回耗时。encoder 只在注册期间加载。"""
    from arktts_runtime.registration import VoiceRegistration

    registration = VoiceRegistration(model_dir / "registration", voices_dir, fingerprint)
    state = registration.status()
    if not state["available"]:
        raise SystemExit(f"注册功能不可用：{state['reason']}")

    start = time.perf_counter()
    registration.register(data, filename, reference_text, name, overwrite=True)
    return time.perf_counter() - start


def bench_one(runtime, text: str, voice: str, max_new_tokens: int,
              save_path: Path | None) -> dict:
    """测一条文本：整段合成 + 流式首块延迟。"""
    import numpy as np
    import soundfile as sf

    rate = int(runtime.manifest["sample_rate"])

    start = time.perf_counter()
    audio, _codes = runtime.synthesize(text=text, voice=voice,
                                       max_new_tokens=max_new_tokens)
    total = time.perf_counter() - start
    duration = audio.size / rate

    # 流式：只关心第一块出来的时刻，那才是感知延迟
    start = time.perf_counter()
    first_chunk = None
    chunks: list[np.ndarray] = []
    for event in runtime.stream(chunk_frames=12, text=text, voice=voice,
                                max_new_tokens=max_new_tokens):
        if event["type"] == "audio_chunk":
            if first_chunk is None:
                first_chunk = time.perf_counter() - start
            chunks.append(event["audio"])
    stream_total = time.perf_counter() - start

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_path), audio, rate)

    # 流式能否连续播放：生成速度必须 >= 播放速度（RTF <= 1），否则缓冲只会枯竭
    streamed = sum(c.size for c in chunks) / rate
    return {
        "chars": len(text),
        "duration": duration,
        "total": total,
        "rtf": total / duration if duration else float("inf"),
        "first_chunk": first_chunk if first_chunk is not None else float("inf"),
        "stream_total": stream_total,
        "streamed": streamed,
        "sustainable": (stream_total / streamed) <= 1.0 if streamed else False,
    }


def report(text: str, stats: dict) -> None:
    head = text if len(text) <= 16 else text[:16] + "…"
    print(f"  {head:<18} {stats['chars']:>3}字  音频 {fmt(stats['duration'])}"
          f"  整段 {fmt(stats['total'])}  RTF {stats['rtf']:5.2f}"
          f"  首块 {fmt(stats['first_chunk'])}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="0.6b")
    parser.add_argument("--texts", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=5,
                        help="ONNX Runtime 线程数（官方默认 5）")
    parser.add_argument("--sweep-threads", action="store_true",
                        help="扫多个线程数找最优点（耗时成倍增加）")
    parser.add_argument("--reference", type=Path,
                        default=FIXTURES / "reference_huihui.wav")
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args(argv)

    variant = VARIANTS[args.variant]
    onnx_dir: Path = variant["dir"]
    model_dir = onnx_dir / "model"
    voices_dir = onnx_dir / "voices"
    voice = args.voice or variant["voice"]
    out_dir = ROOT / "data" / "bench" / args.variant

    if not (model_dir / "runtime_manifest.json").is_file():
        raise SystemExit(f"模型未下载：{model_dir}\n详见 README")

    # `import arktts_runtime` 要能找到：它是仓库内的包，不在 venv 里
    sys.path.insert(0, str(onnx_dir))

    texts = args.texts or DEFAULT_TEXTS
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Audio8 {variant['label']} · 纯 CPU 合成速度实测")
    print("=" * 78)

    import json
    fingerprint = str(json.loads(
        (model_dir / "runtime_manifest.json").read_text(encoding="utf-8")
    )["model_fingerprint"])

    already = (voices_dir / voice / "codes.npy").is_file()
    if already or variant["auto_voice"]:
        print(f"\n音色 {voice!r} 已就绪，跳过注册")
    else:
        data, reference_text = load_reference(args.reference, args.reference_text)
        print(f"参考音频  {args.reference.name}  {len(data) / 1024:.0f} KiB")
        elapsed = register_voice(model_dir, voices_dir, fingerprint, data,
                                 args.reference.name, reference_text, voice)
        print(f"\n音色注册完成  用时 {elapsed:.2f}s  -> {voices_dir / voice}")

    thread_options = [args.threads]
    if args.sweep_threads:
        candidates = {max(1, min(t, os.cpu_count() or 1))
                      for t in (2, 4, 6, 8, 12, 16)}
        thread_options = sorted(candidates)

    print(f"\nCPU 逻辑核心 {os.cpu_count()}")

    from arktts_runtime.runtime import ArkTtsRuntime

    for threads in thread_options:
        print(f"\n{'-' * 78}")
        print(f"ONNX Runtime 线程数 = {threads}")
        print(f"{'-' * 78}")

        load_start = time.perf_counter()
        runtime = ArkTtsRuntime(model_dir, voices_dir, None, None, threads)
        print(f"  模型加载 {time.perf_counter() - load_start:.2f}s\n")
        print(f"  {'文本':<18} {'长度':>5}  {'音频时长':>10}  {'整段耗时':>9}"
              f"  {'RTF':>6}  {'首块延迟':>9}")

        for index, text in enumerate(texts):
            stats = bench_one(runtime, text, voice, args.max_new_tokens,
                              out_dir / f"t{threads}_{index}.wav")
            report(text, stats)
            if not stats["sustainable"]:
                print(f"  {'':<18} └ 流式生成速度 {1 / stats['rtf']:.2f}x 播放速度"
                      f" -> 连续播放会断流")
        del runtime

    print(f"\n{'=' * 78}")
    print(f"试听文件：{out_dir}")
    print("=" * 78)
    print("\n怎么读这些数字：")
    print("  首块延迟  < 2s  -> 对话体验自然")
    print("            2-4s  -> 能接受，配合「先出文字再出声」更顺")
    print("            > 5s  -> 交互式对话基本不可用")
    print("  RTF       < 1   -> 比实时快，流式可以连续播放")
    print("            > 1   -> 比实时慢，长回复必然断流，只能分段播")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
