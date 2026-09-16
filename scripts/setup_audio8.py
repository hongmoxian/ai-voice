"""一键准备 Audio8：克隆第三方仓库 + 下载 ONNX INT4 模型。

官方说明：模型下完后运行时就**不需要** PyTorch / Transformers / huggingface_hub，
所以这个脚本只在首次安装时跑一次。

    python scripts/setup_audio8.py                # 自动选路：直连 -> 本地代理 -> 镜像
    python scripts/setup_audio8.py --proxy        # 强制走本地代理（默认探测 Clash 端口）
    python scripts/setup_audio8.py --mirror       # 强制走 hf-mirror.com
    python scripts/setup_audio8.py --check        # 只校验，不下载

Windows 和 macOS 通用（官方只在 macOS arm64 上测过，但核心就是 onnxruntime）。
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO_ID = "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4"

# 钉住上游 commit：0.6B 和 0.1B 的 ONNX 图形互不兼容，上游改动可能直接
# 破坏 bench/应用层的调用，所以两台机器必须拿到同一份运行时。
REPO_URL = "https://github.com/Edge0-AI/Audio8_TTS.git"
REPO_COMMIT = "07e40f5d0b03fc473635ef378654bfb581027ac3"

ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = ROOT / "vendor" / "Audio8_TTS"
ONNX_DIR = REPO_DIR / "onnx_runtime"
MODEL_DIR = ONNX_DIR / "model"
HF_CACHE = ROOT / "models" / "hf"

# 本地代理常见端口，Clash Verge 默认是 7897。
PROXY_CANDIDATES = ("127.0.0.1:7897", "127.0.0.1:7890", "127.0.0.1:7891",
                    "127.0.0.1:10809", "127.0.0.1:8080")
MIRROR_ENDPOINT = "https://hf-mirror.com"


def _tcp_ok(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _detect_proxy() -> str | None:
    for candidate in PROXY_CANDIDATES:
        host, _, port = candidate.partition(":")
        if _tcp_ok(host, int(port), timeout=0.4):
            return f"http://{candidate}"
    return None


def choose_route(force_proxy: bool, force_mirror: bool) -> str:
    """决定怎么出去，并把结果写进环境变量。

    关键：`HF_ENDPOINT` 是 huggingface_hub 在 **import 时**读的模块级常量，所以
    必须在 import 之前定好，不能失败了再改。因此这里先探测、再导入。
    """
    if force_mirror:
        os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
        return f"镜像站 {MIRROR_ENDPOINT}"

    if force_proxy:
        proxy = _detect_proxy()
        if not proxy:
            print("[!] 没探测到本地代理，改走镜像站")
            os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
            return f"镜像站 {MIRROR_ENDPOINT}"
        os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = proxy
        return f"代理 {proxy}"

    if _tcp_ok("huggingface.co", 443):
        return "直连 huggingface.co"

    proxy = _detect_proxy()
    if proxy:
        os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = proxy
        return f"代理 {proxy}（huggingface.co 直连不通，自动探测到本地代理）"

    os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
    return f"镜像站 {MIRROR_ENDPOINT}（没有可用代理）"

# 官方 README 列出的必需文件（.data 是同名 onnx 的外部权重，可能有也可能内嵌）
REQUIRED = [
    "slow_ar_int4.onnx",
    "fast_ar_int4.onnx",
    "codec_decoder_fp16.onnx",
    "runtime_manifest.json",
    "tokenizer/tokenizer.json",
    "registration/codec_encoder_fp16.onnx",
    "registration/registration_manifest.json",
]


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _git(args: list[str]) -> int:
    try:
        return subprocess.run(["git", *args], check=False).returncode
    except FileNotFoundError:
        print("[X] 找不到 git，请先安装")
        return 1


def ensure_repo() -> int:
    """克隆 Audio8 仓库并切到钉住的 commit。"""
    marker = ONNX_DIR / "arktts_runtime" / "runtime.py"
    if marker.is_file():
        print(f"[OK] 第三方仓库已就绪：{REPO_DIR}")
        return 0
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"克隆 {REPO_URL}")
    if _git(["clone", "--quiet", REPO_URL, str(REPO_DIR)]) != 0:
        print("[X] 克隆失败。检查网络，或手动 clone 到 vendor/Audio8_TTS")
        return 1
    if REPO_COMMIT:
        if _git(["-C", str(REPO_DIR), "checkout", "--quiet", REPO_COMMIT]) != 0:
            print(f"[!] 无法切到 {REPO_COMMIT[:8]}，继续使用默认分支（可能有兼容风险）")
        else:
            print(f"[OK] 已切到 {REPO_COMMIT[:8]}")
    return 0


def verify(verbose: bool = True) -> bool:
    if not MODEL_DIR.is_dir():
        print(f"[X] 模型目录不存在：{MODEL_DIR}")
        return False
    missing = [rel for rel in REQUIRED if not (MODEL_DIR / rel).is_file()]
    for rel in REQUIRED:
        mark = "  " if (MODEL_DIR / rel).is_file() else "X "
        if verbose:
            print(f"{mark}{rel}")
    if missing:
        print(f"\n[X] 缺少 {len(missing)} 个必需文件")
        return False
    print(f"\n[OK] 模型完整，共 {human(dir_size(MODEL_DIR))}")
    return True


def download(force_proxy: bool, force_mirror: bool) -> int:
    # 缓存落在项目盘里：C 盘只剩十几个 GB，HF 默认会往那儿塞近 1GB。
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    # 出口必须在 import huggingface_hub 之前定好（HF_ENDPOINT 是 import 期常量）。
    route = choose_route(force_proxy, force_mirror)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[X] 需要 huggingface_hub：pip install 'huggingface_hub[cli]'")
        return 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载 {REPO_ID}")
    print(f"  出口：{route}")
    print(f"  ->  {MODEL_DIR}")
    print("  （约 572 MiB 在线权重，含注册编码器共约 968 MiB）\n")
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=str(MODEL_DIR))
    except Exception as exc:  # noqa: BLE001 - 网络/鉴权各种错都直接报给用户
        print(f"\n[X] 下载失败：{type(exc).__name__}: {exc}")
        print("    换一条路再试：--proxy（本地代理）或 --mirror（hf-mirror.com）")
        return 1
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验，不下载")
    parser.add_argument("--proxy", action="store_true", help="强制走本地代理")
    parser.add_argument("--mirror", action="store_true", help="强制走 hf-mirror.com")
    args = parser.parse_args(argv)

    if not args.check:
        if ensure_repo() != 0:
            return 1
        code = download(args.proxy, args.mirror)
        if code != 0:
            return code
    ok = verify()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
