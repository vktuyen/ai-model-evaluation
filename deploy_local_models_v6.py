#!/usr/bin/env python3
"""
Deploy all 3 local LLM endpoints — ONE reusable script, mobile-faithful (v6).

Each model is served on the SAME runtime + quantization the Persona mobile app
uses, so the eval predicts on-device behaviour (the parity concern Mustafa
raised):

  Gemma 4 E2B  -> port 8011  via LiteRT-LM   (.litertlm, litert-community)
  Qwen3.5 2B   -> port 8002  via llama.cpp    (GGUF Q4_K_M, unsloth)
  Llama 3.2 3B -> port 8003  via llama.cpp    (GGUF Q4_K_M, unsloth)

WHY MIXED RUNTIMES
------------------
  * Mobile Gemma runs on LiteRT-LM (`flutter_gemma_litertlm`) on the exact
    `.litertlm` bundle. A llama.cpp/GGUF conversion is a different runtime AND
    quantization, so it would NOT reflect on-device Gemma. -> LiteRT-LM here.
  * Mobile Qwen (and Llama 3.2 3B, if shipped) runs on llama.cpp (`fllama`)
    with the SAME unsloth GGUF Q4_K_M this serves. -> llama.cpp here.

This supersedes v4/v5 (llama.cpp-only). Ports + served names match
rephrase-eval/promptfooconfig.yaml, so the eval runs unchanged.

Usage:
  python3 deploy_local_models_v6.py --restart --rebuild --wait-minutes 30
  python3 deploy_local_models_v6.py --status
  python3 deploy_local_models_v6.py --stop
  python3 deploy_local_models_v6.py --restart --models gemma --wait-minutes 20

BACKENDS (verified on the box, litert-lm 0.14.0)
------------------------------------------------
  * llama.cpp (qwen/llama): GPU via -ngl (CUDA build with --rebuild). Flag:
    --gpu-layers N (0 = CPU).
  * LiteRT-LM (gemma): `litert-lm serve` has NO backend flag — it uses the
    backend baked into the `.litertlm` bundle. The litert-community Gemma 4
    bundle is constrained to CPU (`section_backend_constraint: cpu`), which is
    EXACTLY what the mobile app runs on Android (its WebGPU/Dawn GPU path fails
    on Android, so Android Gemma is CPU too — see the app's ai_config.dart).
    So CPU here is the faithful match, not a fallback. There is nothing to
    "probe": the server runs the model's configured (CPU) backend.

NOTES
-----
  * `litert-lm serve` hosts ALL imported models and routes by the request's
    `model` field; we import the bundle under the id `gemma-4-e2b` so the
    eval's `model: gemma-4-e2b` resolves. It has no --model flag.
  * First LiteRT-LM request is a cold load (~40s for Gemma 4 E2B on CPU);
    later requests reuse the loaded model.
  * The litert-lm CLI is installed into a dedicated venv (LITERT_VENV) because
    it needs its own deps; the system Python is left untouched.

Watch logs:
  tail -F ~/local_model_server/logs/*.log
"""

from __future__ import annotations

import argparse
import enum
import os
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

HOST = "0.0.0.0"
PUBLIC_HOST = "10.30.11.110"
GPU_LAYERS = 99  # llama.cpp -ngl: layers to offload to GPU. 99 = all (2-4B fit).

APP_DIR = Path.home() / "local_model_server"
LOG_DIR = APP_DIR / "logs"
MODEL_DIR = APP_DIR / "models"
LLAMA_CPP_DIR = APP_DIR / "llama.cpp"
LLAMA_SERVER = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"

# Dedicated venv for the litert-lm CLI (keeps system Python clean; system
# Python may also be PEP-668 externally-managed).
LITERT_VENV = Path.home() / "litert-venv"
LITERT_BIN = LITERT_VENV / "bin" / "litert-lm"


class Runtime(enum.Enum):
    LLAMA_CPP = "llama.cpp"   # GGUF via llama-server
    LITERT_LM = "litert-lm"   # .litertlm via `litert-lm serve`


@dataclass(frozen=True)
class ModelSpec:
    key: str
    session: str
    port: int
    served_name: str
    runtime: Runtime
    # llama.cpp fields:
    gguf_url: str = ""
    gguf_path: Path | None = None
    context: int = 4096
    # LiteRT-LM fields:
    hf_repo: str = ""
    hf_file: str = ""

    def log_path(self) -> Path:
        return LOG_DIR / f"{self.session}.log"


MODELS: dict[str, ModelSpec] = {
    # ---- Gemma: LiteRT-LM (.litertlm) — matches mobile runtime + quantization
    "gemma": ModelSpec(
        key="gemma",
        session="gemma-litertlm",
        port=8011,
        served_name="gemma-4-e2b",   # == promptfoo provider id + OpenAI model field
        runtime=Runtime.LITERT_LM,
        hf_repo="litert-community/gemma-4-E2B-it-litert-lm",
        hf_file="gemma-4-E2B-it.litertlm",
    ),
    # ---- Qwen: llama.cpp GGUF Q4_K_M — same as mobile (fllama)
    "qwen": ModelSpec(
        key="qwen",
        session="qwen35-2b",
        port=8002,
        served_name="qwen3.5-2b",
        runtime=Runtime.LLAMA_CPP,
        gguf_url=(
            "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF"
            "/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"
        ),
        gguf_path=MODEL_DIR / "qwen35_2b" / "Qwen3.5-2B-Q4_K_M.gguf",
    ),
    # ---- Llama: llama.cpp GGUF Q4_K_M — Meta's on-device 3B; same engine
    "llama": ModelSpec(
        key="llama",
        session="llama32-3b",
        port=8003,
        served_name="llama-3.2-3b-instruct",
        runtime=Runtime.LLAMA_CPP,
        gguf_url=(
            "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF"
            "/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        ),
        gguf_path=MODEL_DIR / "llama32_3b" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
}


def run(cmd: list[str], *, check: bool = True, capture: bool = False, cwd: Path | None = None):
    print("$ " + " ".join(shlex.quote(x) for x in cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def exists(command: str) -> bool:
    return shutil.which(command) is not None


def apt_install(packages: list[str]) -> None:
    missing = [p for p in packages if not exists(p)]
    if not missing:
        return
    if not exists("apt-get"):
        print(f"Missing tools: {', '.join(missing)}; install them manually.")
        return
    sudo = ["sudo"] if exists("sudo") and os.geteuid() != 0 else []
    run(sudo + ["apt-get", "update"], check=False)
    run(sudo + ["apt-get", "install", "-y"] + missing, check=False)


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def ensure_tmux() -> None:
    if not exists("tmux"):
        apt_install(["tmux"])
    if not exists("tmux"):
        raise RuntimeError("tmux is required.")


def gpu_available() -> bool:
    if not exists("nvidia-smi"):
        return False
    return run(["nvidia-smi", "-L"], check=False, capture=True).returncode == 0


# =============================== tmux / ports ================================

def tmux_session_exists(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session], check=False, capture=True).returncode == 0


def stop_session(session: str) -> None:
    if tmux_session_exists(session):
        run(["tmux", "kill-session", "-t", session], check=False)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port(port: int, timeout: int = 15) -> bool:
    if not port_open(port):
        return True
    print(f"Port {port} still in use; freeing it...")
    # Both llama-server and `litert-lm serve` carry `--port <port>` in argv;
    # this precise pattern avoids matching unrelated processes.
    run(["pkill", "-f", f"--port {port}"], check=False)
    end = time.time() + timeout
    while time.time() < end:
        if not port_open(port):
            return True
        time.sleep(1)
    return not port_open(port)


def tmux_start(session: str, command: list[str], log_path: Path) -> None:
    """Start `command` detached in tmux, teeing combined output to log_path."""
    wrapped = ("set -e; " + " ".join(shlex.quote(x) for x in command)
               + f" 2>&1 | tee -a {shlex.quote(str(log_path))}")
    run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", wrapped])


# ============================ llama.cpp backend ==============================

def ensure_llama_cpp(rebuild: bool = False) -> Path:
    if LLAMA_SERVER.exists() and not rebuild:
        return LLAMA_SERVER

    apt_install(["git", "cmake", "make", "g++", "curl"])
    if not exists("git") or not exists("cmake"):
        raise RuntimeError("git and cmake are required to build llama.cpp.")

    if not LLAMA_CPP_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_CPP_DIR)])

    build_dir = LLAMA_CPP_DIR / "build"
    if rebuild and build_dir.exists():
        print("Removing existing build dir to force a clean recompile...")
        shutil.rmtree(build_dir)

    have_gpu = gpu_available()
    use_cuda = have_gpu and exists("nvcc")
    if have_gpu and not exists("nvcc"):
        print("WARNING: GPU detected but 'nvcc' (CUDA toolkit) not found; building CPU-only.")
        print("         Install nvidia-cuda-toolkit and re-run with --rebuild for GPU.")

    cmake_cfg = ["cmake", "-S", str(LLAMA_CPP_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"]
    if use_cuda:
        cmake_cfg.append("-DGGML_CUDA=ON")
        print("Building llama.cpp WITH CUDA (GPU) support.")
    else:
        print("Building llama.cpp CPU-only.")

    jobs = str(max(1, (os.cpu_count() or 2) - 1))
    run(cmake_cfg)
    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j", jobs, "--target", "llama-server"])

    if not LLAMA_SERVER.exists():
        raise RuntimeError("llama.cpp build failed: llama-server not found.")
    return LLAMA_SERVER


def ensure_gguf_file(model: ModelSpec, redownload: bool) -> Path:
    assert model.gguf_path is not None
    model.gguf_path.parent.mkdir(parents=True, exist_ok=True)

    if redownload and model.gguf_path.exists():
        model.gguf_path.unlink()

    if model.gguf_path.exists() and model.gguf_path.stat().st_size > 100 * 1024 * 1024:
        return model.gguf_path

    if not exists("curl"):
        apt_install(["curl"])
    if not exists("curl"):
        raise RuntimeError("curl is required to download models.")

    run(["curl", "-L", "--fail", "--continue-at", "-", "--output",
         str(model.gguf_path), model.gguf_url])
    return model.gguf_path


def start_llama_cpp(model: ModelSpec, llama_server: Path, redownload: bool, gpu_layers: int) -> None:
    gguf_path = ensure_gguf_file(model, redownload)
    command = [
        str(llama_server),
        "-m", str(gguf_path),
        "--host", HOST,
        "--port", str(model.port),
        "-c", str(model.context),
        "--alias", model.served_name,
        "-ngl", str(gpu_layers),   # 99 = all layers on GPU, 0 = CPU only
        "--jinja",                 # use the model's chat template
    ]
    tmux_start(model.session, command, model.log_path())
    print(f"Started (llama.cpp): {model.session} (-ngl {gpu_layers}) on {model.port}")


# ============================= LiteRT-LM backend =============================

def ensure_litert_lm() -> str:
    """Ensure the litert-lm CLI exists in its venv; return the binary path."""
    if LITERT_BIN.exists():
        return str(LITERT_BIN)
    if not LITERT_VENV.exists():
        print(f"Creating venv at {LITERT_VENV} ...")
        run(["python3", "-m", "venv", str(LITERT_VENV)])
    pip = str(LITERT_VENV / "bin" / "pip")
    print("Installing litert-lm into the venv (needs Python 3.10+)...")
    run([pip, "install", "--upgrade", "pip"], check=False)
    run([pip, "install", "--upgrade", "litert-lm"])
    if not LITERT_BIN.exists():
        raise RuntimeError(f"litert-lm not found at {LITERT_BIN} after install.")
    return str(LITERT_BIN)


def litert_model_imported(cli: str, model_id: str) -> bool:
    res = run([cli, "list"], check=False, capture=True)
    return bool(res.stdout) and any(
        line.split()[:1] == [model_id] for line in res.stdout.splitlines()
    )


def ensure_litert_imported(cli: str, model: ModelSpec, reimport: bool) -> None:
    """Register the HF `.litertlm` bundle under served_name in the local cache.

    Real CLI signature (litert-lm 0.14.0):
        litert-lm import --from-huggingface-repo <repo> <FILE.litertlm> <MODEL_ID>
    """
    if reimport:
        run([cli, "delete", model.served_name], check=False)
    elif litert_model_imported(cli, model.served_name):
        print(f"Model '{model.served_name}' already imported; skipping download.")
        return
    print(f"Importing {model.hf_repo}/{model.hf_file} as '{model.served_name}' "
          "(first run downloads the bundle ~2.4 GB)...")
    run([cli, "import", "--from-huggingface-repo", model.hf_repo,
         model.hf_file, model.served_name])


def start_litert_lm(cli: str, model: ModelSpec, reimport: bool) -> None:
    ensure_litert_imported(cli, model, reimport)
    # `serve` takes NO model id and NO backend flag: it hosts all imported
    # models and routes by the request's `model` field, on the backend baked
    # into the bundle (CPU for litert-community Gemma 4 = the Android path).
    command = [cli, "serve", "--host", HOST, "--port", str(model.port), "--verbose"]
    tmux_start(model.session, command, model.log_path())
    print(f"Started (LiteRT-LM, CPU per bundle): {model.session} on {model.port}")


# ================================ orchestration ==============================

def start_model(model: ModelSpec, *, restart: bool, redownload: bool, reimport: bool,
                gpu_layers: int, llama_server: Path | None, litert_cli: str | None) -> None:
    if tmux_session_exists(model.session):
        if restart:
            stop_session(model.session)
        else:
            print(f"{model.session} already exists; use --restart to recreate it.")
            return

    if not free_port(model.port):
        print(f"WARNING: port {model.port} still in use; {model.key} may fail to bind.")

    if model.runtime is Runtime.LLAMA_CPP:
        assert llama_server is not None
        start_llama_cpp(model, llama_server, redownload, gpu_layers)
    else:
        assert litert_cli is not None
        start_litert_lm(litert_cli, model, reimport)
    print(f"Log: {model.log_path()}")


def wait_ready(models: list[ModelSpec], minutes: int) -> None:
    if minutes <= 0:
        return
    end = time.time() + minutes * 60
    pending = {m.key: m for m in models}
    print(f"Waiting up to {minutes} minutes for endpoints...")
    while pending and time.time() < end:
        for key, model in list(pending.items()):
            if port_open(model.port):
                print(f"Ready: {model.key} on port {model.port}")
                del pending[key]
        if pending:
            time.sleep(5)
    for model in pending.values():
        print(f"Not ready: {model.key}. Check: {model.log_path()}")


def print_status() -> None:
    print("\nGPU:")
    if gpu_available():
        run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv"], check=False)
    else:
        print("  no NVIDIA GPU detected")
    print("\nSessions:")
    run(["tmux", "ls"], check=False)
    print("\nPorts:")
    for model in MODELS.values():
        state = "OPEN" if port_open(model.port) else "not open"
        print(f"  {model.key:6} {model.port} [{model.runtime.value:9}]: {state}")
    print("\nEndpoints:")
    for model in MODELS.values():
        print(f"  {model.key:6} http://{PUBLIC_HOST}:{model.port}/v1/chat/completions  (model: {model.served_name})")


def select_models(value: str) -> list[ModelSpec]:
    if value == "all":
        return list(MODELS.values())
    selected = []
    for key in value.split(","):
        key = key.strip().lower()
        if key not in MODELS:
            raise argparse.ArgumentTypeError(f"Unknown model: {key}")
        selected.append(MODELS[key])
    return selected


def deploy(models: list[ModelSpec], *, restart: bool, wait_minutes: int, redownload: bool,
           reimport: bool, rebuild: bool, gpu_layers: int) -> None:
    ensure_dirs()
    ensure_tmux()
    # Only bootstrap the runtimes actually needed by the selection.
    llama_server = None
    litert_cli = None
    if any(m.runtime is Runtime.LLAMA_CPP for m in models):
        llama_server = ensure_llama_cpp(rebuild)
    if any(m.runtime is Runtime.LITERT_LM for m in models):
        litert_cli = ensure_litert_lm()
    for model in models:
        start_model(model, restart=restart, redownload=redownload, reimport=reimport,
                    gpu_layers=gpu_layers, llama_server=llama_server, litert_cli=litert_cli)
    wait_ready(models, wait_minutes)
    print_status()


def stop(models: list[ModelSpec]) -> None:
    ensure_tmux()
    for model in models:
        stop_session(model.session)
    print_status()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy local model endpoints (LiteRT-LM for Gemma, llama.cpp for Qwen/Llama) — mobile parity.")
    parser.add_argument("--models", default="all", type=select_models, help="all, gemma, qwen, llama, or comma-separated")
    parser.add_argument("--restart", action="store_true", help="Kill existing tmux sessions and start again")
    parser.add_argument("--stop", action="store_true", help="Stop selected model sessions")
    parser.add_argument("--status", action="store_true", help="Show GPU, sessions, ports, endpoints")
    parser.add_argument("--wait-minutes", type=int, default=0, help="Wait for ports to become ready")
    parser.add_argument("--redownload", action="store_true", help="Delete + re-download llama.cpp GGUF files")
    parser.add_argument("--reimport", action="store_true", help="Delete + re-import LiteRT-LM (.litertlm) models")
    parser.add_argument("--rebuild", action="store_true", help="Recompile llama.cpp (needed to switch on CUDA/GPU)")
    parser.add_argument("--gpu-layers", type=int, default=GPU_LAYERS, help="llama.cpp -ngl. 0 = CPU only")
    args = parser.parse_args()

    try:
        if args.status:
            ensure_tmux()
            print_status()
        elif args.stop:
            stop(args.models)
        else:
            deploy(args.models, restart=args.restart, wait_minutes=args.wait_minutes,
                   redownload=args.redownload, reimport=args.reimport, rebuild=args.rebuild,
                   gpu_layers=args.gpu_layers)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print(f"Logs: {LOG_DIR}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
