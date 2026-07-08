#!/usr/bin/env python3
"""
Deploy 3 local LLM endpoints with llama.cpp + tmux.

Models:
  Gemma 4 E2B Q4_K_M          -> port 8011
  Qwen3.5 2B Q4_K_M           -> port 8002
  Llama 3.2 3B Instruct Q4_K_M -> port 8003

Usage:
  python3 deploy_local_models.py --restart --wait-minutes 60
  python3 deploy_local_models.py --status
  python3 deploy_local_models.py --stop

Watch logs:
  tail -F ~/local_model_server/logs/*.log
"""

from __future__ import annotations

import argparse
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

APP_DIR = Path.home() / "local_model_server"
LOG_DIR = APP_DIR / "logs"
MODEL_DIR = APP_DIR / "models"
LLAMA_CPP_DIR = APP_DIR / "llama.cpp"
LLAMA_SERVER = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    session: str
    port: int
    served_name: str
    gguf_url: str
    gguf_path: Path
    context: int = 4096


MODELS: dict[str, ModelSpec] = {
    "gemma": ModelSpec(
        key="gemma",
        session="gemma-e2b",
        port=8011,
        served_name="gemma-4-e2b",
        gguf_url=(
            "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF"
            "/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
        ),
        gguf_path=MODEL_DIR / "gemma4_e2b" / "gemma-4-E2B-it-Q4_K_M.gguf",
    ),
    "qwen": ModelSpec(
        key="qwen",
        session="qwen35-2b",
        port=8002,
        served_name="qwen3.5-2b",
        gguf_url=(
            "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF"
            "/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"
        ),
        gguf_path=MODEL_DIR / "qwen35_2b" / "Qwen3.5-2B-Q4_K_M.gguf",
    ),
    "llama": ModelSpec(
        key="llama",
        session="llama32-3b",
        port=8003,
        served_name="llama-3.2-3b-instruct",
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
        print(f"Missing tools: {', '.join(missing)}")
        print("apt-get is not available; install them manually.")
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


def ensure_llama_cpp() -> Path:
    if LLAMA_SERVER.exists():
        return LLAMA_SERVER

    apt_install(["git", "cmake", "make", "g++", "curl"])
    if not exists("git") or not exists("cmake"):
        raise RuntimeError("git and cmake are required to build llama.cpp.")

    if not LLAMA_CPP_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_CPP_DIR)])

    build_dir = LLAMA_CPP_DIR / "build"
    jobs = str(max(1, (os.cpu_count() or 2) - 1))
    run(["cmake", "-S", str(LLAMA_CPP_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j", jobs, "--target", "llama-server"])

    if not LLAMA_SERVER.exists():
        raise RuntimeError("llama.cpp build failed: llama-server not found.")
    return LLAMA_SERVER


def ensure_model_file(model: ModelSpec, redownload: bool) -> Path:
    model.gguf_path.parent.mkdir(parents=True, exist_ok=True)

    if redownload and model.gguf_path.exists():
        model.gguf_path.unlink()

    if model.gguf_path.exists() and model.gguf_path.stat().st_size > 100 * 1024 * 1024:
        return model.gguf_path

    if not exists("curl"):
        apt_install(["curl"])
    if not exists("curl"):
        raise RuntimeError("curl is required to download models.")

    run([
        "curl",
        "-L",
        "--fail",
        "--continue-at",
        "-",
        "--output",
        str(model.gguf_path),
        model.gguf_url,
    ])
    return model.gguf_path


def tmux_session_exists(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session], check=False, capture=True).returncode == 0


def stop_session(session: str) -> None:
    if tmux_session_exists(session):
        run(["tmux", "kill-session", "-t", session], check=False)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_model(model: ModelSpec, llama_server: Path, restart: bool, redownload: bool) -> None:
    if tmux_session_exists(model.session):
        if restart:
            stop_session(model.session)
        else:
            print(f"{model.session} already exists; use --restart to recreate it.")
            return

    if port_open(model.port):
        print(f"WARNING: port {model.port} is already open. {model.key} may fail if another process owns it.")

    gguf_path = ensure_model_file(model, redownload)
    log_path = LOG_DIR / f"{model.session}.log"
    command = [
        str(llama_server),
        "-m", str(gguf_path),
        "--host", HOST,
        "--port", str(model.port),
        "-c", str(model.context),
        "--alias", model.served_name,
    ]
    wrapped = "set -e; " + " ".join(shlex.quote(x) for x in command) + f" 2>&1 | tee -a {shlex.quote(str(log_path))}"
    run(["tmux", "new-session", "-d", "-s", model.session, "bash", "-lc", wrapped])
    print(f"Started: {model.session}")
    print(f"Log: {log_path}")


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
        print(f"Not ready: {model.key}. Check: {LOG_DIR / (model.session + '.log')}")


def print_status() -> None:
    print("\nSessions:")
    run(["tmux", "ls"], check=False)
    print("\nPorts:")
    for model in MODELS.values():
        state = "OPEN" if port_open(model.port) else "not open"
        print(f"  {model.key:6} {model.port}: {state}")
    print("\nEndpoints:")
    for model in MODELS.values():
        print(f"  {model.key:6} http://{PUBLIC_HOST}:{model.port}/v1/chat/completions")


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


def deploy(models: list[ModelSpec], restart: bool, wait_minutes: int, redownload: bool) -> None:
    ensure_dirs()
    ensure_tmux()
    llama_server = ensure_llama_cpp()
    for model in models:
        start_model(model, llama_server, restart, redownload)
    wait_ready(models, wait_minutes)
    print_status()


def stop(models: list[ModelSpec]) -> None:
    ensure_tmux()
    for model in models:
        stop_session(model.session)
    print_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy local GGUF model servers with llama.cpp + tmux.")
    parser.add_argument("--models", default="all", type=select_models, help="all, gemma, qwen, llama, or comma-separated")
    parser.add_argument("--restart", action="store_true", help="Kill existing tmux sessions and start again")
    parser.add_argument("--stop", action="store_true", help="Stop selected model sessions")
    parser.add_argument("--status", action="store_true", help="Show tmux sessions, ports, and endpoints")
    parser.add_argument("--wait-minutes", type=int, default=0, help="Wait for ports to become ready")
    parser.add_argument("--redownload", action="store_true", help="Delete selected model files and download again")
    args = parser.parse_args()

    try:
        if args.status:
            ensure_tmux()
            print_status()
        elif args.stop:
            stop(args.models)
        else:
            deploy(args.models, args.restart, args.wait_minutes, args.redownload)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print(f"Logs: {LOG_DIR}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
