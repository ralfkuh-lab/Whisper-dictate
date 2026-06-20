#!/usr/bin/env python3
"""Headless Whisper-STT-Server — gleiche HTTP-API wie die Dictate-GUI, aber ohne
Tkinter/Hotkey. Damit kann das Local-AI Cockpit Whisper als eigenständige Instanz
starten und (wichtiger) stoppen, um den VRAM (~1.2 GB) für ein großes LLM freizugeben.

Wiederverwendung von `http_server.create_app` → identische Endpunkte:
  GET  /health                     -> {model_loaded, model_size}
  POST /transcribe                 -> roh-PCM (float32 LE @16k mono) -> {text}
  POST /v1/audio/transcriptions    -> OpenAI-kompatibel (multipart file)

Liest dieselbe Config wie dictate.py (~/.config/whisper-dictate/config.json) und
lädt dasselbe faster-whisper-Modell. dictate.py bleibt unangetastet; der Server-State
wird hier minimal nachgebildet (nur was http_server.create_app braucht).

Start (im Repo-Verzeichnis, wegen uv-Deps):
  uv run whisper-server.py
Env-Overrides: WHISPER_HOST (default 127.0.0.1), WHISPER_PORT (default 8350).
"""
import json
import os
import platform
import sys
from pathlib import Path

# http_server.py liegt im selben Verzeichnis — robust unabhängig vom cwd importierbar machen.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from faster_whisper import WhisperModel

import http_server

# 1:1 aus dictate.py dupliziert, damit dieser Server NICHT dictate.py importieren muss
# (dessen Top-Level-Imports ziehen tkinter/pynput/sounddevice — unnötig & headless-fragil).
DEFAULT_CONFIG = {
    "model_size": "medium",
    "language": "de",
    "device": "auto",
    "initial_prompt": "Git, Commit, Push, Pull, Merge, Branch, Claude Code, Repository, "
    "API, Token, Frontend, Backend, Deploy, Release, Sprint, Ticket",
    "cpu_threads": 0,
}


def _config_path() -> Path:
    if platform.system() == "Windows":
        return Path(os.environ["APPDATA"]) / "whisper-dictate" / "config.json"
    return Path.home() / ".config" / "whisper-dictate" / "config.json"


def load_config() -> dict:
    path = _config_path()
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


class ServerState:
    """Minimaler Ersatz für DictateApp — nur die Attribute, die http_server nutzt."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.model_loaded = False

    def load_model(self):
        # Identisch zu DictateApp._load_model (dictate.py:130).
        device = self.cfg.get("device", "auto")
        compute_type = "int8_float16" if device == "cuda" else "int8"
        cpu_threads = self.cfg.get("cpu_threads", 0)
        self.model = WhisperModel(
            self.cfg["model_size"],
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads or 0,
        )
        self.model_loaded = True

    def _is_prompt_hallucination(self, text: str) -> bool:
        # Identisch zu DictateApp._is_prompt_hallucination (dictate.py:436).
        prompt = self.cfg.get("initial_prompt", "")
        if not prompt or not text:
            return False
        prompt_words = set(
            w.strip("., ").lower()
            for w in prompt.replace(",", " ").split()
            if w.strip("., ")
        )
        text_words = [w.strip("., ").lower() for w in text.split() if w.strip("., ")]
        if not text_words:
            return True
        matches = sum(1 for w in text_words if w in prompt_words)
        return matches / len(text_words) > 0.8


def main():
    host = os.environ.get("WHISPER_HOST", "127.0.0.1")
    port = int(os.environ.get("WHISPER_PORT", "8350"))
    cfg = load_config()
    state = ServerState(cfg)
    print(
        f"[whisper-server] lade Modell '{cfg['model_size']}' "
        f"(device={cfg.get('device', 'auto')}) …",
        flush=True,
    )
    state.load_model()
    print(f"[whisper-server] bereit auf http://{host}:{port}", flush=True)
    api = http_server.create_app(lambda: state)
    uvicorn.run(api, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
