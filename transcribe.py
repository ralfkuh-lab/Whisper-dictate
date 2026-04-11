"""Transkribiert Audio/Video-Dateien mit faster-whisper."""

import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe(file_path: str, model_size: str = "medium", language: str | None = None):
    path = Path(file_path)
    if not path.exists():
        print(f"Fehler: Datei '{file_path}' nicht gefunden.")
        sys.exit(1)

    print(f"Lade Modell '{model_size}' (CPU)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transkribiere '{path.name}'...")
    start = time.time()

    segments, info = model.transcribe(str(path), language=language, beam_size=5, vad_filter=True)

    print(f"Erkannte Sprache: {info.language} ({info.language_probability:.0%})")
    print("-" * 60)

    srt_lines = []
    txt_lines = []

    for i, segment in enumerate(segments, 1):
        start_ts = format_timestamp(segment.start)
        end_ts = format_timestamp(segment.end)
        text = segment.text.strip()

        # Konsolenausgabe
        print(f"[{start_ts} -> {end_ts}] {text}")

        # SRT-Format
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(text)
        srt_lines.append("")

        txt_lines.append(text)

    elapsed = time.time() - start
    print("-" * 60)
    print(f"Fertig in {elapsed:.1f}s")

    # Ausgabe-Dateien schreiben
    out_srt = path.with_suffix(".srt")
    out_txt = path.with_suffix(".txt")

    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"SRT gespeichert: {out_srt}")
    print(f"TXT gespeichert: {out_txt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: uv run transcribe.py <audio/video-datei> [modell] [sprache]")
        print()
        print("Modelle:  tiny, base, small, medium (Standard), large-v3, large-v3-turbo")
        print("Sprache:  de, en, fr, ... (auto-detect wenn leer)")
        print()
        print("Beispiele:")
        print('  uv run transcribe.py interview.mp3')
        print('  uv run transcribe.py video.mp4 small de')
        print('  uv run transcribe.py podcast.wav large-v3')
        sys.exit(0)

    file_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    language = sys.argv[3] if len(sys.argv) > 3 else None

    transcribe(file_path, model_size, language)
