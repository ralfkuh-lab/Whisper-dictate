# WhisperDictate

Diktiertool mit globalem Hotkey (Ctrl+Alt+D) und GUI-Dialog. Nutzt faster-whisper für lokale Spracherkennung.

## Architektur

- `dictate.py` — Hauptmodul: GUI (tkinter), Aufnahme (sounddevice), Hotkey (pynput), Transkription (faster-whisper)
- `transcribe.py` — Standalone-Transkriptionsskript (CLI)
- `main.py` — Einstiegspunkt (ruft `dictate.main()` auf)

## Wichtige Klasse

`DictateApp` in `dictate.py`:
- Zustandsmaschine: idle → recording → transcribing → idle
- Modell wird asynchron im Hintergrund geladen
- `_start_recording` / `_wait_for_model` arbeiten zusammen: wenn Modell noch nicht geladen, pollt `_wait_for_model` alle 500ms und ruft dann `_start_recording` erneut auf
- Aufnahme läuft in eigenem Thread (`_record_loop`), Transkription ebenso (`_transcribe`)
- `_is_prompt_hallucination` filtert Whisper-Halluzinationen, bei denen nur der initial_prompt wiederholt wird
- Dialog-Größe skaliert dynamisch mit Bildschirmauflösung (35%/30%, min 650x340)

## Plattform

Cross-platform (Windows + Linux). Plattform-spezifisch:
- Schriftarten: `Segoe UI` (Windows) / `Sans` (Linux) — gesteuert über Konstanten `FONT_*`
- Autostart: VBScript (Windows) / `.desktop`-Datei (Linux) — in `install_autostart()`

### Linux-Voraussetzungen
```
sudo apt install python3-tk python3-dev libportaudio2
```

Für CUDA (optional):
```
sudo apt install libcublas12 libcublaslt12 nvidia-cudnn
```

## Entwicklung

```bash
uv run dictate.py                    # Starten
uv run dictate.py --install-autostart  # Autostart einrichten
uv run dictate.py --remove-autostart   # Autostart entfernen
```

## Konfiguration

Einstellungen werden in einer JSON-Datei gespeichert:
- Windows: `%APPDATA%/whisper-dictate/config.json`
- Linux: `~/.config/whisper-dictate/config.json`

Konfigurierbare Werte (auch über den Zahnrad-Button im Dialog änderbar):
- `model_size` — Whisper-Modellgröße (tiny/base/small/medium/large-v3)
- `language` — Sprache ("de", "" für auto-detect)
- `hotkey` — Globaler Hotkey im pynput-Format (z.B. `<ctrl>+<alt>+d`)
- `initial_prompt` — Vokabular-Hilfe: kommagetrennte Begriffe, die Whisper als Kontext-Hinweis erhält (verbessert Erkennung von Fachbegriffen/Fremdwörtern)

Defaults: siehe `DEFAULT_CONFIG` in `dictate.py`.
