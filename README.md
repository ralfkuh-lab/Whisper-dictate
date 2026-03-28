# WhisperDictate

Lokales Diktiertool mit globalem Hotkey und GUI-Dialog. Nutzt [faster-whisper](https://github.com/SYSTRAN/faster-whisper) fuer offline Spracherkennung — keine Cloud, keine API-Keys.

![WhisperDictate Screenshot](Screenshot.png)

## Features

- **Globaler Hotkey** (Ctrl+Alt+D) — Diktat aus jeder Anwendung starten/stoppen
- **Lokale Transkription** mit faster-whisper (laeuft komplett offline)
- **Editierbarer Dialog** — Text vor dem Kopieren korrigieren
- **Mehrfach-Aufnahme** — mehrere Diktate im selben Dialog aneinanderhaengen
- **Einstellungen** — Modellgroesse, Sprache, Hotkey und Geraet (CPU/CUDA) konfigurierbar
- **Autostart** — optional beim Systemstart ausfuehren
- **Cross-platform** — Windows und Linux

## Voraussetzungen

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (empfohlen) oder pip

### Linux

```bash
sudo apt install python3-tk python3-dev libportaudio2
```

Fuer CUDA-Unterstuetzung (NVIDIA GPU):

```bash
sudo apt install libcublas12 libcublaslt12 nvidia-cudnn
```

## Installation & Start

```bash
git clone <repo-url>
cd whisper-dictate
uv run dictate.py
```

### Windows: Start/Restart per Script

`restart-whisper-dictate.cmd` startet WhisperDictate im Hintergrund. Laeuft der Prozess bereits, wird er vorher beendet und neu gestartet. Das Script nutzt relative Pfade — es kann z.B. per Verknuepfung von einem beliebigen Ort aus aufgerufen werden.

Beim ersten Start wird das Whisper-Modell heruntergeladen (je nach Groesse 75 MB – 3 GB).

## Verwendung

1. **Ctrl+Alt+D** druecken — Aufnahmedialog oeffnet sich, Aufnahme startet
2. Sprechen
3. **Ctrl+Alt+D** erneut druecken oder **Aufnahme stoppen** klicken — Transkription laeuft
4. Text im Dialog pruefen/bearbeiten
5. **Kopieren & Schliessen** — Text landet in der Zwischenablage

Fuer weitere Diktate im selben Dialog: **Neue Aufnahme** klicken.

## Konfiguration

Ueber den Zahnrad-Button im Dialog oder direkt in der Konfigurationsdatei:

- **Windows:** `%APPDATA%/whisper-dictate/config.json`
- **Linux:** `~/.config/whisper-dictate/config.json`

| Einstellung  | Beschreibung                          | Default           |
|-------------|---------------------------------------|-------------------|
| `model_size` | Whisper-Modell (tiny/base/small/medium/large-v3) | `medium`  |
| `language`   | Sprache (`de`, `en`, ... oder `""` fuer Auto-Detect) | `de` |
| `hotkey`     | Globaler Hotkey (pynput-Format)       | `<ctrl>+<alt>+d`  |
| `device`     | Geraet fuer Inferenz (auto/cpu/cuda)  | `auto`            |
| `initial_prompt` | Vokabular-Hilfe (kommagetrennte Begriffe) | Fachbegriffe (Git, API, ...) |

## Autostart

```bash
uv run dictate.py --install-autostart   # Autostart einrichten
uv run dictate.py --remove-autostart    # Autostart entfernen
```

## Lizenz

MIT
