"""Diktierfunktion mit GUI-Dialog.

Ctrl+Alt+D startet/stoppt Aufnahme. Transkription wird in einem
editierbaren Dialog angezeigt und kann in die Zwischenablage kopiert werden.
Mehrere Aufnahmen werden im selben Dialog aneinandergehängt.

Funktioniert ohne Admin-Rechte.
"""

import json
import os
import sys
import platform
import threading
import tempfile
from pathlib import Path

import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from pynput import keyboard

# -- Konstanten --
SAMPLE_RATE = 16000
DIALOG_WIDTH_RATIO = 0.35   # 35% der Bildschirmbreite
DIALOG_HEIGHT_RATIO = 0.30  # 30% der Bildschirmhöhe
DIALOG_MIN_WIDTH = 650
DIALOG_MIN_HEIGHT = 340

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
AVAILABLE_DEVICES = ["auto", "cpu", "cuda"]

# -- Konfiguration --

DEFAULT_CONFIG = {
    "model_size": "medium",
    "language": "de",
    "hotkey": "<ctrl>+<alt>+d",
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
                stored = json.load(f)
            # Merge mit Defaults für fehlende Keys
            return {**DEFAULT_CONFIG, **stored}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def hotkey_label(hotkey_str: str) -> str:
    """Erzeugt ein lesbares Label aus einem pynput-Hotkey-String wie '<ctrl>+<alt>+d'."""
    parts = hotkey_str.split("+")
    nice = []
    for p in parts:
        p = p.strip()
        if p.startswith("<") and p.endswith(">"):
            nice.append(p[1:-1].capitalize())
        else:
            nice.append(p.upper())
    return "+".join(nice)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# Plattformunabhängige Schriftarten
FONT_FAMILY = "Segoe UI" if IS_WINDOWS else "Sans"
FONT_NORMAL = (FONT_FAMILY, 11)
FONT_BTN = (FONT_FAMILY, 10)
FONT_BTN_BOLD = (FONT_FAMILY, 10, "bold")
FONT_STATUS = (FONT_FAMILY, 9)
FONT_STATUS_BOLD = (FONT_FAMILY, 9, "bold")


class DictateApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.withdraw()
        self.root.title("Whisper Dictate")

        self.cfg = load_config()

        self.recording = False
        self.transcribing = False
        self.audio_chunks: list = []
        self.stream = None
        self.model = None
        self.model_loaded = False
        self.dialog: tk.Toplevel | None = None
        self._pulse_step = 0

        # GUI-Widget-Referenzen
        self.result_text: tk.Text | None = None
        self.status_bar: tk.Label | None = None
        self.record_btn: tk.Button | None = None
        self.copy_btn: tk.Button | None = None
        self.cancel_btn: tk.Button | None = None

        # Modell im Hintergrund laden
        threading.Thread(target=self._load_model, daemon=True).start()

        # Hotkey-Listener starten
        self._start_hotkey_listener()

    def _load_model(self):
        from faster_whisper import WhisperModel
        device = self.cfg.get("device", "auto")
        compute_type = "int8_float16" if device == "cuda" else "int8"
        cpu_threads = self.cfg.get("cpu_threads", 0)
        self.model = WhisperModel(
            self.cfg["model_size"], device=device, compute_type=compute_type,
            cpu_threads=cpu_threads or 0,
        )
        self.model_loaded = True

    def _start_hotkey_listener(self):
        hotkey_keys = keyboard.HotKey.parse(self.cfg["hotkey"])
        hotkey = keyboard.HotKey(hotkey_keys, self._on_hotkey)

        def on_press(key):
            hotkey.press(self._listener.canonical(key))

        def on_release(key):
            hotkey.release(self._listener.canonical(key))

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def _restart_hotkey_listener(self):
        """Hotkey-Listener mit neuer Konfiguration neu starten."""
        if hasattr(self, '_listener') and self._listener.is_alive():
            self._listener.stop()
        self._start_hotkey_listener()

    def _on_hotkey(self):
        self.root.after(0, self._hotkey_action)

    def _hotkey_action(self):
        """Hotkey öffnet den Dialog oder toggelt die Aufnahme wenn schon offen."""
        if self.dialog and self.dialog.winfo_exists():
            self._toggle_recording()
        else:
            self._ensure_dialog()
            self._update_button_states()

    def _toggle_recording(self):
        if self.transcribing:
            return  # Während Transkription keine Aktion
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    # -- Dialog --

    def _focus_dialog(self):
        """Dialog in den Vordergrund bringen und Fokus setzen."""
        if not self.dialog or not self.dialog.winfo_exists():
            return
        self.dialog.lift()
        self.dialog.focus_force()

    def _ensure_dialog(self):
        """Dialog erstellen falls noch nicht offen."""
        if self.dialog and self.dialog.winfo_exists():
            return

        self.dialog = tk.Toplevel(self.root)
        self.dialog.title("Whisper Dictate")
        self.dialog.attributes("-topmost", True)
        self.dialog.resizable(True, True)
        self.dialog.protocol("WM_DELETE_WINDOW", self._close_dialog)
        self.dialog.minsize(DIALOG_MIN_WIDTH, DIALOG_MIN_HEIGHT)

        # Größe dynamisch an Bildschirm anpassen
        screen_w = self.dialog.winfo_screenwidth()
        screen_h = self.dialog.winfo_screenheight()
        dlg_w = max(DIALOG_MIN_WIDTH, int(screen_w * DIALOG_WIDTH_RATIO))
        dlg_h = max(DIALOG_MIN_HEIGHT, int(screen_h * DIALOG_HEIGHT_RATIO))
        x = (screen_w - dlg_w) // 2
        y = (screen_h - dlg_h) // 2
        self.dialog.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")

        main_frame = tk.Frame(self.dialog, padx=15, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Grid-Layout: Textfeld expandiert, Buttons und Status bleiben sichtbar
        main_frame.rowconfigure(0, weight=1)  # Textfeld bekommt übrigen Platz
        main_frame.rowconfigure(1, weight=0)  # Buttons feste Höhe
        main_frame.rowconfigure(2, weight=0)  # Statusleiste feste Höhe
        main_frame.columnconfigure(0, weight=1)

        # Editierbares Textfeld
        self.result_text = tk.Text(
            main_frame, font=FONT_NORMAL, wrap=tk.WORD, padx=5, pady=5
        )
        self.result_text.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.result_text.mark_set(tk.INSERT, tk.END)

        # Button-Leiste
        btn_frame = tk.Frame(main_frame)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        # Aufnahme (Toggle: Start/Stop)
        self.record_btn = tk.Button(
            btn_frame,
            text="\U0001f3a4 Aufnahme (F5)",
            command=self._toggle_recording,
            font=FONT_BTN,
            bg="#d4edda",
            fg="#155724",
            padx=12,
            pady=5,
            cursor="hand2",
        )
        self.record_btn.pack(side=tk.LEFT)

        # Kopieren & Schliessen (rechts)
        self.copy_btn = tk.Button(
            btn_frame,
            text="Kopieren & Schliessen (F6)",
            command=self._copy_and_close,
            font=FONT_BTN_BOLD,
            bg="#cce5ff",
            fg="#004085",
            padx=12,
            pady=5,
            cursor="hand2",
        )
        self.copy_btn.pack(side=tk.RIGHT)

        # Abbrechen
        self.cancel_btn = tk.Button(
            btn_frame,
            text="Abbrechen (Esc)",
            command=self._close_dialog,
            font=FONT_BTN,
            padx=10,
            pady=5,
            cursor="hand2",
        )
        self.cancel_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # Clear
        self.clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            command=self._clear_text,
            font=FONT_BTN,
            padx=10,
            pady=5,
            cursor="hand2",
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # Statusleiste (ganz unten) — Frame mit Label + Config-Button
        status_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=1)
        status_frame.grid(row=2, column=0, sticky="ew")

        self.status_bar = tk.Label(
            status_frame,
            text="Bereit",
            font=FONT_STATUS,
            fg="#888",
            anchor=tk.W,
            padx=6,
            pady=2,
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        config_btn = tk.Button(
            status_frame,
            text="\u2699",
            command=self._open_settings,
            font=(FONT_FAMILY, 11),
            relief=tk.FLAT,
            cursor="hand2",
            bd=0,
            padx=6,
            pady=0,
        )
        config_btn.pack(side=tk.RIGHT)

        # Shortcuts
        self.dialog.bind("<Escape>", lambda e: self._close_dialog())
        self.dialog.bind("<Control-Return>", lambda e: self._copy_and_close())
        self.dialog.bind("<F5>", lambda e: self._toggle_recording())
        self.dialog.bind("<F6>", lambda e: self._copy_and_close())

        # Fokus erzwingen
        self.dialog.after(50, self._focus_dialog)

    def _update_button_states(self):
        """Buttons je nach Zustand aktivieren/deaktivieren."""
        if not self.dialog or not self.dialog.winfo_exists():
            return

        if self.recording:
            self.record_btn.config(
                text="\U0001f3a4 Aufnahme (F5)",
                bg="#f8d7da", fg="#721c24",
                state=tk.NORMAL,
            )
            self.copy_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.NORMAL)
        elif self.transcribing:
            self.record_btn.config(state=tk.DISABLED)
            self.copy_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.DISABLED)
        else:
            self.record_btn.config(
                text="\U0001f3a4 Aufnahme (F5)",
                bg="#d4edda", fg="#155724",
                state=tk.NORMAL,
            )
            self.copy_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.NORMAL)

    def _set_status(self, text: str, fg: str = "#888", bold: bool = False):
        """Statusleiste aktualisieren."""
        if not self.dialog or not self.dialog.winfo_exists():
            return
        font = FONT_STATUS_BOLD if bold else FONT_STATUS
        self.status_bar.config(text=text, fg=fg, font=font)

    # -- Recording --

    def _start_recording(self):
        if not self.model_loaded:
            self._ensure_dialog()
            self._set_status("Modell wird geladen, bitte warten...")
            self._update_button_states()
            self._wait_for_model()
            return

        self._ensure_dialog()

        self.audio_chunks = []
        self.recording = True
        self._pulse_step = 0

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

        self._set_status("Aufnahme...", fg="#cc0000", bold=True)
        self._update_button_states()
        self._pulse()
        self.dialog.focus_force()

    def _wait_for_model(self):
        if not self.dialog or not self.dialog.winfo_exists():
            return
        if self.model_loaded:
            self._start_recording()
        else:
            self.root.after(500, self._wait_for_model)

    def _record_loop(self):
        """Blockierende Aufnahme in eigenem Thread."""
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
                while self.recording:
                    data, overflowed = stream.read(int(SAMPLE_RATE * 0.1))  # 100ms Blöcke
                    self.audio_chunks.append(data.copy())
        except Exception:
            pass

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        # Warten bis Record-Thread fertig, dann transkribieren
        self._set_status("Aufnahme wird beendet...", fg="#e68a00")
        threading.Thread(target=self._finish_recording, daemon=True).start()

    def _finish_recording(self):
        # Warten bis der Record-Thread sich beendet hat
        if hasattr(self, '_record_thread') and self._record_thread.is_alive():
            self._record_thread.join(timeout=2.0)

        if not self.audio_chunks:
            self.root.after(0, self._no_audio)
            return

        self.root.after(0, self._start_transcription)

    def _no_audio(self):
        self.transcribing = False
        self._set_status("Keine Audiodaten.")
        self._update_button_states()

    def _start_transcription(self):
        self.transcribing = True
        self._set_status("Transkribiere...", fg="#e68a00", bold=True)
        self._update_button_states()
        threading.Thread(target=self._transcribe, daemon=True).start()

    def _is_prompt_hallucination(self, text: str) -> bool:
        """Prüft ob der Text nur aus Wörtern des initial_prompt besteht (Halluzination)."""
        prompt = self.cfg.get("initial_prompt", "")
        if not prompt or not text:
            return False
        # Prompt-Wörter normalisieren
        prompt_words = set(
            w.strip("., ").lower() for w in prompt.replace(",", " ").split()
            if w.strip("., ")
        )
        # Transkribierte Wörter normalisieren
        text_words = [
            w.strip("., ").lower() for w in text.split()
            if w.strip("., ")
        ]
        if not text_words:
            return True
        # Wenn >80% der Wörter aus dem Prompt stammen, ist es eine Halluzination
        matches = sum(1 for w in text_words if w in prompt_words)
        return matches / len(text_words) > 0.8

    def _transcribe(self):
        try:
            audio = np.concatenate(self.audio_chunks, axis=0)
            audio_int16 = (audio * 32767).astype(np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                wavfile.write(tmp_path, SAMPLE_RATE, audio_int16)

            lang = self.cfg["language"] or None
            prompt = self.cfg.get("initial_prompt", "") or None
            segments, info = self.model.transcribe(
                tmp_path, language=lang, beam_size=5, initial_prompt=prompt,
                no_speech_threshold=0.6, log_prob_threshold=-1.0,
                hallucination_silence_threshold=2.0,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)

            # Halluzinierte Prompt-Wiederholungen filtern
            if self._is_prompt_hallucination(text):
                text = ""

            Path(tmp_path).unlink(missing_ok=True)

            self.root.after(0, self._append_result, text)
        except Exception as e:
            self.root.after(0, self._append_result, f"[Fehler: {e}]")

    def _append_result(self, text: str):
        self.transcribing = False

        if not self.dialog or not self.dialog.winfo_exists():
            return

        if text and text.strip():
            # Text an aktueller Cursor-Position einfügen
            cursor_pos = self.result_text.index(tk.INSERT)
            # Prüfen ob vor dem Cursor schon Text steht -> Leerzeichen einfügen
            if cursor_pos != "1.0":
                char_before = self.result_text.get(f"{cursor_pos} -1c", cursor_pos)
                if char_before and not char_before.isspace():
                    self.result_text.insert(tk.INSERT, " ")
            self.result_text.insert(tk.INSERT, text.strip())

        self._set_status(
            "Fertig. F5 oder Aufnahme klicken fuer weiteres Diktat.",
            fg="#228b22",
        )
        self._update_button_states()
        self.result_text.focus_set()

    # -- Pulse Animation --

    def _pulse(self):
        if not self.recording or not self.dialog or not self.dialog.winfo_exists():
            return
        dots = "." * ((self._pulse_step % 3) + 1)
        self._set_status(f"Aufnahme{dots}", fg="#cc0000", bold=True)
        self._pulse_step += 1
        self.dialog.after(500, self._pulse)

    # -- Actions --

    def _clear_text(self):
        """Textfeld komplett leeren."""
        if self.result_text:
            self.result_text.delete("1.0", tk.END)

    def _copy_and_close(self):
        if self.result_text:
            text = self.result_text.get("1.0", tk.END).strip()
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
        self._close_dialog()

    def _close_dialog(self):
        # Laufende Aufnahme stoppen
        self.recording = False
        self.audio_chunks = []
        if self.dialog:
            self.dialog.destroy()
            self.dialog = None

    # -- Einstellungen --

    def _open_settings(self):
        """Einstellungs-Dialog öffnen."""
        # Globalen Hotkey-Listener pausieren, damit er nicht bei Tastendruck feuert
        if hasattr(self, '_listener') and self._listener.is_alive():
            self._listener.stop()

        settings = tk.Toplevel(self.dialog or self.root)
        settings.title("Einstellungen")
        settings.attributes("-topmost", True)
        settings.resizable(False, False)
        settings.grab_set()

        def _on_close():
            settings.destroy()
            self._start_hotkey_listener()

        settings.protocol("WM_DELETE_WINDOW", _on_close)

        # Größe automatisch berechnen nach Layout
        settings.update_idletasks()
        screen_w = settings.winfo_screenwidth()
        screen_h = settings.winfo_screenheight()

        frame = tk.Frame(settings, padx=20, pady=15)
        frame.pack(fill=tk.BOTH, expand=True)

        # -- Modellauswahl --
        tk.Label(frame, text="Whisper-Modell:", font=FONT_NORMAL).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        model_var = tk.StringVar(value=self.cfg["model_size"])
        model_combo = ttk.Combobox(
            frame,
            textvariable=model_var,
            values=AVAILABLE_MODELS,
            state="readonly",
            width=18,
        )
        model_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 8))

        # -- Gerät (CPU/GPU) --
        tk.Label(frame, text="Gerät:", font=FONT_NORMAL).grid(
            row=1, column=0, sticky="w", pady=(0, 8)
        )
        device_var = tk.StringVar(value=self.cfg.get("device", "auto"))
        device_combo = ttk.Combobox(
            frame,
            textvariable=device_var,
            values=AVAILABLE_DEVICES,
            state="readonly",
            width=18,
        )
        device_combo.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(0, 8))

        # -- CPU-Threads --
        tk.Label(frame, text="CPU-Threads:", font=FONT_NORMAL).grid(
            row=2, column=0, sticky="w", pady=(0, 8)
        )
        threads_var = tk.StringVar(value=str(self.cfg.get("cpu_threads", 0)))
        threads_spin = tk.Spinbox(
            frame, textvariable=threads_var, from_=0, to=32,
            width=5, font=FONT_NORMAL,
        )
        threads_spin.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(0, 8))
        tk.Label(frame, text="(0 = auto)", font=FONT_STATUS, fg="#888").grid(
            row=2, column=2, sticky="w", padx=(4, 0), pady=(0, 8)
        )

        # -- Sprache --
        tk.Label(frame, text="Sprache:", font=FONT_NORMAL).grid(
            row=3, column=0, sticky="w", pady=(0, 8)
        )
        lang_var = tk.StringVar(value=self.cfg["language"] or "")
        lang_entry = tk.Entry(frame, textvariable=lang_var, width=20, font=FONT_NORMAL)
        lang_entry.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(0, 8))
        tk.Label(frame, text="(leer = auto)", font=FONT_STATUS, fg="#888").grid(
            row=3, column=2, sticky="w", padx=(4, 0), pady=(0, 8)
        )

        # -- Initial Prompt (Vokabular-Hilfe) --
        tk.Label(frame, text="Vokabular:", font=FONT_NORMAL).grid(
            row=4, column=0, sticky="nw", pady=(0, 8)
        )
        prompt_text = tk.Text(frame, width=25, height=3, font=FONT_NORMAL, wrap=tk.WORD)
        prompt_text.insert("1.0", self.cfg.get("initial_prompt", ""))
        prompt_text.grid(row=4, column=1, sticky="we", padx=(10, 0), pady=(0, 8))
        tk.Label(frame, text="(Begriffe, kommagetrennt)", font=FONT_STATUS, fg="#888").grid(
            row=4, column=2, sticky="nw", padx=(4, 0), pady=(0, 8)
        )

        # -- Hotkey --
        tk.Label(frame, text="Hotkey:", font=FONT_NORMAL).grid(
            row=5, column=0, sticky="w", pady=(0, 8)
        )
        hotkey_var = tk.StringVar(value=self.cfg["hotkey"])
        hotkey_entry = tk.Entry(
            frame, textvariable=hotkey_var, width=20, font=FONT_NORMAL
        )
        hotkey_entry.grid(row=5, column=1, sticky="w", padx=(10, 0), pady=(0, 8))

        # Hotkey-Capture: Tastenkombination aufzeichnen
        captured_keys: set[str] = set()
        capturing = [False]

        def _start_capture(event=None):
            capturing[0] = True
            captured_keys.clear()
            hotkey_entry.config(fg="#cc0000")
            hotkey_var.set("Taste drücken...")

        def _on_key(event):
            if not capturing[0]:
                return
            key_name = event.keysym
            # Modifier-Keys erkennen
            modifiers = {
                "Control_L": "<ctrl>", "Control_R": "<ctrl>",
                "Alt_L": "<alt>", "Alt_R": "<alt>",
                "Shift_L": "<shift>", "Shift_R": "<shift>",
            }
            if key_name in modifiers:
                captured_keys.add(modifiers[key_name])
            else:
                captured_keys.add(key_name.lower())
                # Fertig: Hotkey zusammensetzen
                capturing[0] = False
                # Sortierung: Modifier zuerst, dann der normale Key
                mods = sorted(k for k in captured_keys if k.startswith("<"))
                normals = sorted(k for k in captured_keys if not k.startswith("<"))
                result = "+".join(mods + normals)
                hotkey_var.set(result)
                hotkey_entry.config(fg="#000")

        hotkey_entry.bind("<FocusIn>", _start_capture)
        hotkey_entry.bind("<Key>", _on_key)

        # -- Fehler-Label --
        error_label = tk.Label(frame, text="", font=FONT_STATUS, fg="#cc0000")
        error_label.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # -- Buttons --
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=7, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def _save():
            new_hotkey = hotkey_var.get().strip()
            new_model = model_var.get()
            new_device = device_var.get()
            new_lang = lang_var.get().strip() or ""
            new_prompt = prompt_text.get("1.0", "end-1c").strip()
            new_threads = int(threads_var.get() or 0)

            # Hotkey validieren
            try:
                keyboard.HotKey.parse(new_hotkey)
            except Exception:
                error_label.config(text=f"Ungültiger Hotkey: {new_hotkey}")
                return

            model_changed = (
                new_model != self.cfg["model_size"]
                or new_device != self.cfg.get("device", "auto")
                or new_threads != self.cfg.get("cpu_threads", 0)
            )

            self.cfg["model_size"] = new_model
            self.cfg["device"] = new_device
            self.cfg["language"] = new_lang
            self.cfg["hotkey"] = new_hotkey
            self.cfg["initial_prompt"] = new_prompt
            self.cfg["cpu_threads"] = new_threads
            save_config(self.cfg)

            if model_changed:
                self.model = None
                self.model_loaded = False
                threading.Thread(target=self._load_model, daemon=True).start()

            settings.destroy()
            self._start_hotkey_listener()

        tk.Button(
            btn_frame, text="Abbrechen", command=_on_close,
            font=FONT_BTN, padx=10, pady=4,
        ).pack(side=tk.RIGHT, padx=(8, 0))

        tk.Button(
            btn_frame, text="Speichern", command=_save,
            font=FONT_BTN_BOLD, bg="#cce5ff", fg="#004085",
            padx=10, pady=4, cursor="hand2",
        ).pack(side=tk.RIGHT)

        # Fenstergröße automatisch an Inhalt anpassen und zentrieren
        settings.update_idletasks()
        w = settings.winfo_reqwidth()
        h = settings.winfo_reqheight()
        settings.geometry(f"{w}x{h}+{(screen_w - w) // 2}+{(screen_h - h) // 2}")


# -- Autostart --

def _get_autostart_path() -> Path:
    """Plattformspezifischen Autostart-Pfad ermitteln."""
    if IS_WINDOWS:
        return (
            Path(os.environ["APPDATA"])
            / r"Microsoft\Windows\Start Menu\Programs\Startup"
            / "whisper-dictate.vbs"
        )
    if IS_LINUX:
        return (
            Path.home() / ".config" / "autostart" / "whisper-dictate.desktop"
        )
    raise RuntimeError(f"Autostart nicht unterstützt auf {platform.system()}")


def install_autostart():
    script = Path(__file__).resolve()
    project_dir = script.parent
    target = _get_autostart_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if IS_WINDOWS:
        cmd_script = project_dir / "restart-whisper-dictate.cmd"
        content = (
            'Set WshShell = CreateObject("WScript.Shell")\n'
            f'WshShell.CurrentDirectory = "{project_dir}"\n'
            f'WshShell.Run """{cmd_script}""", 0, False\n'
        )
    elif IS_LINUX:
        content = (
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=WhisperDictate\n"
            "Comment=Diktierfunktion mit Whisper\n"
            f"Exec=uv run {script}\n"
            f"Path={project_dir}\n"
            "Terminal=false\n"
            "X-GNOME-Autostart-enabled=true\n"
        )
    else:
        raise RuntimeError(f"Autostart nicht unterstützt auf {platform.system()}")

    target.write_text(content, encoding="utf-8")
    print(f"Autostart installiert: {target}")


def remove_autostart():
    target = _get_autostart_path()
    if target.exists():
        target.unlink()
        print(f"Autostart entfernt: {target}")
    else:
        print("Autostart war nicht installiert.")


def main():
    if "--install-autostart" in sys.argv:
        install_autostart()
        return
    if "--remove-autostart" in sys.argv:
        remove_autostart()
        return

    root = tk.Tk()
    DictateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
