"""FastAPI-Server für Whisper-Transkription.

Läuft im selben Prozess wie DictateApp und teilt sich das Modell.
Akzeptiert rohe PCM-Bytes (float32 little-endian @ 16 kHz mono) per POST /transcribe.
"""

import io
import threading

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper.audio import decode_audio


SAMPLE_RATE = 16000


def create_app(get_app_state):
    """Erzeuge FastAPI-App. `get_app_state` liefert die laufende DictateApp-Instanz."""
    api = FastAPI()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    transcribe_lock = threading.Lock()

    @api.get("/health")
    def health():
        state = get_app_state()
        return {
            "model_loaded": bool(state and state.model_loaded),
            "model_size": state.cfg.get("model_size") if state else None,
        }

    @api.post("/transcribe")
    async def transcribe(request: Request):
        state = get_app_state()
        if not state or not state.model_loaded:
            return {"error": "model_not_ready"}

        body = await request.body()
        if not body:
            return {"text": ""}

        audio = np.frombuffer(body, dtype=np.float32)
        if audio.size < SAMPLE_RATE // 4:
            return {"text": ""}

        lang = state.cfg.get("language") or None
        prompt = request.query_params.get("prompt") or state.cfg.get("initial_prompt") or None

        with transcribe_lock:
            segments, _info = state.model.transcribe(
                audio,
                language=lang,
                beam_size=5,
                initial_prompt=prompt,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                hallucination_silence_threshold=2.0,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

        if state._is_prompt_hallucination(text):
            text = ""

        return {"text": text}

    @api.post("/v1/audio/transcriptions")
    async def openai_transcriptions(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),          # wird ignoriert, wir nutzen das geladene Modell
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),          # akzeptiert, aber ungenutzt
    ):
        """OpenAI-kompatibler Transkriptions-Endpoint."""
        state = get_app_state()
        if not state or not state.model_loaded:
            return JSONResponse({"error": {"message": "model_not_ready", "type": "server_error"}}, status_code=503)

        data = await file.read()
        if not data:
            return {"text": ""}

        audio = decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)
        lang = language or state.cfg.get("language") or None
        pr = prompt or state.cfg.get("initial_prompt") or None

        with transcribe_lock:
            segments, info = state.model.transcribe(
                audio,
                language=lang,
                beam_size=5,
                initial_prompt=pr,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                hallucination_silence_threshold=2.0,
                vad_filter=True,
            )
            seg_list = list(segments)
            text = " ".join(s.text.strip() for s in seg_list).strip()

        if state._is_prompt_hallucination(text):
            text = ""

        if response_format == "text":
            return PlainTextResponse(text)
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": info.language,
                "duration": info.duration,
                "text": text,
                "segments": [
                    {"id": i, "start": s.start, "end": s.end, "text": s.text.strip()}
                    for i, s in enumerate(seg_list)
                ],
            }
        else:
            return {"text": text}

    return api


def run_in_background(get_app_state, port: int = 8350):
    """Startet uvicorn in einem Daemon-Thread."""
    import uvicorn

    api = create_app(get_app_state)

    def _serve():
        uvicorn.run(api, host="127.0.0.1", port=port, log_level="warning")

    t = threading.Thread(target=_serve, daemon=True, name="whisper-http")
    t.start()
    return t
