"""Streamovací český TTS server: text -> normalizace -> věty -> Pocket TTS -> PCM, s cache pevných frází. MIT licence.

  POCKET_CHECKPOINT=/cesta/checkpoint.pt POCKET_PROMPT=/cesta/prompt.wav python -m ceskyhlas.server

Proměnné prostředí:
  POCKET_CHECKPOINT  váš doladěný checkpoint (.pt) – povinné
  POCKET_PROMPT      hlasový prompt mluvčího (wav, 5–12 s čisté řeči) – povinné
  POCKET_CONFIG      konfigurace modelu (výchozí: česká konfigurace z HF repa vvolhejn/pocket-tts-czech)
  POCKET_TEMP / POCKET_DECODE_STEPS / POCKET_NOISE_CLAMP   dekódování, výchozí 0.5 / 1 / 1.0
  POCKET_WORKERS     instance modelu v procesu (výchozí 1; generátor Pocketu není thread-safe; na dva souběžné
                     hovory jsou lepší dva procesy než dva workery v jednom)
  POCKET_THREADS     vlákna torch na CPU (výchozí 4)
  POCKET_PHRASES     složka cache frází (ceskyhlas.phrase_cache); prázdné = bez cache
  TTS_BIND / PORT    adresa (výchozí 127.0.0.1) a port (výchozí 8765); jiná adresa než loopback vyžaduje
                     TTS_TOKEN (>= 16 znaků) a každý požadavek pak musí poslat "Authorization: Bearer <token>"
Endpointy:
  GET  /health                    model (sha256 checkpointu), prompt, dekódování, stav cache frází
  POST /normalize {"text"}        {"text": normalizovaný text, "sentences": [...]}
  POST /tts {"text", "rate": 8000|16000|24000, "normalize": true}
       -> chunkované PCM s16le mono; věty se generují postupně a posílají, jak je Pocket vyrábí. Pevná fráze
          (shoda normalizovaného textu bez ohledu na velikost písmen a koncovou interpunkci) se pošle hned
          z cache, hlavička X-Phrase: <id>#<take>.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

DEFAULT_CONFIG = "hf://vvolhejn/pocket-tts-czech/czech.yaml@7c1fbd0acba765617749dd17f3dbddc2be791cc7"
DECODE = {"temp": float(os.environ.get("POCKET_TEMP", "0.5")),
          "sampler_decode_steps": int(os.environ.get("POCKET_DECODE_STEPS", "1")),
          "noise_clamp": float(os.environ.get("POCKET_NOISE_CLAMP", "1.0"))}
MODEL_SR = 24000


def sha256_12(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()[:12]


def phrase_key(text: str) -> str:
    """Normalizovaný text -> klíč cache: velikost písmen, mezery a koncová interpunkce nehrají roli."""
    return re.sub(r"\s+", " ", re.sub(r"[\s.!?…,;:]+$", "", text)).strip().lower()


class PhraseCache:
    """<dir>/manifest.json + <dir>/<id>/meta.json ("normalized", "active", "takes") + take<k>.wav (24 kHz).
    Změnu meta.json (ceskyhlas.phrase_cache set) načte do 10 s."""

    def __init__(self, root: str | None) -> None:
        self.root = Path(root) if root else None
        self.map: dict = {}
        self.version: dict | None = None
        self.stamp: tuple = ()
        self.checked = 0.0
        self.lock = threading.Lock()
        self._load()

    def _stamp(self) -> tuple:
        if not self.root or not self.root.is_dir():
            return ()
        return tuple(sorted((str(p), p.stat().st_mtime) for p in self.root.glob("*/meta.json")))

    def _load(self) -> None:
        import soundfile as sf

        m, ver = {}, None
        if self.root and (self.root / "manifest.json").exists():
            ver = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
            for mf in sorted(self.root.glob("*/meta.json")):
                meta = json.loads(mf.read_text(encoding="utf-8"))
                k = int(meta.get("active", 1))
                x, sr = sf.read(str(mf.parent / meta["takes"][k - 1]["file"]), dtype="float32")
                m[phrase_key(meta["normalized"])] = (meta["id"], k, x, sr)
        self.map, self.version, self.stamp = m, ver, self._stamp()

    def get(self, normalized: str):
        with self.lock:
            if time.time() - self.checked > 10:
                self.checked = time.time()
                if self._stamp() != self.stamp:
                    self._load()
            return self.map.get(phrase_key(normalized))

    def info(self) -> dict:
        v = self.version or {}
        return {"version": v.get("version"), "n": len(self.map), "checkpoint_sha256": v.get("checkpoint_sha256"),
                "prompt": v.get("prompt"), "active": {x[0]: x[1] for x in self.map.values()}}


def phrase_status(ph: dict, sha: str, prompt: str) -> dict:
    """Cache patří k jednomu checkpointu + promptu; po jejich změně ji postavte znovu a znovu vyberte takes."""
    if ph["n"]:
        ph["matches_model"] = ph["checkpoint_sha256"] == sha and ph.get("prompt") in (None, prompt)
        if not ph["matches_model"]:
            ph["warning"] = (f"cache postavená s checkpointem {ph['checkpoint_sha256']} / promptem {ph.get('prompt')}, "
                             f"načteno {sha} / {prompt}: postavte cache znovu")
    return ph


def pcm_chunks(audio: np.ndarray, sr: int, rate: int, chunk: int = 4000):
    import soxr

    y = soxr.resample(audio, sr, rate, quality="HQ") if sr != rate else audio
    b = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    for i in range(0, len(b), chunk * 2):
        yield b[i:i + chunk * 2]


class TTSRequest(BaseModel):   # na úrovni modulu: FastAPI musí umět vyhodnotit anotace
    text: str
    rate: int = 8000
    normalize: bool = True


class TextRequest(BaseModel):
    text: str


class Worker:
    def __init__(self, config: str, checkpoint: str, prompt: str) -> None:
        from pocket_tts import TTSModel

        self.model = TTSModel.load_model(config=config, checkpoint=checkpoint, **DECODE)
        self.state = self.model.get_state_for_audio_prompt(prompt)

    def stream(self, sentences: list[str], rate: int, stop: threading.Event):
        import soxr

        rs = soxr.ResampleStream(MODEL_SR, rate, 1, dtype="float32", quality="HQ") if rate != MODEL_SR else None
        for s in sentences:
            for chunk in self.model.generate_audio_stream(self.state, s, stop=stop):
                x = chunk.detach().cpu().numpy().reshape(-1).astype(np.float32)
                if rs is not None:
                    x = rs.resample_chunk(x, last=False)
                if x.size:
                    yield (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
                if stop.is_set():
                    return
        if rs is not None:
            tail = rs.resample_chunk(np.zeros(0, np.float32), last=True)
            if tail.size:
                yield (np.clip(tail, -1, 1) * 32767).astype("<i2").tobytes()


def create_app():
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    from ceskyhlas.normalize_cs import normalize, sentences

    ckpt, prompt = os.environ.get("POCKET_CHECKPOINT"), os.environ.get("POCKET_PROMPT")
    if not ckpt or not prompt:
        raise SystemExit("nastavte POCKET_CHECKPOINT a POCKET_PROMPT")
    config = os.environ.get("POCKET_CONFIG", DEFAULT_CONFIG)
    workers = int(os.environ.get("POCKET_WORKERS", "1"))
    torch.set_num_threads(int(os.environ.get("POCKET_THREADS", "4")))
    info = {"checkpoint_sha256": sha256_12(ckpt), "prompt": Path(prompt).name, "decode": DECODE}
    pool: queue.Queue[Worker] = queue.Queue()
    for _ in range(workers):
        pool.put(Worker(config, ckpt, prompt))
    phrases = PhraseCache(os.environ.get("POCKET_PHRASES"))
    app = FastAPI(title="Český TTS (Pocket TTS)")

    @app.get("/health")
    def health():
        return {"ok": True, **info, "workers": workers, "free": pool.qsize(),
                "phrases": phrase_status(phrases.info(), info["checkpoint_sha256"], info["prompt"])}

    @app.post("/normalize")
    def do_normalize(req: TextRequest):
        t = normalize(req.text)
        return {"text": t, "sentences": sentences(t)}

    @app.post("/tts")
    def tts(req: TTSRequest):
        if req.rate not in (8000, 16000, 24000):
            raise HTTPException(400, "rate musí být 8000, 16000 nebo 24000")
        text = normalize(req.text) if req.normalize else req.text
        if re.search(r"\d", text):
            raise HTTPException(400, "po normalizaci zůstaly číslice")
        sents = sentences(text)
        if not sents:
            raise HTTPException(400, "prázdný text")
        media = f"audio/L16;rate={req.rate};channels=1"
        hit = phrases.get(text)
        if hit is not None:
            pid, take, audio, sr = hit
            return StreamingResponse(pcm_chunks(audio, sr, req.rate), media_type=media,
                                     headers={"X-Sample-Rate": str(req.rate), "X-Phrase": f"{pid}#{take}"})
        try:
            worker = pool.get(timeout=float(os.environ.get("POCKET_QUEUE_S", "10")))
        except queue.Empty:
            raise HTTPException(503, "všechny workery jsou obsazené")
        stop = threading.Event()

        def body():
            try:
                yield from worker.stream(sents, req.rate, stop)
            finally:
                stop.set()
                pool.put(worker)

        return StreamingResponse(body(), media_type=media, headers={"X-Sample-Rate": str(req.rate)})

    return app


def main() -> None:
    import uvicorn

    from ceskyhlas.auth import TokenAuth, check_config

    bind = os.environ.get("TTS_BIND", "127.0.0.1")
    token = check_config(bind, os.environ.get("TTS_TOKEN"))   # otevřená adresa bez tokenu = odmítnuto
    uvicorn.run(TokenAuth(create_app(), token), host=bind, port=int(os.environ.get("PORT", "8765")),
                log_level="warning")


if __name__ == "__main__":
    main()
