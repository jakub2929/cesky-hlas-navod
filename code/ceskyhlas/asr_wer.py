"""Srozumitelnost syntézy: přepis lokálním ASR (NVIDIA Parakeet TDT 0.6b v3, umí česky) a WER / CER. MIT licence.

  python -m ceskyhlas.asr_wer <složka s wav> <texty.txt> [--phone alaw]

texty.txt: jeden řádek = jeden text, v pořadí souborů seřazených podle jména (s001.wav, s002.wav, ...).
--phone alaw|ulaw nejdřív pošle zvuk simulovanou telefonní linkou (ceskyhlas.phone_sim). Text se porovnává
malými písmeny bez interpunkce. Pozor: na klipech kratších než ~1,5 s (jedno slovo) Parakeet často vrátí
nic nebo anglické slovo – WER tam neříká nic, poslouchejte.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def wer(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps) if normalize(r)]
    if not pairs:
        return {"wer": None, "cer": None, "utterances": 0}
    o = jiwer.process_words([p[0] for p in pairs], [p[1] for p in pairs])
    return {"wer": round(o.wer, 4), "cer": round(jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs]), 4),
            "utterances": len(pairs), "ref_words": sum(len(p[0].split()) for p in pairs)}


def transcribe(paths: list[str], batch_size: int = 8, model_name: str = ASR_MODEL) -> list[str]:
    """Přepis souborů 16 kHz mono (NeMo; GPU, je-li k dispozici)."""
    import torch
    from nemo.collections.asr.models import ASRModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASRModel.from_pretrained(model_name, map_location=dev).eval()
    out = model.transcribe(paths, batch_size=batch_size, verbose=False)
    if isinstance(out, tuple):   # starší NeMo vrací (best, all)
        out = out[0]
    return [o.text if hasattr(o, "text") else str(o) for o in out]


def to_16k(files: list[Path], phone: str | None) -> list[str]:
    import soundfile as sf

    from ceskyhlas.phone_sim import PHONE_SR, phone_line, resample

    tmp = Path(tempfile.mkdtemp())
    out = []
    for i, f in enumerate(files):
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if phone:
            x, sr = phone_line(x, sr, phone), PHONE_SR
        p = tmp / f"{i:05d}.wav"
        sf.write(str(p), resample(x, sr, 16000), 16000)
        out.append(str(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wavs")
    ap.add_argument("texts")
    ap.add_argument("--phone", choices=["alaw", "ulaw"])
    ap.add_argument("--out", help="JSON s přepisy a výsledkem")
    a = ap.parse_args()
    files = sorted(Path(a.wavs).glob("*.wav"))
    refs = [l.strip() for l in Path(a.texts).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(files) != len(refs):
        sys.exit(f"{len(files)} souborů, ale {len(refs)} textů")
    hyps = transcribe(to_16k(files, a.phone))
    res = wer(refs, hyps)
    if a.out:
        Path(a.out).write_text(json.dumps({**res, "hyps": hyps}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
