"""Simulace telefonní linky G.711 (A-law / μ-law) pro hodnocení TTS „po telefonu“. MIT licence.

  python -m ceskyhlas.phone_sim <vstupní složka s wav> <výstupní složka> [--codec alaw|ulaw]

Řetězec: převzorkování na 8 kHz (soxr VHQ) -> pásmová propust 300–3400 Hz (Butterworth 4. řádu, kauzální jako
filtr linky) -> úroveň na lince −20 dBFS RMS (a níž, kdyby špička přesáhla rozsah G.711, takže nic neklipuje)
-> kódování a dekódování G.711 (A-law = PCMA, evropský standard; μ-law = PCMU, USA) -> 8 kHz 16 bit.

Pozor na dvě chyby, které jsme sami udělali: (1) měkký antialiasingový filtr (scipy resample_poly) nechá
v pásmu 3,6–4 kHz o ~15 dB víc energie než soxr; (2) hlasitý TTS výstup s malým crest faktorem po pásmové
propusti překmitne a G.711 ho tvrdě ořízne. Obojí zhorší WER „po telefonu“ víc, než skutečná linka.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import soxr
from scipy.signal import butter, sosfilt

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop  # Python <= 3.12
    except ImportError:  # Python 3.13+: pip install audioop-lts
        import audioop_lts as audioop  # type: ignore[no-redef]

PHONE_SR = 8000
BAND = (300.0, 3400.0)
LEVEL_DBFS = -20.0
PEAK = {"ulaw": 32124 / 32768, "alaw": 32256 / 32768}   # největší amplituda, kterou daný zákon G.711 vyjádří
_SOS = butter(4, BAND, btype="bandpass", fs=PHONE_SR, output="sos")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32)


def phone_line(x: np.ndarray, sr: int, codec: str = "alaw") -> np.ndarray:
    """Zvuk (float, mono) se vzorkováním sr -> co doručí linka G.711: float, 8 kHz."""
    return phone_line_from_8k(resample(x, sr, PHONE_SR), codec)


def phone_line_from_8k(x8: np.ndarray, codec: str = "alaw") -> np.ndarray:
    if codec not in PEAK:
        raise ValueError("codec musí být 'alaw' nebo 'ulaw'")
    y = sosfilt(_SOS, x8).astype(np.float32)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if rms > 0:
        y *= 10 ** (LEVEL_DBFS / 20) / rms
        peak = float(np.max(np.abs(y)))
        if peak > PEAK[codec]:
            y *= PEAK[codec] / peak
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    if codec == "alaw":
        back = audioop.alaw2lin(audioop.lin2alaw(pcm, 2), 2)
    else:
        back = audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2)
    return np.frombuffer(back, dtype="<i2").astype(np.float32) / 32768.0


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--codec", default="alaw", choices=["alaw", "ulaw"])
    a = ap.parse_args()
    files = sorted(Path(a.src).glob("*.wav"))
    if not files:
        sys.exit(f"v {a.src} nejsou žádné .wav")
    Path(a.dst).mkdir(parents=True, exist_ok=True)
    for f in files:
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        sf.write(str(Path(a.dst) / f.name), phone_line(x.mean(axis=1), sr, a.codec), PHONE_SR, subtype="PCM_16")
    print(f"{len(files)} souborů -> {a.dst} (G.711 {a.codec})")


if __name__ == "__main__":
    main()
