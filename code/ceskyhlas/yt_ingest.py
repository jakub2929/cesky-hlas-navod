"""Z videí jednoho mluvčího (YouTube apod.) dataset pro TTS: jen detekce a výběr, žádná separace. MIT licence.

  python -m ceskyhlas.yt_ingest prepare  --in <složka s videi/audiem> --work <pracovní složka>
  python -m ceskyhlas.yt_ingest profile  --ref <složka s čistými ukázkami mluvčího (wav)> --work <...>
  python -m ceskyhlas.yt_ingest analyze  --work <...>          # GPU doporučeno (Sortformer, TitaNet, AST, Parakeet)
  python -m ceskyhlas.yt_ingest select   --work <...> --out <dataset> [--sr 24000]

Stahování videí (např. yt-dlp) řešíte sami – a jen u mluvčího, který s klonováním hlasu výslovně souhlasil.

prepare  každý soubor -> <work>/src/<id>.flac (mono, původní vzorkování; jediný ztrátový krok je kodek zdroje).
profile  hlasový otisk mluvčího: průměr TitaNet embeddingů z 5–20 čistých ukázek, které jste ověřili poslechem.
analyze  na video: diarizace (NVIDIA Sortformer, až 4 mluvčí) -> energetický VAD s hysterezí -> promluvy
         stříhané jen v pauzách (úsek se přiřadí mluvčímu, který v něm převládá; sousední úseky téhož mluvčího
         se spojují do MAX_S, pauza >= END_SIL_S promluvu vždy ukončí; úsek delší než MAX_S bez pauzy se zahodí,
         protože by se musel střihnout uprostřed slova) -> pro každou promluvu: podobnost s otiskem (i po
         simulované telefonní lince), zvukové události AudioSet (AST: hudba, klikání...), přepis Parakeet
         s konfidencí -> <work>/rows/<id>.jsonl (průběžně, přerušený běh pokračuje).
select   v každém videu vybere lokálního mluvčího s nejvyšší mediánovou podobností (aspoň MIN_SIM), projde
         filtry v pořadí FUNNEL, převede čísla na slova (ceskyhlas.normalize_cs) a exportuje LJSpeech:
         <out>/wavs/<id>.wav (mono PCM16, úroveň aktivních snímků −20 dBFS, špička <= −1 dBFS) + metadata.csv
         ("id|text") + funnel.csv (kolik minut odpadlo na kterém filtru).
Pak spusťte ceskyhlas.ctc_filter na výsledný dataset (nezávislá kontrola textu proti zvuku).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np

MEDIA_EXT = {".webm", ".m4a", ".mp4", ".mkv", ".opus", ".ogg", ".mp3", ".aac", ".wav", ".flac", ".mov"}
SR16 = 16000

# VAD (energie po 10 ms, hystereze mezi LO a HI nad šumovým dnem)
HOP_S = 0.01
HPF_HZ = 70.0
FLOOR_PCT, LOUD_PCT = 10, 97
HI_FRAC, HI_MIN_DB = 0.45, 10.0
LO_FRAC, LO_MIN_DB = 0.25, 6.0
MIN_SPEECH_S = 0.25

# segmentace
CUT_SIL_S = 0.2          # pauza, ve které se smí střihnout (okluze plozivy je < 0,1 s)
END_SIL_S = 0.6          # tak dlouhá pauza promluvu vždy ukončí
MIN_S, MAX_S = 2.0, 15.0
EDGE_PAD_S = 0.15        # okraj do pauzy, nejvýš polovina pauzy
EDGE_PROBE_S = 0.03      # krajních 30 ms musí být pod úrovní LO (jinak je střih uprostřed hlásky)

# modely (všechny lokálně, bez cloudu)
DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"   # CC-BY-4.0
DIAR_THRESHOLD, DIAR_FRAME_S = 0.5, 0.08
SPK_MODEL = "nvidia/speakerverification_en_titanet_large"
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"       # BSD-3, jen detekce
AST_WIN_S = 10.24
ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
SPEECH_LABELS = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
                 "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Inside, small room",
                 "Inside, large room or hall", "Silence"}

# filtry výběru
MIN_SIM = 0.40           # podobnost po telefonní lince; nastavte nad nejvyšší hodnotu „cizích“ mluvčích
MAX_OVERLAP = 0.02       # podíl snímků, kde mluví někdo další
MAX_MUSIC_P = 0.50       # sigmoid AST >= 0,5 = klasifikátor říká „je slyšet“
MAX_EVENT_P = 0.50
CLIP_PEAK_DBFS = -0.1
MIN_CONFIDENCE = 0.80
MIN_CHARS = 8
MAX_FOREIGN = 2          # cizí slova v promluvě (jejich česká výslovnost z písmen neplatí)
FUNNEL = ["duration", "speaker", "overlap", "midword_edge", "music", "sound_event", "similarity", "clipping", "asr",
          "foreign", "digits"]
TARGET_RMS_DBFS, PEAK_MAX_DBFS = -20.0, -1.0


# ---------------------------------------------------------------- zvuk

def highpass(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    if len(x) == 0:
        return x
    return sosfilt(butter(4, HPF_HZ, btype="highpass", fs=sr, output="sos"), x).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(round(sr * HOP_S)))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    fr = x[: n * hop].reshape(n, hop).astype(np.float64)
    return (10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)).astype(np.float32)


def power_db(db: np.ndarray) -> float:
    if not len(db):
        return -100.0
    return float(10 * np.log10(np.mean(10 ** (db.astype(np.float64) / 10)) + 1e-10))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


@dataclass
class Vad:
    segments: list[tuple[float, float]]
    lo_db: float
    db: np.ndarray


def split_speech(x: np.ndarray, sr: int, min_sil_s: float = CUT_SIL_S) -> Vad:
    """Energetický VAD s hysterezí: úsek nad LO, který někde přesáhne HI, je řeč; úseky bližší než min_sil_s se
    spojí; kratší než MIN_SPEECH_S se zahodí. x má být po horní propusti."""
    db = frame_db(x, sr)
    if not len(db):
        return Vad([], -100.0, db)
    floor = float(np.percentile(db, FLOOR_PCT))
    rng = max(float(np.percentile(db, LOUD_PCT)) - floor, 0.0)
    hi = floor + max(HI_MIN_DB, HI_FRAC * rng)
    lo = floor + max(LO_MIN_DB, LO_FRAC * rng)
    islands = [(s, e) for s, e in runs(db > lo) if (db[s:e] > hi).any()]
    gap = int(round(min_sil_s / HOP_S))
    merged: list[tuple[int, int]] = []
    for s, e in islands:
        if merged and s - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    segs = [(s * HOP_S, e * HOP_S) for s, e in merged if e - s >= int(round(MIN_SPEECH_S / HOP_S))]
    return Vad(segs, lo, db)


def build_utterances(chunks: list[tuple[float, float]], act: np.ndarray, frame_s: float) -> tuple[list[dict], list[dict]]:
    """chunks = úseky řeči z VAD (s), act = (snímky, mluvčí) bool. Vrací (promluvy, zahozené úseky)."""
    n_fr = act.shape[0]
    info = []
    for s, e in chunks:
        a = min(int(s / frame_s), n_fr - 1)
        b = min(max(a + 1, int(np.ceil(e / frame_s))), n_fr)
        cnt = act[a:b].sum(axis=0)
        info.append({"s": s, "e": e, "spk": int(np.argmax(cnt)) if cnt.max() > 0 else -1})
    utts, dropped, cur = [], [], []

    def flush() -> None:
        if cur:
            utts.append({"s": cur[0]["s"], "e": cur[-1]["e"], "spk": cur[0]["spk"], "n_chunks": len(cur)})
            cur.clear()

    for c in info:
        if c["e"] - c["s"] > MAX_S - 2 * EDGE_PAD_S:
            flush()
            dropped.append(c)
            continue
        if cur and (c["spk"] != cur[-1]["spk"] or c["s"] - cur[-1]["e"] >= END_SIL_S
                    or c["e"] - cur[0]["s"] > MAX_S - 2 * EDGE_PAD_S):
            flush()
        cur.append(c)
    flush()
    return utts, dropped


def measure(x: np.ndarray, sr: int, vad: Vad, utts: list[dict], act: np.ndarray, frame_s: float,
            raw: np.ndarray | None = None) -> list[dict]:
    """Okraje do pauz, překryv s dalšími mluvčími, úrovně; vrací řádky bez modelových měření. x = signál po horní
    propusti (VAD), raw = původní signál (špička pro klipování; výchozí x)."""
    raw = x if raw is None else raw
    ends = np.array([e for _s, e in vad.segments])
    starts = np.array([s for s, _e in vad.segments])
    probe = max(1, int(round(EDGE_PROBE_S / HOP_S)))
    rows = []
    for u in utts:
        before, after = ends[ends <= u["s"]], starts[starts >= u["e"]]
        prev_e = float(before.max()) if len(before) else 0.0
        next_s = float(after.min()) if len(after) else len(x) / sr
        s = max(u["s"] - EDGE_PAD_S, (prev_e + u["s"]) / 2, 0.0)
        e = min(u["e"] + EDGE_PAD_S, (u["e"] + next_s) / 2, len(x) / sr)
        a_ = act[int(s / frame_s): min(int(np.ceil(e / frame_s)), len(act))]
        if u["spk"] >= 0 and act.shape[1] > 1:
            other = np.delete(a_, u["spk"], axis=1).any(axis=1)
        else:
            other = a_.any(axis=1)
        db = vad.db[int(s / HOP_S): int(e / HOP_S)]
        seg = raw[int(s * sr): int(e * sr)]
        rows.append({"start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3), "local_spk": u["spk"],
                     "other_frac": round(float(other.mean()) if len(other) else 0.0, 3),
                     "edge_head_db": round(float(db[:probe].max()) - vad.lo_db, 1) if len(db) else 0.0,
                     "edge_tail_db": round(float(db[-probe:].max()) - vad.lo_db, 1) if len(db) else 0.0,
                     "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(seg)) + 1e-9)), 2) if len(seg) else -100.0})
    return rows


# ---------------------------------------------------------------- příkazy

def cmd_prepare(a) -> None:
    src = Path(a.work) / "src"
    src.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.inp).iterdir() if p.suffix.lower() in MEDIA_EXT)
    index = {}
    for p in files:
        vid = hashlib.sha1(p.name.encode()).hexdigest()[:10]
        out = src / f"{vid}.flac"
        index[vid] = p.name
        if out.exists():
            continue
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(p), "-vn", "-ac", "1",
                        "-c:a", "flac", "-sample_fmt", "s32", str(out)], check=True)
        print("prepare", p.name, "->", out.name, flush=True)
    (Path(a.work) / "videos.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} souborů v {src}")


def _models() -> dict:
    import torch
    from nemo.collections.asr.models import ASRModel, EncDecSpeakerLabelModel, SortformerEncLabelModel
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    diar = SortformerEncLabelModel.from_pretrained(DIAR_MODEL, map_location=dev).eval()
    sm = diar.sortformer_modules   # „very high latency“ (offline) nastavení z karty modelu
    sm.chunk_len, sm.chunk_right_context, sm.fifo_len = 340, 40, 40
    sm.spkcache_update_period, sm.spkcache_len = 300, 188
    return {"dev": dev, "diar": diar,
            "spk": EncDecSpeakerLabelModel.from_pretrained(SPK_MODEL, map_location=dev).eval(),
            "ast_fe": ASTFeatureExtractor.from_pretrained(AST_MODEL),
            "ast": ASTForAudioClassification.from_pretrained(AST_MODEL).to(dev).eval(),
            "asr": ASRModel.from_pretrained(ASR_MODEL, map_location=dev).eval()}


def embed(m: dict, x16: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        _, e = m["spk"].forward(input_signal=torch.from_numpy(x16.astype(np.float32))[None].to(m["dev"]),
                                input_signal_length=torch.tensor([len(x16)]).to(m["dev"]))
    v = e[0].float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)


def cmd_profile(a) -> None:
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = {"dev": dev, "spk": EncDecSpeakerLabelModel.from_pretrained(SPK_MODEL, map_location=dev).eval()}
    vs = []
    for f in sorted(Path(a.ref).glob("*.wav")):
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        vs.append(embed(m, resample(x.mean(axis=1), sr, SR16)))
    if len(vs) < 3:
        sys.exit("potřeba aspoň 3 ukázky (doporučeno 5–20)")
    c = np.mean(vs, axis=0)
    np.save(Path(a.work) / "profile.npy", c / np.linalg.norm(c))
    print(f"otisk z {len(vs)} ukázek, podobnost ukázek s otiskem: "
          f"{np.round([float(v @ (c / np.linalg.norm(c))) for v in vs], 3).tolist()}")


def cmd_analyze(a) -> None:
    import soundfile as sf
    import torch

    from ceskyhlas.phone_sim import PHONE_SR, phone_line

    work = Path(a.work)
    prof_f = work / "profile.npy"
    if not prof_f.exists():
        sys.exit("nejdřív `profile` (hlasový otisk mluvčího)")
    profile = np.load(prof_f)
    m = _models()
    labels = m["ast"].config.id2label
    music_i = next(i for i, n in labels.items() if n == "Music")
    other_i = [i for i, n in labels.items() if n not in SPEECH_LABELS and n != "Music"]
    (work / "rows").mkdir(exist_ok=True)
    for f in sorted((work / "src").glob("*.flac")):
        out = work / "rows" / f"{f.stem}.jsonl"
        if out.exists():
            continue
        x, sr = sf.read(str(f), dtype="float32")
        xh = highpass(x, sr)
        x16 = resample(x, sr, SR16)
        with torch.inference_mode():
            _segs, probs = m["diar"].diarize(audio=[x16], batch_size=1, sample_rate=SR16,
                                             include_tensor_outputs=True, verbose=False)
        p = probs[0]
        p = (p.detach().float().cpu().numpy() if hasattr(p, "detach") else np.asarray(p, np.float32))
        act = p.reshape(-1, p.shape[-1]) >= DIAR_THRESHOLD
        vad = split_speech(xh, sr)
        utts, dropped = build_utterances(vad.segments, act, DIAR_FRAME_S)
        rows = measure(xh, sr, vad, utts, act, DIAR_FRAME_S, raw=x)
        tmp = work / "tmp16"
        tmp.mkdir(exist_ok=True)
        paths = []
        for i, r in enumerate(rows):
            r["id"] = f"{f.stem}_{i:04d}"
            y16 = resample(xh[int(r["start"] * sr): int(r["end"] * sr)], sr, SR16)
            r["sim_raw"] = round(float(embed(m, y16) @ profile), 4)
            r["sim_phone"] = round(float(embed(m, resample(phone_line(y16, SR16), PHONE_SR, SR16)) @ profile), 4)
            win = int(AST_WIN_S * SR16)
            parts = [y16] if len(y16) <= win else [y16[:win], y16[-win:]]
            feats = m["ast_fe"](parts, sampling_rate=SR16, return_tensors="pt").input_values.to(m["dev"])
            with torch.inference_mode():
                q = torch.sigmoid(m["ast"](feats).logits).float().cpu().numpy().max(axis=0)
            oi = other_i[int(np.argmax(q[other_i]))]
            r.update(music_p=round(float(q[music_i]), 4), event=labels[oi], event_p=round(float(q[oi]), 4))
            pth = tmp / f"{r['id']}.wav"
            sf.write(str(pth), np.clip(y16, -1, 1), SR16)
            paths.append(str(pth))
        if paths:
            hyps = m["asr"].transcribe(paths, batch_size=8, return_hypotheses=True, verbose=False)
            hyps = hyps[0] if isinstance(hyps, tuple) else hyps
            for r, h in zip(rows, hyps):
                tc = getattr(h, "token_confidence", None)
                r["text"] = h.text.strip()
                r["confidence"] = round(float(np.mean([float(c) for c in tc])), 4) if tc else None
        for pth in paths:
            os.remove(pth)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"analyze {f.stem}: {len(rows)} promluv, bez pauzy zahozeno {len(dropped)}", flush=True)


def is_foreign(tok: str) -> bool:
    """Hrubý odhad cizího slova (q, w, x mimo běžná česká slova, anglické koncovky). Rozšiřte vlastním seznamem."""
    t = tok.lower().strip(".,;:!?\"'()…")
    return bool(re.search(r"[qw]|x(?!t)|th|sh|ee|oo|ing$|tion$", t)) and not re.fullmatch(r"(ex|max|taxi|text)\w*", t)


def reject(r: dict, spk: int | None, sim_ok: bool) -> str:
    if not (MIN_S <= r["duration"] <= MAX_S):
        return "duration"
    if spk is None or r["local_spk"] != spk or not sim_ok:
        return "speaker"
    if r["other_frac"] > MAX_OVERLAP:
        return "overlap"
    if r["edge_head_db"] > 0 or r["edge_tail_db"] > 0:
        return "midword_edge"
    if r["music_p"] >= MAX_MUSIC_P:
        return "music"
    if r["event_p"] >= MAX_EVENT_P:
        return "sound_event"
    if r["sim_phone"] < MIN_SIM:
        return "similarity"
    if r["peak_dbfs"] >= CLIP_PEAK_DBFS:
        return "clipping"
    if (r.get("confidence") or 0) < MIN_CONFIDENCE or len(r.get("text", "")) < MIN_CHARS:
        return "asr"
    if sum(is_foreign(t) for t in r["text"].split()) > MAX_FOREIGN:
        return "foreign"
    return ""


def level(y: np.ndarray, sr: int) -> np.ndarray:
    vad = split_speech(y, sr)
    db = vad.db
    act = power_db(db[db > vad.lo_db])
    g = 10 ** ((TARGET_RMS_DBFS - act) / 20)
    peak = float(np.max(np.abs(y))) * g
    if peak > 10 ** (PEAK_MAX_DBFS / 20):
        g *= 10 ** (PEAK_MAX_DBFS / 20) / peak
    return (y * g).astype(np.float32)


def cmd_select(a) -> None:
    import soundfile as sf

    from ceskyhlas.normalize_cs import normalize

    work, out = Path(a.work), Path(a.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    funnel = {k: 0.0 for k in FUNNEL}
    kept_min, meta = 0.0, []
    for f in sorted((work / "rows").glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        by: dict[int, list[float]] = {}
        for r in rows:
            by.setdefault(r["local_spk"], []).append(r["sim_phone"])
        cand = [(float(np.median(v)), k) for k, v in by.items() if k >= 0 and len(v) >= 3]
        med, spk = max(cand) if cand else (0.0, None)
        x = sr = None
        for r in rows:
            why = reject(r, spk, med >= MIN_SIM)
            text = normalize(r.get("text", "")) if not why else ""
            if not why and re.search(r"\d", text):
                why = "digits"
            if why:
                funnel[why] += r["duration"] / 60
                continue
            if x is None:
                x, sr = sf.read(str(work / "src" / f"{f.stem}.flac"), dtype="float32")
                x = highpass(x, sr)
            y = level(resample(x[int(r["start"] * sr): int(r["end"] * sr)], sr, a.sr), a.sr)
            sf.write(str(out / "wavs" / f"{r['id']}.wav"), y, a.sr, subtype="PCM_16")
            meta.append(f"{r['id']}|{text}")
            kept_min += r["duration"] / 60
        print(f"{f.stem}: mluvčí {spk} (medián podobnosti {med:.3f})", flush=True)
    (out / "metadata.csv").write_text("\n".join(meta) + "\n", encoding="utf-8")
    with open(out / "funnel.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filtr", "minut_odpadlo"])
        w.writerows([k, round(v, 2)] for k, v in funnel.items())
        w.writerow(["ponecháno", round(kept_min, 2)])
    print(f"ponecháno {len(meta)} promluv, {kept_min:.1f} min -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--work", required=True)
    q = sub.add_parser("profile")
    q.add_argument("--ref", required=True)
    q.add_argument("--work", required=True)
    r = sub.add_parser("analyze")
    r.add_argument("--work", required=True)
    s = sub.add_parser("select")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--sr", type=int, default=24000, help="24000 pro Pocket TTS, 22050 pro Piper")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "profile": cmd_profile, "analyze": cmd_analyze, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
