"""Jak dobře syntéza vyslovuje jednotlivé české hlásky (r, ř, š, ž, z, č, ť, ď, ň). MIT licence.

  python -m ceskyhlas.phoneme_metrics <složka s wav> <texty.txt> --out phonemes.json

texty.txt: jeden řádek = jeden text v pořadí souborů seřazených podle jména (jako ceskyhlas.asr_wer).

Reference = fonémy textu z espeak-ng (příkazová řádka `espeak-ng -v cs -q --ipa`, musí být nainstalovaný).
Hypotéza = rozpoznávač fonémů facebook/wav2vec2-xlsr-53-espeak-cv-ft (Apache-2.0; XLSR-53 doladěný na
fonémy ve stylu espeak), greedy CTC. Obojí se normalizuje stejně (NFC; bez přízvuku, interpunkce a mezer;
afrikáty rozdělené na dvě hlásky; neznělé ř = ř; délka zůstává u samohlásky) a zarovná Levenshteinem.

Metriky na skupinu nahrávek:
  recall_<hláska>   podíl referenčních výskytů rozpoznaných jako ta hláska; del_<hláska> = podíl spolknutých
  recall_ř_fric     ř uznané, když ho rozpoznávač zapíše jako jakoukoli ř/š/ž frikativu. Striktní recall ř je
                    k ničemu: rozpoznávač píše r̝ zřídka i u skutečného mluvčího
  ř_to_s            podíl ř slyšených jako s nebo z – typická chyba syntézy („tři“ -> „tsi“)
  r_after_cons      r hned po souhlásce („pro“, „dobrý“): podíl smazaných a slyšených jako j („Dobjej den“)
Rozpoznávač je nedokonalý: měřte vždy i skutečného mluvčího stejnou cestou a porovnávejte s ním, ne s nulou.
Jeden seed syntézy hýbe r a ř o ±10 bodů – měřte aspoň dva seedy a ~150+ výskytů ř.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
DROP = set("ˈˌ.,;:!?-'\" |0123456789")
TARGETS = {"r": "r", "ř": "r̝", "ž": "ʒ", "z": "z", "ť": "c", "ď": "ɟ", "ň": "ɲ"}
R_FRIC = {"r̝", "ʃ", "ʒ", "ʂ", "ʐ"}
VOWELS = set("aeiouyɛɪʊəɔæ")


def phones(ipa: str) -> list[str]:
    s = unicodedata.normalize("NFD", ipa)
    s = s.replace("͡", "").replace("͜", "")      # spojovníky: afrikáta = dvě hlásky
    s = s.replace("r̝̊", "r̝")                  # neznělé ř = ř
    out: list[str] = []
    for ch in s:
        if ch in DROP or ch.isspace():
            continue
        if unicodedata.category(ch) == "Mn" or ch in "ːˑʲʰ":
            if out:
                out[-1] += ch
            continue
        out.append(ch)
    return [unicodedata.normalize("NFC", p) for p in out]


def espeak_ipa(text: str) -> str:
    r = subprocess.run(["espeak-ng", "-v", "cs", "-q", "--ipa", text], capture_output=True, text=True, check=True)
    return re.sub(r"\([^)]+\)", "", " ".join(r.stdout.split()))   # bez značek přepnutí jazyka "(en)"


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """Levenshteinovo zarovnání -> páry (ref, hyp); '' = smazání / vložení."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((ref[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", hyp[j - 1]))
            j -= 1
    return pairs[::-1]


def metrics(utts: list[tuple[list[str], list[str]]]) -> dict:
    """utts = [(referenční fonémy, rozpoznané fonémy)] -> metriky popsané v hlavičce modulu."""
    hit, tot, dele, subs = Counter(), Counter(), Counter(), {}
    fric = rs = r_cl = r_cl_del = r_cl_j = 0
    for ref, hyp in utts:
        pairs = [p for p in align(ref, hyp) if p[0]]
        for i, (a, b) in enumerate(pairs):
            keys = [k for k, ph in TARGETS.items() if a == ph]
            if a == "t" and i + 1 < len(pairs) and pairs[i + 1][0] == "ʃ":   # č = t + ʃ
                b2 = pairs[i + 1][1]
                tot["č"] += 1
                hit["č"] += b == "t" and b2 == "ʃ"
                dele["č"] += b == "" and b2 == ""
            if a == "ʃ" and not (i > 0 and pairs[i - 1][0] == "t"):
                keys.append("š")
            for k in keys:
                tot[k] += 1
                hit[k] += a == b
                dele[k] += b == ""
                if a != b:
                    subs.setdefault(k, Counter())[b or "∅"] += 1
            if a == TARGETS["ř"]:
                fric += b in R_FRIC
                rs += b in ("s", "z")
            if a == "r" and i > 0 and pairs[i - 1][0][0] not in VOWELS:
                r_cl += 1
                r_cl_del += b == ""
                r_cl_j += b == "j"
    out: dict = {}
    for k in list(TARGETS) + ["š", "č"]:
        out[f"recall_{k}"] = round(hit[k] / tot[k], 3) if tot[k] else None
        out[f"del_{k}"] = round(dele[k] / tot[k], 3) if tot[k] else None
        out[f"n_{k}"] = tot[k]
        out[f"sub_{k}"] = "; ".join(f"{p}:{c}" for p, c in subs.get(k, Counter()).most_common(3))
    out["recall_ř_fric"] = round(fric / tot["ř"], 3) if tot["ř"] else None
    out["ř_to_s"] = round(rs / tot["ř"], 3) if tot["ř"] else None
    out["r_after_cons"] = {"n": r_cl, "del": round(r_cl_del / r_cl, 3) if r_cl else None,
                           "to_j": round(r_cl_j / r_cl, 3) if r_cl else None}
    out["per"] = round(sum(sum(a != b for a, b in align(r, h)) for r, h in utts) / max(1, sum(len(r) for r, _ in utts)), 4)
    return out


class Recognizer:
    def __init__(self, threads: int = 4) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        torch.set_num_threads(threads)
        vocab = json.load(open(hf_hub_download(MODEL, "vocab.json"), encoding="utf-8"))
        self.inv = {i: t for t, i in vocab.items()}
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()

    def __call__(self, wav: Path) -> list[str]:
        import numpy as np
        import soundfile as sf
        import soxr
        import torch

        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = soxr.resample(x.mean(axis=1), sr, 16000).astype(np.float32)
        with torch.inference_mode():
            ids = self.model(self.fe(x, sampling_rate=16000, return_tensors="pt").input_values).logits[0].argmax(-1)
        toks, prev = [], None
        for t in ids.tolist():
            if t != prev and self.inv[t] not in ("<pad>", "<s>", "</s>", "<unk>"):
                toks.append(self.inv[t])
            prev = t
        return phones(" ".join(toks))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wavs")
    ap.add_argument("texts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(Path(a.wavs).glob("*.wav"))
    texts = [l.strip() for l in Path(a.texts).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(files) != len(texts):
        sys.exit(f"{len(files)} souborů, ale {len(texts)} textů")
    rec = Recognizer()
    utts = [(phones(espeak_ipa(t)), rec(f)) for f, t in zip(files, texts)]
    res = metrics(utts)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("per", "recall_r", "recall_ř_fric", "ř_to_s", "r_after_cons")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
