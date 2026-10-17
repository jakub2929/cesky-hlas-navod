"""Nezávislá kontrola, že text odpovídá zvuku: CTC forced alignment českým wav2vec2. MIT licence.

  python -m ceskyhlas.ctc_filter score  <dataset> [--ref <referenční dataset>] --out scores.csv
  python -m ceskyhlas.ctc_filter select scores.csv --out keep.csv [--max-min 180] [--min-min 120]

<dataset> = složka ve formátu LJSpeech: metadata.csv ("id|text", text už bez číslic) + wavs/<id>.wav.

Proč: přepisy dělá ASR (Parakeet). Když stejný model text zároveň kontroluje, chyby si „potvrdí“. Proto
skóre počítá jiný model, který přepisy nedělal: comodoro/wav2vec2-xls-r-300m-cs-250 (XLS-R 300M doladěný na
češtinu, Apache-2.0), greedy CTC posteriory po 20 ms, bez jazykového modelu.

Skóre: Viterbi přes standardní CTC topologii (blank mezi tokeny, opakovaný token potřebuje blank); pro každé
písmeno průměrná log-pravděpodobnost jeho snímků na nejlepší cestě; skóre promluvy = průměr přes písmena
(blíž nule = lepší shoda). low_share = podíl písmen s pravděpodobností < 0,1. Text, který se do zvuku
nevejde, má skóre -inf.

Výběr (select): seřadit podle skóre a brát od nejlepších, dokud nedosáhnete --max-min minut nebo skóre
nepadne pod 10. percentil referenční sady (--ref: např. syntetický čistý hlas čtoucí tytéž texty = dokonalá
shoda textu); když projde méně než --min-min minut, vezme se prostě nejlepších --min-min minut.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

MODEL = "comodoro/wav2vec2-xls-r-300m-cs-250"
SR = 16000
LOW_P = 0.1


def read_dataset(d: Path) -> list[tuple[str, Path, str]]:
    out = []
    for line in (d / "metadata.csv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = line.split("|")
            uid = parts[0].removesuffix(".wav")
            out.append((uid, d / "wavs" / f"{uid}.wav", parts[-1]))
    return out


def norm_text(text: str, vocab: dict[str, int]) -> tuple[list[int], int]:
    """Id tokenů textu v abecedě modelu ('|' mezi slovy) a počet vypuštěných písmen."""
    ids, dropped = [], 0
    for word in text.lower().split():
        chars = [c for c in word if c in vocab and c != "|"]
        dropped += sum(1 for c in word if c not in vocab and c.isalpha())
        if chars:
            if ids:
                ids.append(vocab["|"])
            ids += [vocab[c] for c in chars]
    return ids, dropped


def ctc_viterbi(logp: np.ndarray, tokens: list[int], blank: int) -> list[int] | None:
    """Nejlepší CTC cesta pro `tokens` přes log-posteriory (T, V). Pro každý snímek index tokenu, který vydává
    (-1 = blank), nebo None, když se text do T snímků nevejde."""
    T, S = logp.shape[0], 2 * len(tokens) + 1
    ext = [blank] * S
    ext[1::2] = tokens
    need = len(tokens) + sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    if need > T:
        return None
    em = logp[:, ext]
    dp = np.full((T, S), -np.inf)
    bp = np.zeros((T, S), dtype=np.int8)   # 0 zůstat, 1 z s-1, 2 z s-2
    dp[0, 0] = em[0, 0]
    if S > 1:
        dp[0, 1] = em[0, 1]
    skip = np.zeros(S, dtype=bool)
    for s in range(2, S):
        skip[s] = ext[s] != blank and ext[s] != ext[s - 2]
    for t in range(1, T):
        prev = dp[t - 1]
        c1 = np.concatenate(([-np.inf], prev[:-1]))
        c2 = np.where(skip, np.concatenate(([-np.inf, -np.inf], prev[:-2])), -np.inf)
        stack = np.stack([prev, c1, c2])
        bp[t] = np.argmax(stack, axis=0)
        dp[t] = stack[bp[t], np.arange(S)] + em[t]
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if not np.isfinite(dp[-1, s]):
        return None
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = (s - 1) // 2 if s % 2 == 1 else -1
        s -= int(bp[t, s])
    return path


def score_one(logp: np.ndarray, tokens: list[int], blank: int, delim: int) -> dict:
    path = ctc_viterbi(logp, tokens, blank) if tokens else None
    if path is None:
        return {"score": -math.inf, "low_share": 1.0}
    per: dict[int, list[float]] = {}
    for t, k in enumerate(path):
        if k >= 0:
            per.setdefault(k, []).append(float(logp[t, tokens[k]]))
    char = [float(np.mean(v)) for k, v in sorted(per.items()) if tokens[k] != delim]
    if not char:
        return {"score": -math.inf, "low_share": 1.0}
    return {"score": round(float(np.mean(char)), 4),
            "low_share": round(float(np.mean([math.exp(c) < LOW_P for c in char])), 4)}


class Scorer:
    def __init__(self, threads: int = 4) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        torch.set_num_threads(threads)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = Wav2Vec2Processor.from_pretrained(MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(self.dev).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank, self.delim = self.proc.tokenizer.pad_token_id, self.vocab["|"]

    def __call__(self, wav: Path, text: str) -> dict:
        import soundfile as sf
        import soxr
        import torch

        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = soxr.resample(x.mean(axis=1), sr, SR).astype(np.float32)
        feats = self.proc(x, sampling_rate=SR, return_tensors="pt").input_values.to(self.dev)
        with torch.inference_mode():
            logp = torch.log_softmax(self.model(feats).logits[0].float(), dim=-1).cpu().numpy()
        tokens, dropped = norm_text(text, self.vocab)
        return {**score_one(logp, tokens, self.blank, self.delim), "dropped_chars": dropped,
                "minutes": round(len(x) / SR / 60, 4)}


def cmd_score(a) -> None:
    sc = Scorer()
    rows = []
    for name, d in (("data", Path(a.dataset)), ("ref", Path(a.ref) if a.ref else None)):
        if d is None:
            continue
        for uid, wav, text in read_dataset(d):
            rows.append({"set": name, "id": uid, **sc(wav, text)})
    if not any(r["set"] == "data" for r in rows):
        sys.exit(f"{a.dataset}: metadata.csv je prázdné")
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    data = np.array([r["score"] for r in rows if r["set"] == "data"])
    fin = np.isfinite(data)
    print(f"{len(data)} promluv, nevejde se {int((~fin).sum())}, skóre p10/p50/p90 "
          f"{np.percentile(data[fin], [10, 50, 90]).round(3).tolist() if fin.any() else '-'} -> {a.out}")


def select(rows: list[dict], max_min: float, min_min: float) -> tuple[list[dict], str]:
    data = sorted((r for r in rows if r["set"] == "data"), key=lambda r: -float(r["score"]))
    ref = np.array([float(r["score"]) for r in rows if r["set"] == "ref"])
    p10 = float(np.percentile(ref, 10)) if len(ref) else -math.inf
    keep, acc = [], 0.0
    for r in data:
        if acc >= max_min or float(r["score"]) < p10:
            break
        keep.append(r)
        acc += float(r["minutes"])
    rule = f"skóre >= p10 reference ({p10:.3f})" if acc < max_min else f"strop {max_min:.0f} min"
    if acc < min_min:
        keep, acc = [], 0.0
        for r in data:
            if acc >= min_min:
                break
            keep.append(r)
            acc += float(r["minutes"])
        rule = f"podlaha: nejlepších {min_min:.0f} min (nad p10 reference jich prošlo méně)"
    return keep, f"{len(keep)} promluv, {acc:.1f} min, pravidlo: {rule}"


def cmd_select(a) -> None:
    rows = list(csv.DictReader(open(a.scores, encoding="utf-8")))
    if not any(r["set"] == "data" for r in rows):
        sys.exit(f"{a.scores}: žádné promluvy")
    keep, msg = select(rows, a.max_min, a.min_min)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score", "minutes"])
        w.writerows([r["id"], r["score"], r["minutes"]] for r in keep)
    print(msg, "->", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("dataset")
    s.add_argument("--ref")
    s.add_argument("--out", required=True)
    t = sub.add_parser("select")
    t.add_argument("scores")
    t.add_argument("--out", required=True)
    t.add_argument("--max-min", type=float, default=180.0)
    t.add_argument("--min-min", type=float, default=120.0)
    a = ap.parse_args()
    {"score": cmd_score, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
