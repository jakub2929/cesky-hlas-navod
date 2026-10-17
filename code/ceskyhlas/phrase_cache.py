"""Cache pevných frází: každou opakovanou větu vyrenderovat N×, automaticky seřadit, nechat nejlepší 3. MIT licence.

  python -m ceskyhlas.phrase_cache synth --phrases fraze.txt --checkpoint ckpt.pt --prompt prompt.wav --work <w>
  python -m ceskyhlas.phrase_cache rank  --work <w> --out <cache>        # (jiné prostředí: NeMo + transformers)
  python -m ceskyhlas.phrase_cache set   --out <cache> <id> <take>       # ručně vybrat jiný take (1–3)

Proč: Pocket TTS je stochastický – stejná věta jednou zní dobře, podruhé spolkne hlásku. Věty, které agent
říká pořád dokola (pozdrav, rozloučení, „Moment.“), se proto vyrenderují předem a server je posílá z cache.

fraze.txt: řádky "id|text" (# = komentář). synth: text projde normalizací, seedy 1..N (výchozí 8), dekódování
jako server (temp 0.5, decode steps 1, noise clamp 1.0) -> <w>/<id>/seed<k>.wav + synth.json.
rank: pro každý take CER přepisu (Parakeet; u jednoslovných frází se přeskočí – ASR na ~1 s klipu selhává),
počet chyb r / ř (ceskyhlas.phoneme_metrics: ř slyšené jako s / š / ž / z nebo spolknuté, r spolknuté) a CTC
skóre (ceskyhlas.ctc_filter). Pořadí: CER, chyby r / ř, CTC. Nejlepší 3 -> <cache>/<id>/take1..3.wav +
meta.json ("active": 1), <cache>/manifest.json (verze, sha256 checkpointu, prompt, dekódování) a
<cache>/index.html (poslech). Cache patří k jednomu checkpointu + promptu: po změně ji postavte znovu.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

DECODE = {"temp": 0.5, "sampler_decode_steps": 1, "noise_clamp": 1.0}
DEFAULT_CONFIG = "hf://vvolhejn/pocket-tts-czech/czech.yaml@7c1fbd0acba765617749dd17f3dbddc2be791cc7"
KEEP = 3
RZ_SUB = {"s", "ʃ", "ʒ", "z", "ɕ", "ʑ", "∅"}


def read_phrases(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            pid, text = line.split("|", 1)
            out.append((pid.strip(), text.strip()))
    return out


def cmd_synth(a) -> None:
    import soundfile as sf
    import torch
    from pocket_tts import TTSModel

    from ceskyhlas.normalize_cs import normalize

    model = TTSModel.load_model(config=a.config, checkpoint=a.checkpoint, **DECODE)
    state = model.get_state_for_audio_prompt(a.prompt)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    items = []
    for pid, text in read_phrases(Path(a.phrases)):
        norm = normalize(text)
        (work / pid).mkdir(exist_ok=True)
        for seed in range(1, a.n + 1):
            torch.manual_seed(seed)
            sf.write(str(work / pid / f"seed{seed}.wav"), model.generate_audio(state, norm).numpy(), model.sample_rate,
                     subtype="PCM_16")
        items.append({"id": pid, "text": text, "normalized": norm})
        print(pid, flush=True)
    sha = hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()[:12]
    (work / "synth.json").write_text(json.dumps({"checkpoint_sha256": sha, "prompt": Path(a.prompt).name,
                                                 "decode": DECODE, "n": a.n, "phrases": items},
                                                ensure_ascii=False, indent=1), encoding="utf-8")


def rank_key(t: dict) -> tuple:
    return (t["cer"] if t["words"] >= 2 else 0.0, t["rz_errors"], -t["ctc"])


def cmd_rank(a) -> None:
    import jiwer
    import soundfile as sf
    import soxr

    from ceskyhlas.asr_wer import normalize as norm_words
    from ceskyhlas.asr_wer import transcribe
    from ceskyhlas.ctc_filter import Scorer
    from ceskyhlas.phoneme_metrics import Recognizer, align, espeak_ipa, phones

    work, out = Path(a.work), Path(a.out)
    meta = json.loads((work / "synth.json").read_text(encoding="utf-8"))
    takes = [(ph, work / ph["id"] / f"seed{s}.wav", s) for ph in meta["phrases"] for s in range(1, meta["n"] + 1)]
    tmp = Path(tempfile.mkdtemp())
    p16 = []
    for i, (_, w, _) in enumerate(takes):
        x, sr = sf.read(str(w), dtype="float32")
        sf.write(str(tmp / f"{i}.wav"), soxr.resample(x, sr, 16000), 16000)
        p16.append(str(tmp / f"{i}.wav"))
    hyps = transcribe(p16)
    ctc, rec = Scorer(), Recognizer()
    refs = {ph["id"]: phones(espeak_ipa(ph["normalized"])) for ph in meta["phrases"]}
    rows: dict[str, list[dict]] = {}
    for (ph, w, seed), h in zip(takes, hyps):
        pairs = align(refs[ph["id"]], rec(w))
        rz = sum(1 for x, y in pairs if (x == "r̝" and (y or "∅") in RZ_SUB) or (x == "r" and not y))
        ref_n, hyp_n = norm_words(ph["normalized"]), norm_words(h)
        rows.setdefault(ph["id"], []).append({
            "seed": seed, "cer": round(min(jiwer.cer(ref_n, hyp_n), 1.0), 4) if ref_n else 1.0,
            "words": len(ref_n.split()), "rz_errors": rz, "ctc": ctc(w, ph["normalized"])["score"], "asr": h,
            "src": w})
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    index = []
    for ph in meta["phrases"]:
        ranked = sorted(rows[ph["id"]], key=rank_key)[:KEEP]
        d = out / ph["id"]
        d.mkdir()
        for k, t in enumerate(ranked, 1):
            shutil.copyfile(t.pop("src"), d / f"take{k}.wav")
            t["file"] = f"take{k}.wav"
        (d / "meta.json").write_text(json.dumps({"id": ph["id"], "text": ph["text"], "normalized": ph["normalized"],
                                                 "active": 1, "takes": ranked}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        index.append((ph, ranked))
    version = {"version": time.strftime("%Y%m%d-%H%M%S"), "checkpoint_sha256": meta["checkpoint_sha256"],
               "prompt": meta["prompt"], "decode": meta["decode"], "takes_rendered": meta["n"], "kept": KEEP}
    (out / "manifest.json").write_text(json.dumps(version, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "index.html").write_text(html(index, version), encoding="utf-8")
    for ph, ranked in index:
        b = ranked[0]
        print(f"{ph['id']}: take1 seed {b['seed']} CER {b['cer']} r/ř {b['rz_errors']} CTC {b['ctc']}", flush=True)


def html(index, version) -> str:
    rows = []
    for ph, ranked in index:
        cells = "".join(f"<td><audio controls preload='none' src='{ph['id']}/take{k}.wav'></audio><br>"
                        f"<small>take {k}: CER {t['cer']}, r/ř chyby {t['rz_errors']}, CTC {t['ctc']}</small></td>"
                        for k, t in enumerate(ranked, 1))
        rows.append(f"<tr><td><b>{ph['id']}</b><br>{ph['text']}</td>{cells}</tr>")
    return ("<!doctype html><meta charset='utf-8'><title>Cache frází</title>"
            "<style>body{font-family:sans-serif;margin:16px}td{padding:6px;border-bottom:1px solid #ccc;"
            "vertical-align:top}</style>"
            f"<h1>Cache frází</h1><p>Verze {version['version']}, checkpoint {version['checkpoint_sha256']}, prompt "
            f"{version['prompt']}. Aktivní je take 1. Jiný: <code>python -m ceskyhlas.phrase_cache set --out "
            "&lt;cache&gt; &lt;id&gt; 2</code></p><table>" + "".join(rows) + "</table>")


def cmd_set(a) -> None:
    mf = Path(a.out) / a.id / "meta.json"
    if not mf.exists():
        sys.exit(f"fráze {a.id} v {a.out} není")
    meta = json.loads(mf.read_text(encoding="utf-8"))
    if not 1 <= a.take <= len(meta["takes"]):
        sys.exit(f"take musí být 1–{len(meta['takes'])}")
    meta["active"] = a.take
    tmp = mf.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(mf)
    print(f"{a.id}: aktivní take {a.take} (server převezme do 10 s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--phrases", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--prompt", required=True)
    s.add_argument("--config", default=DEFAULT_CONFIG)
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--work", required=True)
    r = sub.add_parser("rank")
    r.add_argument("--work", required=True)
    r.add_argument("--out", required=True)
    t = sub.add_parser("set")
    t.add_argument("--out", required=True)
    t.add_argument("id")
    t.add_argument("take", type=int)
    a = ap.parse_args()
    {"synth": cmd_synth, "rank": cmd_rank, "set": cmd_set}[a.cmd](a)


if __name__ == "__main__":
    main()
