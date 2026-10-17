"""phone_sim, CTC Viterbi, fonémové metriky a segmentace ingestu na syntetických signálech (bez modelů)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("scipy")
pytest.importorskip("soxr")
from ceskyhlas.ctc_filter import ctc_viterbi, norm_text, score_one, select  # noqa: E402
from ceskyhlas.phone_sim import PHONE_SR, phone_line  # noqa: E402
from ceskyhlas.phoneme_metrics import align, metrics, phones  # noqa: E402
from ceskyhlas.yt_ingest import build_utterances, is_foreign, level, split_speech  # noqa: E402


def tone(f, s, sr=24000, a=0.5):
    t = np.arange(int(s * sr)) / sr
    return (a * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_phone_line_level_band_and_no_clipping():
    y = phone_line(tone(1000, 1.0, a=0.99), 24000)
    assert abs(len(y) - PHONE_SR) < 5
    rms_db = 20 * math.log10(float(np.sqrt(np.mean(y[800:] ** 2))))
    assert -21.5 < rms_db < -18.5
    assert np.max(np.abs(y)) < 0.999
    low = phone_line(np.concatenate([tone(1000, 0.5), tone(100, 0.5)]), 24000)
    assert np.sqrt(np.mean(low[-2000:] ** 2)) < 0.2 * np.sqrt(np.mean(low[1000:3000] ** 2))   # 100 Hz potlačeno


def test_ulaw_and_bad_codec():
    assert len(phone_line(tone(500, 0.2), 24000, "ulaw")) > 0
    with pytest.raises(ValueError):
        phone_line(tone(500, 0.2), 24000, "gsm")


def test_ctc_viterbi_and_score():
    vocab = {"<pad>": 0, "|": 1, "a": 2, "b": 3}
    tokens, dropped = norm_text("Ab ba x", vocab)
    assert tokens == [2, 3, 1, 3, 2] and dropped == 1
    T = 12
    logp = np.full((T, 4), math.log(0.01))
    seq = [2, 2, 0, 3, 0, 1, 0, 3, 3, 0, 2, 0]
    for t, k in enumerate(seq):
        logp[t, k] = math.log(0.97)
    path = ctc_viterbi(logp, tokens, 0)
    assert [p for p in path if p >= 0][0] == 0 and max(path) == len(tokens) - 1
    s = score_one(logp, tokens, 0, 1)
    assert -0.1 < s["score"] <= 0 and s["low_share"] == 0.0
    assert ctc_viterbi(logp[:3], tokens, 0) is None
    assert score_one(logp[:3], tokens, 0, 1)["score"] == -math.inf


def test_ctc_select_rule():
    rows = [{"set": "data", "id": str(i), "score": -0.1 * i, "minutes": 1.0} for i in range(10)]
    rows += [{"set": "ref", "id": f"r{i}", "score": -0.05 * i, "minutes": 1.0} for i in range(10)]
    keep, msg = select(rows, max_min=100, min_min=2)
    assert [r["id"] for r in keep] == ["0", "1", "2", "3", "4"] and "p10" in msg
    keep, msg = select(rows, max_min=100, min_min=8)
    assert len(keep) == 8 and "podlaha" in msg


def test_phones_normalisation():
    assert phones("t͡ʃˈiːslo") == ["t", "ʃ", "iː", "s", "l", "o"]
    assert phones("tr̝̊ˈi") == ["t", "r̝", "i"]


def test_align_and_metrics():
    ref = phones("tr̝ˈi dobriː")
    hyp = phones("tsi dobjiː")            # ř -> s, r po souhlásce -> j
    pairs = align(ref, hyp)
    assert ("r̝", "s") in pairs and ("r", "j") in pairs
    m = metrics([(ref, hyp), (phones("r̝eka"), phones("ʒeka"))])
    assert m["ř_to_s"] == 0.5 and m["recall_ř_fric"] == 0.5
    assert m["r_after_cons"]["n"] == 1 and m["r_after_cons"]["to_j"] == 1.0
    assert m["recall_r"] == 0.0 and m["n_ř"] == 2


def test_vad_and_utterances():
    sr = 16000
    silence = np.random.default_rng(0).normal(0, 1e-4, int(0.5 * sr)).astype(np.float32)
    burst = tone(300, 1.0, sr, 0.3)
    x = np.concatenate([silence, burst, silence[:int(0.3 * sr)], burst, silence, silence, burst, silence])
    vad = split_speech(x, sr)
    assert len(vad.segments) == 3
    act = np.zeros((int(len(x) / sr / 0.08) + 1, 2), bool)
    act[:, 0] = True
    utts, dropped = build_utterances(vad.segments, act, 0.08)
    assert len(utts) == 2 and utts[0]["n_chunks"] == 2 and not dropped   # 0,3 s pauza spojí, 1 s ukončí
    act2 = act.copy()
    s2 = int(vad.segments[1][0] / 0.08)
    act2[s2:, 0], act2[s2:, 1] = False, True
    utts2, _ = build_utterances(vad.segments, act2, 0.08)
    assert len(utts2) == 3                                                 # jiný mluvčí = nová promluva


def test_level_and_foreign():
    y = level(np.concatenate([np.zeros(8000, np.float32), tone(300, 1.0, 16000, 0.05)]), 16000)
    act = y[8000:]
    assert abs(20 * math.log10(float(np.sqrt(np.mean(act ** 2)))) + 20) < 1.0
    assert is_foreign("weekly") and is_foreign("thinking") and is_foreign("Facebook")
    assert not is_foreign("tři") and not is_foreign("maximálně") and not is_foreign("text")
