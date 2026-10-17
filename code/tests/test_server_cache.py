"""ceskyhlas.server: cache frází (shoda normalizovaného textu -> aktivní take, znovunačtení po změně) a stav."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("pydantic")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ceskyhlas.phrase_cache import cmd_set, rank_key  # noqa: E402
from ceskyhlas.server import PhraseCache, pcm_chunks, phrase_key, phrase_status  # noqa: E402


def _make(root: Path) -> None:
    (root / "rozlouceni").mkdir(parents=True)
    for k, f in ((1, 220.0), (2, 440.0)):
        t = np.arange(2400) / 24000
        sf.write(str(root / "rozlouceni" / f"take{k}.wav"), (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32), 24000)
    (root / "rozlouceni" / "meta.json").write_text(json.dumps(
        {"id": "rozlouceni", "text": "Děkuji, na shledanou.", "normalized": "Děkuji, na shledanou.", "active": 1,
         "takes": [{"file": "take1.wav"}, {"file": "take2.wav"}]}, ensure_ascii=False), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"version": "test-1", "checkpoint_sha256": "abc",
                                                    "prompt": "prompt.wav"}), encoding="utf-8")


def test_phrase_key():
    assert phrase_key("Děkuji, na shledanou.") == phrase_key("děkuji,  na shledanou") == "děkuji, na shledanou"
    assert phrase_key("Moment.") != phrase_key("Moment, ověřím to.")


def test_cache_hit_miss_and_reload(tmp_path):
    _make(tmp_path)
    c = PhraseCache(str(tmp_path))
    assert c.info()["version"] == "test-1" and c.info()["n"] == 1
    hit = c.get("děkuji, na shledanou")
    assert hit is not None and hit[0] == "rozlouceni" and hit[1] == 1
    assert c.get("Děkuji, na shledanou, pane.") is None and c.get("Děkuji.") is None

    class A:
        out, id, take = str(tmp_path), "rozlouceni", 2

    cmd_set(A)
    m = tmp_path / "rozlouceni" / "meta.json"
    os.utime(m, (time.time() + 5, time.time() + 5))
    c.checked = 0                                   # přeskočit 10s interval kontroly
    assert c.get("Děkuji, na shledanou.")[1] == 2


def test_no_cache_dir(tmp_path):
    for root in (None, str(tmp_path / "chybi")):
        c = PhraseCache(root)
        assert c.get("Ano.") is None and c.info()["n"] == 0


def test_pcm_8k():
    b = b"".join(pcm_chunks(np.zeros(24000, np.float32), 24000, 8000))
    assert abs(len(b) / 2 - 8000) < 50


def test_phrase_status_mismatch():
    ok = phrase_status({"n": 2, "checkpoint_sha256": "abc", "prompt": "p.wav"}, "abc", "p.wav")
    assert ok["matches_model"] and "warning" not in ok
    for sha, prompt in (("def", "p.wav"), ("abc", "jiny.wav")):
        bad = phrase_status({"n": 2, "checkpoint_sha256": "abc", "prompt": "p.wav"}, sha, prompt)
        assert not bad["matches_model"] and "znovu" in bad["warning"]
    assert "matches_model" not in phrase_status({"n": 0, "checkpoint_sha256": None}, "abc", "p.wav")


def test_rank_key_skips_cer_for_one_word():
    one = [{"cer": 1.0, "words": 1, "rz_errors": 0, "ctc": -0.1}, {"cer": 0.0, "words": 1, "rz_errors": 1, "ctc": -0.05}]
    assert sorted(one, key=rank_key)[0]["rz_errors"] == 0          # u jednoho slova CER nerozhoduje
    two = [{"cer": 0.3, "words": 2, "rz_errors": 0, "ctc": -0.1}, {"cer": 0.0, "words": 2, "rz_errors": 1, "ctc": -0.2}]
    assert sorted(two, key=rank_key)[0]["cer"] == 0.0
