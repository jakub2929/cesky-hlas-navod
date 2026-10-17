# Kód návodu (`ceskyhlas`, MIT)

Dvě oddělená prostředí, protože NeMo a Pocket TTS chtějí různé verze knihoven. Testováno na Pythonu 3.12,
Ubuntu 24.04 (WSL 2), torch 2.14.

```bash
# analýza dat a hodnocení (NeMo: Parakeet, Sortformer, TitaNet; transformers: AST, wav2vec2)
python3.12 -m venv ~/venv-analyza && ~/venv-analyza/bin/pip install -r requirements-analyza.txt
# syntéza a server (Pocket TTS, CPU stačí)
python3.12 -m venv ~/venv-pocket && ~/venv-pocket/bin/pip install -r requirements-pocket.txt
# systémové: ffmpeg (dekódování videí), espeak-ng (fonémy pro phoneme_metrics)
sudo apt install ffmpeg espeak-ng
```

Všechno se spouští ze složky `code/` jako `python -m ceskyhlas.<modul>`. Nápovědu ukáže `--help`.

| krok | prostředí | příkaz |
|---|---|---|
| videa → dataset | analyza | `python -m ceskyhlas.yt_ingest prepare --in videa/ --work prace/`, pak `profile --ref ukazky/ --work prace/`, `analyze --work prace/`, `select --work prace/ --out dataset/` |
| kontrola textu | analyza | `python -m ceskyhlas.ctc_filter score dataset/ --out skore.csv` a `select skore.csv --out keep.csv` |
| telefonní linka | analyza | `python -m ceskyhlas.phone_sim wavy/ wavy_telefon/` |
| srozumitelnost | analyza | `python -m ceskyhlas.asr_wer wavy/ texty.txt --phone alaw` |
| výslovnost hlásek | analyza | `python -m ceskyhlas.phoneme_metrics wavy/ texty.txt --out hlasky.json` |
| cache frází | pocket, pak analyza | `python -m ceskyhlas.phrase_cache synth ...`, `rank ...`, `set ...` |
| server | pocket | `POCKET_CHECKPOINT=... POCKET_PROMPT=... python -m ceskyhlas.server` |

Testy (bez modelů, pár sekund): `python -m pytest tests` (stačí `requirements-analyza.txt` nebo jen numpy,
scipy, soundfile, soxr, jiwer, pydantic, pytest).

Doladění samotného Pocket TTS dělají trénovací skripty z repa Pocket TTS (Kyutai, MIT), viz
[docs/04](../docs/04-pocket-tts-recept.md).
