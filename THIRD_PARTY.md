# Software, modely a data třetích stran

Repo obsahuje jen vlastní kód (MIT) a texty. Nic z níže uvedeného v něm není přibalené. Náš kód si modely
stáhne z Hugging Face nebo volá nainstalované programy. Každý z nich má vlastní licenci, kterou musíte
dodržet. Stav k říjnu 2026; ověřte si aktuální kartu modelu.

## Syntéza (to, z čeho vzniká hlas)

| | licence | uvedení autorů |
|---|---|---|
| Pocket TTS, Kyutai (github.com/kyutai-labs/pocket-tts), kód a kodek Mimi | MIT | Kyutai |
| Pocket TTS Czech (huggingface.co/vvolhejn/pocket-tts-czech), váhy | CC BY 4.0 | povinné: autor modelu, odkaz na licenci, popis změn |
| ParCzech4Speech 1.0 (ÚFAL MFF UK, LINDAT), data českého modelu | CC BY 4.0 | ÚFAL, Univerzita Karlova |

## Analýza a hodnocení (nic z toho se nedostane do hlasu)

| | licence |
|---|---|
| NVIDIA NeMo (toolkit) | Apache-2.0 |
| nvidia/parakeet-tdt-0.6b-v3 | CC BY 4.0 |
| nvidia/diar_streaming_sortformer_4spk-v2 | CC BY 4.0 |
| nvidia/speakerverification_en_titanet_large | CC BY 4.0 |
| MIT/ast-finetuned-audioset-10-10-0.4593 | BSD-3-Clause |
| comodoro/wav2vec2-xls-r-300m-cs-250 | Apache-2.0 |
| facebook/wav2vec2-xlsr-53-espeak-cv-ft | Apache-2.0 |
| Hugging Face transformers | Apache-2.0 |
| espeak-ng (volaný jako samostatný program) | GPL-3.0 |
| ffmpeg (volaný jako samostatný program) | LGPL-2.1+ / GPL podle sestavení |

## Co v repu záměrně není

- **Úpravy piperu** (`piper1-gpl`, GPL-3.0), které jsme během pokusů dělali, např. zarovnání MAS se šumem. Kdo
  je chce zopakovat, popis je v [docs/02](docs/02-co-nefungovalo.md). Kód by musel být pod GPL-3.0.
- Hlasy Piperu odvozené z lessac (např. `cs_CZ-jirka-medium`), XTTS-v2, F5-TTS, Fish Speech, WavLM: viz
  [docs/03](docs/03-licence.md).
- Jakákoli data mluvčího: nahrávky, přepisy, modely, prompty.
