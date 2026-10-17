# Český hlas pro TTS z nahrávek z YouTube – co nefungovalo a co funguje

Praktický návod v češtině: jak z veřejných videí jednoho mluvčího (s jeho výslovným souhlasem) udělat
syntetický hlas, který zvládne česky mluvit do telefonu. Popisujeme celou cestu i slepé uličky, včetně čísel,
a přikládáme kód, který jsme k tomu napsali.

**Pro koho:** vývojáře hlasových asistentů a nadšence, kteří chtějí český hlas lokálně (bez cloudu) a nechtějí
opakovat týdny pokusů. Předpokládá se základní Python, Linux nebo WSL a jedna GPU s 8 GB (na trénink). Provoz
pak běží i na CPU.

## Výsledek v jedné tabulce

Všechno měřeno na jednom mluvčím (dále **mluvčí A**), ~6 hodin jeho spontánní řeči z videí, hodnoceno po
simulované telefonní lince (G.711 A-law). Reference je skutečný mluvčí A, změřený stejnou cestou. Řádek
označený * je z dřívějšího pokusu na jeho telefonních nahrávkách (WER bez telefonní linky).

| přístup | srozumitelnost (WER, nižší = lepší) | podobnost hlasu (0–1) | verdikt |
|---|---|---|---|
| skutečný mluvčí A (reference) | 0,21 | 0,69 | – |
| Piper / VITS doladěný na jeho spontánní řeči (8 variant) | 0,9–1,1 | 0,4–0,6 | **nesrozumitelné** |
| Piper doladěný z jiného českého hlasu (cs Jirka), zmrazené moduly * | 0,1–0,2 | 0,16–0,24 | srozumitelné, ale robotické, jiný hlas, licenčně sporné |
| Pocket TTS čeština bez doladění (jen hlasový prompt) | 0,12–0,20 | 0,29–0,33 | hlas na úrovni „cizího člověka“ |
| Piper učený z výstupů Pocket TTS (destilace, 6 h) | 0,74 | 0,47 | učitele nedohoní |
| **Pocket TTS čeština doladěná na mluvčím A (~3,5 h, 2 500 kroků)** | **0,27** | **0,44** | **funguje, používáme** |

Zbývající slabina doladěného Pocketu: **ř** vysloví jako **s** asi 3,5× častěji než skutečný mluvčí (32 % proti
9 %) a po souhlásce občas spolkne **r**. Částečně to řeší výběr hlasového promptu
([04](docs/04-pocket-tts-recept.md)) a cache pevných frází ([06](docs/06-nasazeni.md)).

## Souhlas a etika – čtěte jako první

- **Klonujte jen hlas člověka, který s tím výslovně souhlasil**, nejlépe písemně, s popsaným účelem (k čemu,
  kde, jak dlouho). Veřejná videa nejsou souhlas.
- Hlas je osobní údaj, a pokud slouží k rozpoznání člověka, i biometrický. Nahrávky, přepisy a modely
  chraňte: neveřejné úložiště, šifrované zálohy, smazání na žádost mluvčího.
- **Žádné vydávání se za někoho.** Syntetický hlas nesmí předstírat, že mluví skutečný člověk: žádné podvodné
  hovory, falešné citace ani obcházení hlasového ověření.
- **V hovoru vždy hned na začátku řekněte, že volá automat / AI se syntetickým hlasem.**
- Mluvčí z veřejných korpusů (např. poslanci v ParCzech) jsou identifikovatelní lidé. Jejich data se smí
  použít k učení výslovnosti, ale nenabízejte je jako hlasy.
- Licence modelů a dat často zakazují komerční použití – viz [03 Licence](docs/03-licence.md).

## Obsah

| | |
|---|---|
| [01 Data z YouTube](docs/01-data.md) | VAD, diarizace, filtr hudby a hluku, ASR, čísla na slova, nezávislý CTC filtr textu |
| [02 Co nefungovalo a proč](docs/02-co-nefungovalo.md) | Piper / VITS na spontánní češtině: zarovnání (MAS) se nikdy nechytí; kontroly a pokusy |
| [03 Licence](docs/03-licence.md) | pasti: Piper cs ← lessac, WavLM ← GigaSpeech, XTTS / F5 / Fish nekomerční; co je čisté |
| [04 Recept na Pocket TTS](docs/04-pocket-tts-recept.md) | formát dat, nastavení, přeučení po ~2–2,5 tis. krocích, prompt, dekódování |
| [05 Hodnocení](docs/05-hodnoceni.md) | telefonní linka, WER, CTC, podobnost, výslovnost hlásek, ř→s, šum seedů |
| [06 Nasazení](docs/06-nasazeni.md) | streamovací server, český normalizátor, cache frází, latence na CPU |
| [07 Praktické](docs/07-prakticke.md) | past s diskem WSL (vhdx), jedna úloha na GPU, zálohy |

## Kód (`code/`, MIT)

| modul | co dělá |
|---|---|
| `ceskyhlas/normalize_cs.py` | čísla, ceny, data, časy, telefonní čísla, procenta, zkratky → slova (jen standardní knihovna, 140+ testů) |
| `ceskyhlas/yt_ingest.py` | videa → dataset LJSpeech: VAD, diarizace, podobnost s mluvčím, hudba / zvuky, ASR, filtry |
| `ceskyhlas/ctc_filter.py` | nezávislá kontrola textu proti zvuku (CTC forced alignment českým wav2vec2) a výběr |
| `ceskyhlas/phone_sim.py` | simulace telefonní linky G.711 (A-law / μ-law) |
| `ceskyhlas/asr_wer.py` | přepis lokálním ASR (Parakeet) a WER / CER |
| `ceskyhlas/phoneme_metrics.py` | výslovnost r, ř, š, ž, z, č, ť, ď, ň; metriky ř→s a r po souhlásce |
| `ceskyhlas/server.py`, `auth.py` | streamovací TTS server (FastAPI, Pocket TTS, PCM 8/16/24 kHz) s tokenem pro síť |
| `ceskyhlas/phrase_cache.py` | předem vyrenderované pevné fráze: 8 pokusů, automatické seřazení, ruční výběr |

Instalace a spuštění: [`code/README.md`](code/README.md). Testy: `cd code && python -m pytest tests`.

Kód odvozený z piperu (GPL-3.0, např. naše úprava zarovnání MAS se šumem) **do repa nedáváme**. Celé repo je
proto pod MIT. Fonémy bereme z programu `espeak-ng` voláním z příkazové řádky (GPL-3.0, samostatný program).

**V repu nejsou žádné nahrávky, přepisy, modely ani prompty mluvčího.** Jen postup, čísla a kód.

## Licence a souhlas

Kód a texty: [MIT](LICENSE). Modely a data třetích stran mají vlastní licence:
[THIRD_PARTY.md](THIRD_PARTY.md) a [docs/03-licence.md](docs/03-licence.md). Nejde o právní radu.

Hlas mluvčího A vznikl s jeho výslovným souhlasem a není tu zveřejněn. Kdo podle návodu klonuje hlas, odpovídá
za souhlas mluvčího, za ochranu jeho dat a za to, že posluchač ví, že mluví stroj.
