# 02 Co nefungovalo a proč

Shrnutí: **Piper (VITS) se na spontánní češtině jednoho mluvčího nenaučil zarovnat text na zvuk.** Výsledek
plynule „mluví“, ale špatnými hláskami – posluchač to popsal jako „řeč pozpátku“. Zkusili jsme osm variant
dat a tréninku i destilaci; žádná nebyla srozumitelná. Funguje až Pocket TTS ([04](04-pocket-tts-recept.md)),
který monotónní zarovnání vůbec nepotřebuje.

## Jak poznat, že zarovnání selhává (MAS)

VITS se během tréninku učí zarovnání fonémů na zvuk (Monotonic Alignment Search, MAS). Když to nejde, MAS
„rozdělí“ zvuk tak, že většina fonémů dostane **jeden jediný snímek** a zbytek času připadne pár fonémům.
Měřítko: podíl fonémů s délkou jednoho snímku (`MAS 1f`) na pevné sadě promluv.

| model / data | MAS 1f (nižší = lepší) |
|---|---|
| zdravý model na vlastních datech (výchozí český hlas) | 0,34 |
| kontrola: anglický základ + 90 min LJSpeech | 0,49 → 0,38 |
| kontrola: syntetická čeština (jiný hlas čte texty mluvčího A) | 0,49 → 0,27 |
| **mluvčí A, spontánní řeč – všechny varianty** | **0,55–0,60, neklesá** |
| mluvčí A zpomalený na 0,70× | 0,41–0,43 (poprvé pod 0,45 – a stejně nesrozumitelné) |

Pokud vám MAS 1f po prvních tisících kroků neklesá pod ~0,45, další epochy nepomohou.

## Kontroly: chyba není v kódu

Než jsme vinili data, ověřili jsme celý řetězec:

- **Kódová cesta:** výchozí hlas přes stejný export a stejné CLI dá WER 0,14 (jako jeho autoři). Mapa fonémů,
  warm start (784/784 tenzorů), ID fonémů v tréninku a při inferenci, párování text–zvuk v cache (WER
  přepisu tréninkového audia 0,026), oříznutí VAD a monotonic_align – všechno sedí.
- **Kontrolní fine-tune na čistých datech:** 150 vět výchozího hlasu → WER 0,08 po první epoše.
- **Anglický základ (Piper `_base_model`, LibriTTS-R) + 90 min LJSpeech:** WER po telefonu 0,015.
- **Syntetická čeština:** výchozí český hlas čte texty mluvčího A (dokonalá shoda textu, čtené tempo):
  WER 0,154. Čeština přes espeak-ng na anglickém základu tedy funguje.
- **Výměna modulů:** výchozí hlas + jeden modul z modelu doladěného na mluvčím A. Textový enkodér z mluvčího A
  zvedne WER na 0,67, ostatní moduly skoro nic. Data mluvčího A **rozbijí textový enkodér**.

## Co jsme zkoušeli na datech mluvčího A

WER = Parakeet po simulované lince A-law, 60 vět (skutečný mluvčí A: 0,21).

| varianta | nápad | výsledek |
|---|---|---|
| telefonní nahrávky, plný fine-tune | nejvíc dat | „řeč pozpátku“, WER 1,1–1,6 |
| telefonní nahrávky, zmrazený textový enkodér a délky (± flow) | učit jen „barvu“ hlasu | srozumitelné (WER 0,1–0,2), ale robotické, podobnost jen 0,16–0,24 a s každou epochou horší |
| interpolace vah výchozí ↔ doladěný | kompromis | podobnost roste jen za cenu srozumitelnosti |
| čistší podmnožiny (bitrate, kodek), filtr MAS po promluvách, rozšíření pásma (LavaSR) | data jsou moc špatná | zarovnání stejné |
| YouTube, plný fine-tune z anglického základu (86 min) | lepší zvuk než telefon | WER 0,95, od ~20. epochy robotické |
| YouTube + CTC filtr (120 min) | přesnější text | WER 1,02–1,09 |
| YouTube, jen nejčistší promluvy (32 min) | kvalita místo kvantity | WER 1,03–1,31 |
| čárky v pauzách ≥ 150 ms, oříznuté okraje | pauzy jako interpunkce | WER 0,89, MAS beze změny |
| mluvčí A + 12 mluvčích ParCzech v jednom běhu | víc češtiny pro textový enkodér | mluvčí A 0,99–1,22 – **a ParCzech mluvčí taky 0,97–1,05** |
| MAS se šumem (jako ve VITS2) | jiné hledání zarovnání | WER 0,92–1,04, MAS 0,56–0,58 |
| zpomalení na 0,70× (rubberband, formanty zachované) | víc snímků na foném (6,4 → 9,15) | MAS 0,41, WER 1,07–1,10, poslechem nepoužitelné |
| syntéza s délkami přímo z MAS (bez prediktoru délek) | je chyba v prediktoru? | WER 0,87 – selhává už zarovnání, ne prediktor |

**Klíčový pokus je ParCzech:** ani skutečná (parlamentní) čeština cizích mluvčích se na anglickém základu
s fonémy espeak-ng nezarovnala, zatímco syntetická čeština ano. Problém tedy není mluvčí A. Je to přirozená
česká řeč (tempo, redukce, koartikulace) v kombinaci s fonémy z espeak-ng a anglickým základem. Zkoušeli
jsme i, jestli se fonémy z espeak-ng neodchylují od skutečné výslovnosti víc než u čtené řeči
(rozpoznávač fonémů proti espeak-ng). Chybovost fonémů byla proti syntetické řeči 1,9× vyšší u mluvčího A,
1,65× u výstupu Pocket TTS a 2,6× u ParCzech. Hranici 2×, kterou jsme si předem dali, překročil jen
ParCzech, takže to selhání samo nevysvětluje.

## Destilace: Piper učený z výstupů Pocket TTS

Nápad: doladěný Pocket TTS ([04](04-pocket-tts-recept.md)) přečte tisíce vět hlasem mluvčího A a Piper se
učí z této „čtené“ řeči. Ta se zarovnává jako syntetická čeština.

- Korpus: 5 280 vět, 6,1 h, vyfiltrovaných podle CTC a podobnosti.
- Výsledek: nejlepší WER 0,74 (34 tis. kroků), podobnost 0,47. Učitel (Pocket) má na stejných větách 0,28.
- MAS 1f zůstal 0,52–0,55: Pocket napodobí i spontánní tempo mluvčího a s ním i problém se zarovnáním.
- Varianta s převodem hlasu kNN-VC (vyšší podobnost) byla poslechem horší a filtr z ní vyhodil 58 %.

Destilace má smysl, jen když potřebujete extrémně rychlý model (Piper ~0,05 RTF). Srozumitelnosti učitele
nedosáhne.

## Převod hlasu (kNN-VC) místo vlastního modelu

Výstup Pocket TTS převedený přes kNN-VC (WavLM + HiFi-GAN) na mluvčího A: podobnost stoupla z 0,32–0,42 na
0,54–0,56 (skutečný mluvčí 0,69), ale WER se zhoršilo o 0,07–0,17 a poslechem to bylo horší. Navíc licence:
WavLM Large je předtrénovaný i na GigaSpeech, jehož audio je jen pro nekomerční výzkum ([03](03-licence.md)).

## Co si z toho vzít

1. Piper / VITS je skvělý na **čtená** data (dobrá výslovnost, jasné pauzy). Na spontánní řeč z videí ho
   nepoužívejte, nebo nejdřív změřte MAS 1f na malém vzorku.
2. Než začnete ladit hyperparametry, udělejte kontroly (čistá data, syntetická data, výměna modulů). Ušetří
   to týdny hledání chyby v kódu, která tam není.
3. Měřte po telefonní lince a vždy proti skutečnému mluvčímu změřenému stejnou cestou ([05](05-hodnoceni.md)).
