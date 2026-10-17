# 01 Data z YouTube

Cíl: ze spontánních videí jednoho mluvčího dostat promluvy 2–15 s, kde mluví **jen on**, bez hudby a zvuků,
s **přesným textem**. Kód: `code/ceskyhlas/yt_ingest.py` a `code/ceskyhlas/ctc_filter.py`.

## Zásady, které se vyplatily

1. **Jen detekovat a vybírat, nic „nečistit“.** Separace hlasu od hudby, odšumování ani rozšíření pásma
   nepomohly. Model se pak učí i artefakty separace. Co je zašuměné, se zahodí.
2. **Stříhat jen v pauzách.** Úsek řeči bez pauzy delší než 15 s se zahodí celý, nikdy se nestříhá
   uprostřed slova. Krajních 30 ms výřezu musí být v tichu (jinak filtr `midword_edge`).
3. **Text kontrolovat jiným modelem, než který ho napsal.** Viz CTC filtr níže.
4. **Validace po celých videích**, ne po náhodných promluvách. Jinak se do validace dostanou sousední věty
   z téhož videa a čísla vypadají lépe, než jsou.
5. **Zdroj dekódovat bezztrátově jednou** (FLAC, původní vzorkování). Jediný ztrátový krok je kodek videa.
6. **VAD pouštět na celé video**, ne na výřezy. Šumové dno a prahy se počítají z percentilů celého souboru.
   Na dvouminutovém výřezu vyšly jinak a vyřadily jiné promluvy.

## Kroky

| krok | nástroj | nastavení, které fungovalo |
|---|---|---|
| dekódování | ffmpeg → FLAC mono | channel mean, původní vzorkování (YouTube obvykle 48 kHz Opus) |
| VAD | energetický, 10ms snímky, hystereze | šumové dno = 10. percentil, řeč začíná nad dno + max(10 dB, 45 % rozsahu), trvá nad dno + max(6 dB, 25 %); úseky < 0,25 s pryč |
| diarizace | NVIDIA Sortformer `diar_streaming_sortformer_4spk-v2` | práh 0,5; offline nastavení z karty modelu (chunk 340, right context 40, fifo 40, cache 188) |
| promluvy | VAD úseky + převládající mluvčí | střih jen v pauze ≥ 0,2 s; pauza ≥ 0,6 s promluvu vždy ukončí; délka 2–15 s; okraj 0,15 s do pauzy |
| kdo je kdo | TitaNet large, otisk z 5–20 ověřených ukázek | v každém videu vyhraje lokální mluvčí s nejvyšším mediánem podobnosti; podobnost i po telefonní lince |
| hudba, zvuky | AST (AudioSet) | hudba ≥ 0,5 nebo jiný zvuk ≥ 0,5 (klikání myši, klávesnice) = pryč |
| přepis | Parakeet TDT 0.6b v3 s konfidencí tokenů | konfidence < 0,80 nebo text < 8 znaků = pryč |
| čísla | `ceskyhlas.normalize_cs` | číslice → slova; promluva, kde po převodu zbude číslice, pryč |
| cizí slova | heuristika + vlastní seznam | víc než 2 cizí slova = pryč (čeština je čte z písmen jinak, než je mluvčí řekl) |
| export | 24 kHz (Pocket) / 22,05 kHz (Piper), PCM16 | horní propust 70 Hz, aktivní snímky −20 dBFS RMS, špička ≤ −1 dBFS |

**Práh podobnosti** nenastavujte od oka. Změřte podobnost svého otisku s nahrávkami několika *jiných*
lidí (stejnou cestou, i po telefonní lince). Práh dejte kousek nad nejvyšší hodnotu. U nás měli cizí
mluvčí nejvýš 0,33, takže práh byl 0,40. Skutečný mluvčí A měl průměr 0,63 po lince a 0,75 bez ní.

## Nezávislý CTC filtr textu

Parakeet přepisuje dobře, ale na spontánní řeči se splete. Typicky polkne slovo, opraví přeřeknutí nebo
„učeší“ hovorový tvar. Když text kontroluje tentýž model, chyby si potvrdí. Proto skóre počítá **jiný**
model: český wav2vec2 (`comodoro/wav2vec2-xls-r-300m-cs-250`). Ten zarovná text na zvuk (CTC Viterbi) a pro
každé písmeno spočítá průměrnou log-pravděpodobnost.

Jako měřítko potřebujete sadu s dokonalou shodou textu. My jsme použili jiný (syntetický) český hlas, který
čte tytéž texty. Čísla z našeho datasetu:

| sada | skóre p10 / p50 / p90 (blíž nule = lépe) |
|---|---|
| syntetický hlas, tytéž texty | −0,187 / −0,074 / −0,028 |
| mluvčí A, spontánní řeč | −0,598 / −0,330 / −0,154 |

Z ~2 000 promluv mluvčího A dosáhlo 10. percentilu reference jen 221 (32 min). Pravidlo výběru proto zní:
brát od nejlepších až do stropu (180 min) nebo do p10 reference. Když toho je méně než podlaha (120 min),
vezme se prostě nejlepších 120 min (`ctc_filter select`).

**Poučení:** spontánní řeč má proti čtené řeči výrazně horší shodu textu a zvuku. Nejde o chybu přepisu,
ale o samotnou spontánní řeč: polykání hlásek, zkracování, rychlé tempo. To se ukázalo jako hlavní důvod,
proč Piper selhal (viz [02](02-co-nefungovalo.md)).

## Kolik dat

| | |
|---|---|
| zdroj | ~70 videí mluvčího A, 12,1 h |
| po segmentaci a filtrech | ~1 900 promluv, 5,7 h |
| data pro Pocket TTS | ~1 170 promluv, 3,5 h: jen jeden rok nahrávek, bez promluv s riskantními čísly, bez nejhorších 10 % podle CTC; validace = celá videa, ~6 % minut |

Náš „trychtýř“ v minutách (`yt_ingest select` zapíše vlastní `funnel.csv`):

| krok | zbývá min | odpadlo |
|---|---|---|
| délka videí | 726 | |
| řeč podle VAD | 644 | 82 (pauzy, ticho) |
| délka 2–15 s, střih jen v pauzách | 635 | 9 |
| diarizace + překryv | 630 | 5 |
| střih uprostřed hlásky | 598 | 32 |
| hudba + zvuky | 584 | 14 |
| podobnost + klipování | 584 | < 1 |
| **konfidence ASR** | 435 | **149** |
| **cizí slova a zbylé číslice** | 341 | **94** |

U spontánního mluvčího tedy nejvíc odpadne na přepisu: hovorová řeč, žargon a anglická slova. Hudba ani
diarizace u videí s jedním mluvčím skoro nic nevezmou.
