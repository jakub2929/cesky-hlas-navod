# 06 Nasazení

## Streamovací server (`ceskyhlas.server`)

```bash
cd code
POCKET_CHECKPOINT=/cesta/checkpoint.pt POCKET_PROMPT=/cesta/prompt.wav python -m ceskyhlas.server
curl -s http://127.0.0.1:8765/health
curl -s -H 'Content-Type: application/json' -d '{"text":"Dobrý den, cena je 499 Kč.","rate":8000}' \
     http://127.0.0.1:8765/tts -o vystup.pcm          # PCM s16le mono 8 kHz
```

- Text → normalizace → věty → Pocket TTS generuje větu po větě a server posílá zvuk, jak vzniká (chunked).
  Do telefonu stačí 8 kHz; převzorkování z 24 kHz dělá soxr průběžně. Kodek (G.711) je věc telefonní ústředny.
- `/health` ukazuje sha256 načteného checkpointu, prompt, dekódování a stav cache frází. Hodí se, když se
  model mění – na hostiteli hned vidíte, co skutečně běží.
- **Síť:** výchozí adresa je 127.0.0.1 (jen ten počítač, bez hesla). Jiná adresa (`TTS_BIND=0.0.0.0`) vyžaduje
  `TTS_TOKEN` (≥ 16 znaků). Bez něj server nenastartuje a požadavky bez tokenu dostanou 401. Port nikdy
  nepřesměrovávejte do internetu.

## Latence na CPU

Měřeno na stolním 6jádrovém CPU (AMD Ryzen 5 5600), 4 vlákna na jeden hovor:

| | první zvuk (medián) | RTF (medián) |
|---|---|---|
| 1 hovor | 136 ms | 0,41 |
| 2 hovory ve 2 procesech (stroj zároveň vytížen jinou prací) | ~220 ms | 0,56 |
| 2 hovory ve 1 procesu (2 workery) | – | 0,86 |

RTF 0,5 = generuje dvakrát rychleji, než se mluví. Paměť ~1 GB na proces.

- Na **dva souběžné hovory spusťte dva procesy** (dva porty), ne dva workery v jednom. Generátor Pocketu není
  thread-safe a v jednom procesu si workery berou vlákna.
- Na VPS chtějte vyhrazená jádra („dedicated / CPU-optimized“), ne sdílené burstable vCPU: 4 vCPU na hovor,
  4 GB RAM. Před dlouhodobým pronájmem si to změřte na hodinové instanci.

## Český normalizátor (`ceskyhlas.normalize_cs`)

Do TTS nesmí dorazit číslice – model je přečte nesmyslně, nebo vůbec. Normalizátor převádí:

- ceny („499 Kč“ → „čtyři sta devadesát devět korun“, haléře, desetinná čísla „celá“), procenta, GB / MB /
  Mbit/s;
- data („15. 3. 2025“), časy („14:30“), telefonní čísla (po skupinách, úvodní nula po číslicích);
- řadové číslovky před malým písmenem („2. patro“ → „druhé patro“), rozsahy („2–4“ → „dva až čtyři“);
- zkratky („tzn.“, „atd.“, „DPH“), značky s číslicí („O2“, „5G“);
- **pád a rod**: podstatné jméno po čísle se skloňuje podle počtu (1 / 2–4 / 5+). Jedna a dvě se shodují
  v rodě se známým následujícím slovem („dvě koruny“, „jedna minuta“).

Co zbyde, přečte po číslicích, takže výstup nikdy číslici neobsahuje. Jen standardní knihovna, 140+ testů.
Nejistý zůstává pád po předložkách („s 5 lidmi“): takové věty raději pište slovy už ve zdroji.

## Cache pevných frází (`ceskyhlas.phrase_cache`)

Věty, které hlas říká pořád (pozdrav, představení, rozloučení, „Moment.“, „Promiňte, můžete to zopakovat?“),
by neměly záviset na štěstí stochastického modelu. Jednou za čas „Dobrý den“ zazní jako „Dobjej den“.

1. `synth`: každou frázi vyrenderuje 8× (seedy 1–8) se stejným promptem a dekódováním jako server.
2. `rank`: seřadí podle CER přepisu (jednoslovné fráze bez CER), chyb r / ř a CTC; nechá nejlepší 3 a
   vytvoří `index.html` na poslech.
3. Člověk poslechne a případně vybere jinou variantu: `phrase_cache set --out <cache> <id> 2`. Server změnu
   převezme do 10 s bez restartu.
4. Server pošle frázi z cache, když se normalizovaný text shoduje (bez ohledu na velikost písmen, mezery a
   koncovou interpunkci). Hlavička `X-Phrase: <id>#<take>`.

Cache patří k **jednomu checkpointu a jednomu promptu**. Po jejich změně ji postavte znovu a znovu vyberte
varianty. `/health` při nesouladu ukáže `phrases.warning`.

## Povinnosti

- Na začátku každého hovoru musí zaznít, že volá automat / AI se syntetickým hlasem.
- Uvedení autorů modelu (CC BY 4.0) patří na stránku o produktu a do informace o nahrávání hovoru
  ([03](03-licence.md)).
- Na hostitele jde jen model, prompt a kód – trénovací data mluvčího nikdy.
