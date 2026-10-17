# 05 Hodnocení

Pravidlo číslo jedna: **každé číslo porovnávejte se skutečným mluvčím, změřeným stejnou cestou.** Rozpoznávače
nejsou dokonalé. Skutečný mluvčí A má po telefonu WER 0,21, ne 0. Cíl „WER 0,05“ je proto nesplnitelný,
rozumný cíl je „blízko skutečnému mluvčímu“.

## Telefonní linka (`ceskyhlas.phone_sim`)

Pokud hlas poběží do telefonu, hodnoťte ho po telefonu: 8 kHz, pásmo 300–3400 Hz, úroveň −20 dBFS, G.711.

- V Evropě **A-law** (PCMA), v USA μ-law (PCMU). Rozdíl ve WER je malý, ale porovnávejte jen stejné s
  stejným.
- Dvě vlastní chyby, které zkreslily naše dřívější čísla:
  1. měkký antialiasingový filtr při převzorkování (v pásmu 3,6–4 kHz o ~15 dB víc energie než soxr);
  2. klipování: hlasitý výstup TTS po pásmové propusti překmitne a G.711 ho ořízne. Úroveň před kodekem
     snižte tak, aby špička zůstala v rozsahu.

## Srozumitelnost: WER / CER (`ceskyhlas.asr_wer`)

- Lokální ASR NVIDIA Parakeet TDT 0.6b v3. Text se porovnává malými písmeny bez interpunkce.
- Sada: 60 vět, jaké bude hlas opravdu říkat (věty z vaší aplikace, čísla po normalizaci).
- **Krátké klipy (~1 s, jedno slovo):** ASR často vrátí nic nebo anglické slovo („Ano.“ → „Uno?“). WER tam
  nic neznamená. Dají se jen seřadit podle výslovnosti a CTC a pak poslechnout.
- CER místo WER, když ASR slévá slova („na shledanou“ → „nasledanou“).

## Shoda textu a zvuku: CTC (`ceskyhlas.ctc_filter`)

Průměrná log-pravděpodobnost písmen na nejlepší CTC cestě (český wav2vec2). Používáme ji dvakrát: jako filtr
trénovacích dat ([01](01-data.md)) a jako třetí kritérium při řazení pokusů syntézy.

## Podobnost hlasu

Kosinová podobnost embeddingů TitaNet s otiskem mluvčího (průměr z ověřených ukázek), i po telefonní lince.

| | po telefonu |
|---|---|
| skutečný mluvčí A, jiné nahrávky | 0,63–0,69 |
| cizí lidé (nejvyšší) | 0,33 |
| Pocket bez doladění | 0,29–0,33 |
| Pocket doladěný | 0,43–0,44 |

Hodnota pod ~0,35 je hlas „někoho jiného“. Podobnost ale není všechno: kNN-VC měl 0,56 a poslechem byl horší.

## Výslovnost hlásek (`ceskyhlas.phoneme_metrics`)

Fonémy textu z espeak-ng proti tomu, co slyší rozpoznávač fonémů (wav2vec2 espeak). Na každou hlásku se počítá
recall (rozpoznáno správně) a podíl smazaných. Dvě metriky se ukázaly jako nejcennější. Odpovídají tomu, co
posluchač u syntézy slyšel:

| metrika | co měří | skutečný mluvčí A | doladěný Pocket |
|---|---|---|---|
| **ř→s** | podíl ř slyšených jako s / z („tři“ → „tsi“) | 0,09 | 0,32 ± 0,03 |
| **r po souhlásce smazané** | „pro“, „dobrý“ → „po“, „dobý“ | 0,055 | 0,19 ± 0,03 |

Pozor na rozpoznávač:

- **Striktní recall ř je nepoužitelný.** Rozpoznávač píše r̝ zřídka i u skutečného mluvčího a ř zapíše jako
  ʃ / ʒ. Používejte „mírný“ recall (jakákoli ř/š/ž frikativa) a ř→s.
- „Dobjej den“ (r → j) se v datech objeví jako *smazané* r, ne jako j.
- Referenci mluvčího měřte na jeho promluvách **bohatých na ř** (u nás 206 promluv se ≥ 3 ř). Na 60 náhodných
  klipech je ř jen pár desítek a čísla skáčou.

## Šum seedů

Pocket TTS je stochastický. Stejný checkpoint a prompt s jiným seedem pohne recallem r a ř o **±10 bodů**
(při ~40 výskytech ř). Proto:

- měřte aspoň **2 seedy** a aspoň **~150 výskytů ř** na variantu (sada 40 vět bohatých na ř + vaše věty);
- rozdíl menší než rozptyl seedů není rozdíl;
- o finále rozhoduje poslech. Čísla slouží k tomu, aby se poslouchalo jen to nejlepší.

## Poslech

Slepé hodnocení (posluchač neví, co je co, pořadí náhodné, stupnice 1–5) zachytí věci, které metriky
neukážou: robotický tón, divné tempo, „řeč pozpátku“. Plánujte ho u každého kandidáta na nasazení.
