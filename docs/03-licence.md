# 03 Licence – kde jsou pasti

Stav k říjnu 2026, ověřeno na kartách modelů a licenčních textech. **Nejde o právní radu.** Než hlas použijete
komerčně, nechte si to potvrdit právníkem.

Hlas má tři vrstvy a každá má vlastní licenci: **kód** (trénink, inference), **váhy** (základní model) a
**data** (na čem se základ učil a na čem ladíte vy). Komerční použití potřebuje, aby to dovolily všechny tři.

## Pasti, na které jsme narazili

| co | proč je to problém |
|---|---|
| **Piper `cs_CZ-jirka-medium`** (a většina hlasů Piperu: sk, pl, ru, de thorsten, …) | karta modelu říká „Finetuned from U.S. English lessac voice“. Data lessac (Blizzard 2013) jsou licencovaná **výhradně pro výzkum** a výslovně vylučují „development … commercialisation … of voice synthesis“. Jestli se to přenáší na doladěné modely, licence neřeší – riziko. Každý hlas doladěný z Jirky to dědí. |
| **Piper (kód `piper1-gpl`)** | GPL-3.0. Pro vlastní použití OK; kdo distribuuje software s upraveným piperem, musí dodržet GPL. Proto naše úpravy piperu v tomto repu nejsou. |
| **WavLM Large** (kNN-VC, převod hlasu) | předtrénovaný mimo jiné na GigaSpeech, jehož audio je jen pro nekomerční výzkum a vzdělávání. Kód kNN-VC je MIT, ale váhy WavLM nesou riziko dat. |
| **XTTS-v2** (Coqui) | Coqui Public Model License – **nekomerční**. |
| **F5-TTS** | váhy CC BY-NC 4.0 – **nekomerční**. |
| **Fish Speech 1.5** | váhy CC BY-NC-SA 4.0 – **nekomerční** (u novějších verzí si licenci ověřte zvlášť). |
| modely bez souboru LICENSE | „tag apache-2.0“ na Hugging Face bez licenčního souboru nestačí; některé README samy píšou, ať se zatím berou jako nelicencované. |

## Co je čisté (s uvedením autorů)

| co | licence | poznámka |
|---|---|---|
| **Pocket TTS** (Kyutai), kód | MIT | trénink i inference |
| **Pocket TTS Czech** (`vvolhejn/pocket-tts-czech`), váhy | CC BY 4.0 | učeno na ParCzech4Speech; uveďte autory a že jste model doladili |
| **ParCzech4Speech 1.0** (ÚFAL MFF UK, LINDAT) | CC BY 4.0 (audio AudioPSP CC0, přepisy ParCzech 4.0 CC0) | mluvčí jsou poslanci – identifikovatelní lidé, nenabízejte je jako hlasy |
| Piper `_base_model` (LibriTTS-R) | CC BY 4.0 | čistý anglický základ pro Piper, trénovaný od nuly |
| Piper LJSpeech medium | public domain | čistý jednohlasý základ |

Uvedení autorů, které posíláme s každou kopií hlasu (upravte podle sebe):

> Syntéza řeči: Pocket TTS Czech (vvolhejn/pocket-tts-czech, CC BY 4.0), založeno na Kyutai Pocket TTS a kodeku
> Mimi (Kyutai, MIT); trénováno na ParCzech4Speech 1.0 (ÚFAL, Univerzita Karlova; CC BY 4.0). Změny: doladěno
> na nahrávkách jednoho mluvčího s jeho souhlasem.

## Modely, které používá náš kód (jen k analýze, nic z nich nejde do hlasu)

| model | k čemu | licence |
|---|---|---|
| NVIDIA Parakeet TDT 0.6b v3 | přepis, WER | CC BY 4.0 |
| NVIDIA Sortformer `diar_streaming_sortformer_4spk-v2` | diarizace | CC BY 4.0 |
| NVIDIA TitaNet large | podobnost hlasu | CC BY 4.0 |
| MIT AST (`ast-finetuned-audioset-10-10-0.4593`) | hudba a zvuky | BSD-3-Clause |
| `comodoro/wav2vec2-xls-r-300m-cs-250` | CTC skóre textu | Apache-2.0 |
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | rozpoznávání fonémů | Apache-2.0 |
| espeak-ng | fonémy textu (volá se jako program) | GPL-3.0 |

## Data mluvčího

Licence modelu nestačí, potřebujete i **právo k hlasu**: výslovný souhlas mluvčího s klonováním a s účelem
použití. To, že jsou videa veřejná, souhlas nenahrazuje. Hlas je osobní údaj (GDPR). Zpracovávejte ho jen
k dohodnutému účelu, chraňte ho a na žádost mluvčího ho smažte.
