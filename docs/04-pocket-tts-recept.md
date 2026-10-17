# 04 Recept: doladění Pocket TTS Czech na jednoho mluvčího

Pocket TTS (Kyutai) je malý autoregresní model s kodekem Mimi, poběží v reálném čase na CPU. Česká verze
`vvolhejn/pocket-tts-czech` (6 vrstev, CC BY 4.0) umí česky výborně, ale hlas z promptu jen naznačí. Doladěním
na ~3,5 h mluvčího A se podobnost zvedla z 0,32 na 0,44 při zachování srozumitelnosti.

Trénovací skripty jsou v repu Pocket TTS (`training/`, MIT). Tady je, jak jsme je použili.

## Data

1. Dataset z [01](01-data.md) po CTC filtru. Zahoďte nejhorších ~10 % podle CTC a promluvy s čísly, jejichž
   pád je nejistý („s 5 lidmi“ – automat neví, jestli „pěti“).
2. Zvuk: 24 kHz mono PCM16 (vzorkování Mimi), horní propust 70 Hz, aktivní úroveň −20 dBFS, špička ≤ −1 dBFS.
3. Validace = **celá videa** (~6 % minut), ne náhodné promluvy.
4. Manifest JSONL, jeden řádek na promluvu:

   ```json
   {"path": "/data/wavs/abc_0001.wav", "duration": 7.4, "transcript": "Text bez číslic, jak ho mluvčí řekl."}
   ```

5. Slovní zarovnání vlastním nástrojem Pocketu s českým wav2vec2:

   ```bash
   cd pocket-tts
   python -m training.scripts.align_data train.jsonl train_aligned.jsonl \
       --model comodoro/wav2vec2-xls-r-300m-cs-250 --device cuda
   ```

   Výstup má navíc `"words": [{"word", "start", "end"}, ...]`.

## Nastavení (GPU 8 GB)

Základ je autorův `finetune.yaml` („stejný jazyk, nový hlas“), zmenšený na 8 GB:

```yaml
start_from_pretrained: true
freeze_head: true            # šetří paměť; s ním vznikl náš nejlepší checkpoint
data:
  max_duration_sec: 30.0
  max_voice_prompt_sec: 5.0
  precompute: false          # předvýpočet latentů kompiluje Mimi (torch.compile) a na WSL padal
flow: {type: lsd}
flow_batch_multiplier: 4
eos_loss_weight: 0.1
text_dropout: 0.2
voice_dropout: 0.2
optim: {lr: 2e-5, schedule: constant, warmup_steps: 500}
batch_size: 8
grad_accum_steps: 8          # efektivní dávka 64 jako v originále
max_steps: 4000
ema_decay: 0.999
ckpt_freq: 250               # ukládejte často, nejlepší bod je brzy
num_ckpt_keep: 999
```

Na GPU s 8 GB trvalo 2 500 kroků asi 1,5 hodiny.

## Přeučení přichází brzy

| kroky | podobnost (telefon) | WER (telefon) |
|---|---|---|
| 0 (jen prompt) | 0,32 | 0,15 |
| 500 | 0,41 | 0,17 |
| 1 500 | 0,43 | 0,19 |
| 3 000 | 0,45 | 0,27 |
| 4 800 | 0,46 | 0,37 |

Podobnost roste pomalu, srozumitelnost padá rychle. Druhé kolo (lr 2e-5, `freeze_head`, stejná data) mělo
nejlepší bod u **2 500 kroků** (podobnost 0,44, WER 0,27 po A-law). Pokračování k 3 400 krokům už horšilo WER
(0,29–0,34) i výslovnost z / ř / r. Prakticky:

- ukládejte checkpoint každých 250 kroků a **všechny si nechte**, dokud je neposlechnete;
- zastavte, když WER dvakrát po sobě přeleze zvolenou mez (u nás 0,30);
- vybírá **ucho**, ne jedno číslo. Posluchač slyší chyby (spolknuté r, „Dobjej den“), které průměrné WER schová.

## Hlasový prompt rozhoduje

Pocket napodobuje prompt: tempo, energii, ale i výslovnost. Na stejném checkpointu:

| prompt | podobnost | WER | poznámka |
|---|---|---|---|
| klidný, 11 s | 0,439 | 0,268 | dobrý výchozí bod |
| vybraný na **ř**, 8,6 s | 0,434 | 0,260 | posluchač slyší jasně lepší r / ř → používáme |
| dva další bohaté na ř | 0,45 | 0,30–0,35 | podobnější, ale hůř srozumitelné |

Postup výběru: vezměte 20–40 kandidátů o délce 8–15 s. Jsou to čisté promluvy s vysokým CTC skóre a
s několika ř, r a z. Vyrenderujte s každým stejnou sadu vět bohatých na ř / r a seřaďte je podle
[05](05-hodnoceni.md) (ř→s, r po souhlásce, WER). Nejlepší 3–4 nakonec poslechněte.

## Dekódování

| parametr | výchozí knihovny | náš výběr |
|---|---|---|
| `temp` | 0,3 | **0,5** |
| `sampler_decode_steps` | 1 | **1** |
| `noise_clamp` | – | **1,0** |

Vyšší teplota s ořezem šumu (noise clamp) zněla s vybraným promptem nejpřirozeněji. Rozdíly mezi kombinacemi
jsou ale menší než šum mezi seedy ([05](05-hodnoceni.md)), proto měřte aspoň dva seedy. Stejné dekódování
použijte v serveru i v cache frází, jinak budou pevné fráze znít jinak než zbytek.

## Co zbývá

Doladěný model říká ř jako s ve 32 % případů (skutečný mluvčí v 9 %). R po souhlásce spolkne v 19 %
(mluvčí v 5,5 %). Připravený, ale zatím nespuštěný další krok: krátké cílené doladění (lr 5e-6, 1 000 kroků)
s 3× opakovanými promluvami bohatými na ř a r po souhlásce. Potom případně mix ~70 % mluvčího + ~30 % vět
z ParCzech bohatých na ř / r.
