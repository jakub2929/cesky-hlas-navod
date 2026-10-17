# 07 Praktické zkušenosti

## WSL a disk: soubor vhdx jen roste

Linux ve WSL 2 žije v jednom souboru `ext4.vhdx`. Ten **roste, ale sám se nezmenšuje**. Smazání 30 GB checkpointů
uvnitř Linuxu nevrátí Windows ani bajt. U nás měl vhdx 178 GB, i když uvnitř bylo obsazeno 64 GB.

Postup zmenšení:

1. uvnitř WSL smažte, co nepotřebujete, a uvolněné bloky ohlaste: `sudo fstrim -av`;
2. `wsl --shutdown`;
3. v PowerShellu **jako správce** přes `diskpart`: `select vdisk file="...\ext4.vhdx"`, `attach vdisk readonly`,
   `compact vdisk`, `detach vdisk`.

Po úklidu se u nás vhdx zmenšil ze 178 GB přes 97 a 68 GB na 44 GB. Další rady:

- Distribuci přesuňte z disku C: na datový disk (`wsl --export` / `wsl --import`, nebo `wsl --manage --move`).
  Plný C: shodí Windows i WSL najednou.
- Hlídejte volné místo na všech dotčených discích (Linux /, disk s daty, C:) a pod ~40 GB nespouštějte
  dlouhé běhy. Checkpointy mají stovky MB až GB a ukládají se často.
- Během kompaktace je vhdx zamčený a WSL nenastartuje („sharing violation“). Počkejte, až diskpart doběhne.
- Data, na která se jen čte, mohou zůstat na Windows disku (`/mnt/e/...`). Trénink a cache patří na ext4
  uvnitř WSL, přes `/mnt` je pomalý přístup k malým souborům.

## Jedna úloha na GPU

S 8 GB VRAM se dva tréninky (ani trénink + hodnocení na GPU) nevejdou. A když běží dvě sezení či dva lidé
najednou, dřív nebo později se srazí.

- Každá úloha na GPU si vezme **zámek** (soubor se jménem úlohy a PID). Jiná úloha na něj počká a nikdy
  ho neruší.
- Hodnocení (ASR, CTC, metriky) puštěné na CPU vedle tréninku je v pořádku (`CUDA_VISIBLE_DEVICES=""`,
  omezený počet vláken, `nice`).
- Noční fronty spouštějte až podle značky, že předchozí úloha opravdu skončila, ne podle času.
- Nikdy neupravujte bash skript, který právě běží. Bash ho čte po kouscích a změna uprostřed udělá nepředvídatelné
  věci.

## Checkpointy a úklid

- Ukládejte často (každých 250 kroků) a nechte si všechny, dokud je neposlechnete. Nejlepší bod bývá brzy
  ([04](04-pocket-tts-recept.md)).
- Vybraný produkční checkpoint si zkopírujte zvlášť, zapište si jeho sha256 a nic ho automaticky nemaže.
- Pro pokračování tréninku potřebujete i stav optimalizátoru (u Pocketu stovky MB) a konfiguraci.
- Uzavřené pokusy: ponechte tabulky a ukázky, checkpointy smažte.

## Zálohy a soukromí

- Kód a dokumentace do gitu. Nahrávky, přepisy, modely a prompty mluvčího nikdy, ani do soukromého repa.
  Hlídá to `.gitignore` (zvuk, checkpointy, ONNX, safetensors, data/).
- Mimo git zálohujte jen to, co se nedá znovu vyrobit: produkční checkpoint, stav pro pokračování, prompt,
  dataset. Vždy se sha256 a jen do **šifrovaného** úložiště.
- Než cokoli zveřejníte, projděte to hledáním jmen, cest, tokenů, e-mailů a telefonních čísel.
