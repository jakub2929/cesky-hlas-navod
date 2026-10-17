"""ceskyhlas/normalize_cs.py: Czech text -> words, nothing numeric reaches the TTS."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ceskyhlas.normalize_cs import cardinal, normalize, ordinal, sentences  # noqa: E402

CASES = [
    # prices
    ("Tarif stojí 499 Kč.", "Tarif stojí čtyři sta devadesát devět korun."),
    ("Cena je 1 Kč.", "Cena je jedna koruna."),
    ("Stojí to 2 Kč.", "Stojí to dvě koruny."),
    ("Za 3 Kč.", "Za tři koruny."),
    ("Celkem 21 Kč.", "Celkem dvacet jedna korun."),
    ("Stojí 1 299 Kč.", "Stojí tisíc dvě stě devadesát devět korun."),
    ("Stojí 1299 Kč.", "Stojí tisíc dvě stě devadesát devět korun."),
    ("Cena 199,90 Kč.", "Cena sto devadesát devět korun devadesát haléřů."),
    ("Cena 199,- Kč.", "Cena sto devadesát devět korun."),
    ("Poplatek 0,50 Kč.", "Poplatek padesát haléřů."),
    ("Doplatek 12,05 Kč.", "Doplatek dvanáct korun pět haléřů."),
    ("Tarif za 599 Kč/měs.", "Tarif za pět set devadesát devět korun měsíčně."),
    ("Stojí 2 500 000 Kč.", "Stojí dva miliony pět set tisíc korun."),
    ("Je to 10 000 Kč.", "Je to deset tisíc korun."),
    ("Platba 2 koruny.", "Platba dvě koruny."),
    ("Platba 1 koruna.", "Platba jedna koruna."),
    ("Cena v EUR je 20 EUR.", "Cena v EUR je dvacet eur."),
    # percentages
    ("Sleva 20 %.", "Sleva dvacet procent."),
    ("Sleva 20%.", "Sleva dvacet procent."),
    ("Navýšení o 1 %.", "Navýšení o jedno procento."),
    ("Navýšení o 2 %.", "Navýšení o dvě procenta."),
    ("Úrok 2,5 %.", "Úrok dvě celé pět procenta."),
    ("Úrok 1,5 %.", "Úrok jedna celá pět procenta."),
    # data and speed
    ("Máte 10 GB dat.", "Máte deset gigabajtů dat."),
    ("Zbývá 1 GB.", "Zbývá jeden gigabajt."),
    ("Zbývají 3 GB.", "Zbývají tři gigabajty."),
    ("Přidáme 1,5 GB.", "Přidáme jedna celá pět gigabajtu."),
    ("Rychlost 100 Mbit/s.", "Rychlost sto megabitů za sekundu."),
    ("Limit 500 MB.", "Limit pět set megabajtů."),
    ("Neomezené GB.", "Neomezené gigabajtů."),
    # phone numbers
    ("Volejte 123 456 789.", "Volejte sto dvacet tři, čtyři sta padesát šest, sedm set osmdesát devět."),
    ("Volejte 123456789.", "Volejte sto dvacet tři, čtyři sta padesát šest, sedm set osmdesát devět."),
    ("Číslo +420 123 456 789.", "Číslo plus čtyři sta dvacet, sto dvacet tři, čtyři sta padesát šest, sedm set osmdesát devět."),
    ("Linka 800 100 200.", "Linka osm set, sto, dvě stě."),
    ("Číslo 100 012 345.", "Číslo sto, nula jedna dva, tři sta čtyřicet pět."),
    # dates
    ("Faktura z 15. 3. 2025.", "Faktura z patnáctého března dva tisíce dvacet pět."),
    ("Splatnost 1. 4.", "Splatnost prvního dubna."),
    ("Od 15.3.2025 platí.", "Od patnáctého března dva tisíce dvacet pět platí."),
    ("Do 31. prosince.", "Do třicátého prvního prosince."),
    ("Dne 2. ledna 2026.", "Dne druhého ledna dva tisíce dvacet šest."),
    ("Rok 1999.", "Rok tisíc devět set devadesát devět."),
    # times
    ("Technik přijde v 14:30.", "Technik přijde ve čtrnáct hodin třicet minut."),
    ("Otevřeno od 9:00.", "Otevřeno od devět hodin."),
    ("V 8:05 ráno.", "V osm hodin pět minut ráno."),
    ("Ve 21:15.", "Ve dvacet jedna hodin patnáct minut."),
    ("Od 1:00.", "Od jedna hodina."),
    # ordinals, ranges
    ("Ve 2. patře.", "Ve druhé patře."),
    ("Je to 3. linka.", "Je to třetí linka."),
    ("Už 5. den.", "Už pátý den."),
    ("Trvá to 2-4 dny.", "Trvá to dva až čtyři dny."),
    ("Trvá to 3–5 dní.", "Trvá to tři až pět dní."),
    # plain numbers and gender
    ("Máte 1 SIM kartu.", "Máte jednu sim kartu."),
    ("Máte 2 faktury.", "Máte dvě faktury."),
    ("Mám 2 telefony.", "Mám dva telefony."),
    ("Je to 1 smlouva.", "Je to jedna smlouva."),
    ("Celkem 101 zákazníků.", "Celkem sto jedna zákazníků."),
    ("Zbývá 0 minut.", "Zbývá nula minut."),
    ("Za 30 min.", "Za třicet minut."),
    ("Pošleme 3 SMS.", "Pošleme tři esemesky."),
    # brands and abbreviations
    ("Přecházíte od O2?", "Přecházíte od ó dva?"),
    ("Jste u T-Mobile nebo Vodafone?", "Jste u tý mobajl nebo vodafoun?"),
    ("Pokrytí 5G a LTE.", "Pokrytí pět gé a el té é."),
    ("Cena bez DPH, tzn. o něco méně.", "Cena bez dé pé há, to znamená o něco méně."),
    ("Volání, SMS, data atd.", "Volání, esemes, data a tak dále."),
    ("Např. roaming, tj. data v zahraničí.", "Například roaming, to jest data v zahraničí."),
    ("Smlouva č. 12345.", "Smlouva číslo dvanáct tisíc tři sta čtyřicet pět."),
    ("Zadejte PIN a PUK.", "Zadejte pin a puk."),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_normalize(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize("text", [c[0] for c in CASES] + ["Kód 4X7", "Verze 2.0.1", "Model A52s 128GB", "x1y2z3", "3/4"])
def test_no_digit_ever_reaches_the_tts(text):
    assert not re.search(r"\d", normalize(text))


def test_cardinals():
    assert cardinal(0) == "nula"
    assert cardinal(1, "f") == "jedna" and cardinal(1, "n") == "jedno" and cardinal(1) == "jeden"
    assert cardinal(2, "f") == "dvě" and cardinal(2) == "dva"
    assert cardinal(200) == "dvě stě"
    assert cardinal(1000) == "tisíc" and cardinal(2000) == "dva tisíce" and cardinal(5000) == "pět tisíc"
    assert cardinal(21000) == "dvacet jedna tisíc"
    assert cardinal(1_000_000) == "milion" and cardinal(3_000_000) == "tři miliony"


def test_ordinals():
    assert ordinal(1, "gen_m") == "prvního" and ordinal(15, "gen_m") == "patnáctého"
    assert ordinal(21, "gen_m") == "dvacátého prvního"
    assert ordinal(2, "nom_n") == "druhé" and ordinal(2, "nom_f") == "druhá" and ordinal(3, "nom_f") == "třetí"


def test_sentences():
    s = sentences("Dobrý den. Jak vám mohu pomoci? " + "slovo, " * 60 + "konec.")
    assert s[0] == "Dobrý den." and s[1] == "Jak vám mohu pomoci?"
    assert all(len(x) <= 230 for x in s)
