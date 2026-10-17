"""Czech text normalisation for TTS: text -> words only (no digits reach the TTS). MIT licence.

  from ceskyhlas.normalize_cs import normalize
  normalize("Tarif stojí 499 Kč měsíčně, tj. o 20 % méně.")
  # -> 'Tarif stojí čtyři sta devadesát devět korun měsíčně, to jest o dvacet procent méně.'

Order: brand names and abbreviations that contain digits (O2, 5G), phone numbers (+420 and 9-digit groups ->
digit groups read as numbers, a leading zero digit by digit), dates (15. 3. 2025, 15. března), times (14:30),
amounts with units (Kč, haléř, %, GB/MB, Mbit/s, minut, SMS ...; decimals with "celá", koruny + haléře),
ranges (2-4 -> dva až čtyři), ordinals before a lowercase word (2. patro -> druhé patro), remaining
cardinals (thousand separators by space), abbreviations (tzn., atd., DPH ...). The noun after a number is
declined by count (1 / 2-4 / 5+ and decimals); 1 and 2 agree in gender with a known following noun.
Anything numeric left at the end is read digit by digit, so normalize() never returns a digit.
Standard library only.
"""
from __future__ import annotations

import re

ONES = ["nula", "jeden", "dva", "tři", "čtyři", "pět", "šest", "sedm", "osm", "devět", "deset", "jedenáct",
        "dvanáct", "třináct", "čtrnáct", "patnáct", "šestnáct", "sedmnáct", "osmnáct", "devatenáct"]
TENS = ["", "", "dvacet", "třicet", "čtyřicet", "padesát", "šedesát", "sedmdesát", "osmdesát", "devadesát"]
HUNDREDS = ["", "sto", "dvě stě", "tři sta", "čtyři sta", "pět set", "šest set", "sedm set", "osm set", "devět set"]
DIGITS = ["nula", "jedna", "dva", "tři", "čtyři", "pět", "šest", "sedm", "osm", "devět"]

# ordinals: masculine nominative (hard -ý / soft -í); other forms come from the ending
ORD_ONES = ["nultý", "první", "druhý", "třetí", "čtvrtý", "pátý", "šestý", "sedmý", "osmý", "devátý", "desátý",
            "jedenáctý", "dvanáctý", "třináctý", "čtrnáctý", "patnáctý", "šestnáctý", "sedmnáctý", "osmnáctý",
            "devatenáctý"]
ORD_TENS = ["", "", "dvacátý", "třicátý", "čtyřicátý", "padesátý", "šedesátý", "sedmdesátý", "osmdesátý",
            "devadesátý"]
MONTHS_GEN = ["ledna", "února", "března", "dubna", "května", "června", "července", "srpna", "září", "října",
              "listopadu", "prosince"]

# units: (singular nominative, plural nominative 2-4, genitive plural 5+, genitive singular for decimals, gender)
UNITS = {
    "kč": ("koruna", "koruny", "korun", "koruny", "f"),
    "korun": ("koruna", "koruny", "korun", "koruny", "f"),
    "haléř": ("haléř", "haléře", "haléřů", "haléře", "m"),
    "hal": ("haléř", "haléře", "haléřů", "haléře", "m"),
    "%": ("procento", "procenta", "procent", "procenta", "n"),
    "gb": ("gigabajt", "gigabajty", "gigabajtů", "gigabajtu", "m"),
    "mb": ("megabajt", "megabajty", "megabajtů", "megabajtu", "m"),
    "tb": ("terabajt", "terabajty", "terabajtů", "terabajtu", "m"),
    "mbit/s": ("megabit za sekundu", "megabity za sekundu", "megabitů za sekundu", "megabitu za sekundu", "m"),
    "mb/s": ("megabit za sekundu", "megabity za sekundu", "megabitů za sekundu", "megabitu za sekundu", "m"),
    "gbit/s": ("gigabit za sekundu", "gigabity za sekundu", "gigabitů za sekundu", "gigabitu za sekundu", "m"),
    "min": ("minuta", "minuty", "minut", "minuty", "f"),
    "sms": ("esemeska", "esemesky", "esemesek", "esemesky", "f"),
    "ks": ("kus", "kusy", "kusů", "kusu", "m"),
    "km": ("kilometr", "kilometry", "kilometrů", "kilometru", "m"),
    "eur": ("euro", "eura", "eur", "eura", "n"),
    "€": ("euro", "eura", "eur", "eura", "n"),
}
# gender of nouns that commonly follow a number (only 1 and 2 change: jeden / jedna / jedno, dva / dvě)
FEM = ("korun", "koruny", "koruna", "hodin", "hodiny", "hodina", "minut", "minuty", "minuta", "sekund", "sekundy",
       "sekunda", "faktur", "faktury", "faktura", "smlouv", "smlouvy", "smlouva", "služb", "služby", "služba",
       "karty", "karta", "SIM", "linky", "linka", "zprávy", "zpráva", "esemesky", "esemeska", "platby", "platba",
       "splátky", "splátka", "pobočky", "pobočka", "noci", "noc", "věci", "věc", "osoby", "osoba", "týdny")
NEUT = ("procenta", "procento", "čísla", "číslo", "čísel", "eura", "euro", "zařízení", "telefonní")
MASC_OVERRIDE = ("týdny", "týdnů", "týden", "dny", "dní", "den", "měsíce", "měsíců", "měsíc", "roky", "let", "rok")


FEM_ACC = ("kartu", "fakturu", "smlouvu", "službu", "linku", "zprávu", "platbu", "splátku", "pobočku", "korunu",
           "hodinu", "minutu", "sekundu", "osobu", "esemesku", "noc", "věc")


def _gender_of(words: str) -> str:
    """m / f / n / fa (feminine accusative: jednu) from the word(s) after a number; a SIM before the noun is skipped."""
    ws = [w.strip(".,;:!?").lower() for w in words.split()]
    if len(ws) > 1 and ws[0] in ("sim", "esim", "nové", "novou", "nová", "další"):
        ws = ws[1:]
    w = ws[0] if ws else ""
    if w in FEM_ACC:
        return "fa"
    if w in MASC_OVERRIDE:
        return "m"
    if w in (x.lower() for x in NEUT):
        return "n"
    if w in (x.lower() for x in FEM):
        return "f"
    return "m"


def _below_1000(n: int, gender: str = "m", final: bool = True) -> str:
    h, rest = divmod(n, 100)
    parts = [HUNDREDS[h]] if h else []
    if rest:
        if rest < 20:
            if final and rest == 1:
                parts.append({"m": "jeden", "f": "jedna", "n": "jedno", "fa": "jednu"}[gender] if not parts else "jedna")
            elif final and rest == 2 and gender in ("f", "n", "fa"):
                parts.append("dvě")
            else:
                parts.append(ONES[rest])
        else:
            t, o = divmod(rest, 10)
            parts.append(TENS[t])
            if o:
                parts.append("jedna" if o == 1 else ("dvě" if o == 2 and gender in ("f", "n", "fa") else ONES[o]))
    return " ".join(parts)


def cardinal(n: int, gender: str = "m") -> str:
    """Czech cardinal in words; gender (m / f / n) affects a final 1 or 2 (jeden / jedna / jedno, dva / dvě)."""
    if n < 0:
        return "mínus " + cardinal(-n, gender)
    if n == 0:
        return "nula"
    out = []
    for value, (sg, pl, gen) in ((10 ** 9, ("miliarda", "miliardy", "miliard")),
                                 (10 ** 6, ("milion", "miliony", "milionů")),
                                 (10 ** 3, ("tisíc", "tisíce", "tisíc"))):
        q, n = divmod(n, value)
        if q:
            g = "f" if value == 10 ** 9 else "m"
            if q == 1:
                out.append(sg if value != 10 ** 9 else "jedna miliarda")
            else:
                word = pl if 2 <= q <= 4 else gen
                out.append(f"{_below_1000(q, g)} {word}")
    if n:
        out.append(_below_1000(n, gender))
    return " ".join(out)


def _form(n: int, unit: tuple) -> str:
    return unit[0] if n == 1 else (unit[1] if 2 <= n <= 4 else unit[2])


def ordinal(n: int, case: str = "nom_m") -> str:
    """Ordinal 1-99: nom_m (druhý), nom_n (druhé), nom_f (druhá), gen_m (druhého)."""
    if n < 20:
        words = [ORD_ONES[n]]
    else:
        t, o = divmod(n, 10)
        words = [ORD_TENS[t]] + ([ORD_ONES[o]] if o else [])

    def decline(w: str) -> str:
        soft = w.endswith("í")
        if case == "gen_m":
            return w[:-1] + ("ího" if soft else "ého")
        if case == "nom_n":
            return w if soft else w[:-1] + "é"
        if case == "nom_f":
            return w if soft else w[:-1] + "á"
        return w
    return " ".join(decline(w) for w in words)


def _num(s: str) -> int:
    return int(re.sub(r"[\s ]", "", s))


def _decimal(int_part: int, frac: str, unit: tuple | None = None, gender: str = "f") -> str:
    """1,5 -> jedna celá pět; 2,25 -> dvě celé dvacet pět; noun after a decimal: genitive singular."""
    cela = "celá" if int_part == 1 else ("celé" if 2 <= int_part <= 4 else "celých")
    frac_words = " ".join(DIGITS[int(d)] for d in frac) if frac.startswith("0") else cardinal(int(frac), "f")
    text = f"{cardinal(int_part, 'f')} {cela} {frac_words}"
    return f"{text} {unit[3]}" if unit else text


def _phone(m: re.Match) -> str:
    out = []
    if m.group(1):
        out.append("plus " + " ".join(cardinal(int(d)) if len(d) > 1 else DIGITS[int(d)] for d in [m.group(1).strip("+ ")]))
    for g in m.groups()[1:]:
        out.append(" ".join(DIGITS[int(d)] for d in g) if g.startswith("0") else cardinal(int(g)))
    return ", ".join(out)


def _amount(m: re.Match) -> str:
    num, frac, unit_raw = m.group("num"), m.group("frac"), m.group("unit")
    key = unit_raw.lower().replace(" ", "").rstrip(".")
    unit = UNITS.get(key)
    n = _num(num)
    if unit is None:
        return m.group(0)
    if frac and frac != "-" and key in ("kč", "korun"):
        hal = int(frac.ljust(2, "0")[:2])
        text = f"{cardinal(n, 'f')} {_form(n, unit)}"
        hal_text = f"{cardinal(hal, 'm')} {_form(hal, UNITS['haléř'])}"
        if n == 0:
            return hal_text if hal else text
        return text if hal == 0 else f"{text} {hal_text}"
    if frac and frac != "-":
        return _decimal(n, frac, unit)
    return f"{cardinal(n, unit[4])} {_form(n, unit)}"


def _date(m: re.Match) -> str:
    day, month, year = int(m.group("d")), m.group("m"), m.group("y")
    mon = MONTHS_GEN[int(month) - 1] if month.isdigit() else month
    out = f"{ordinal(day, 'gen_m')} {mon}"
    return out + (f" {cardinal(int(year))}" if year else "")


def _time(m: re.Match) -> str:
    h, mi = int(m.group("h")), int(m.group("mi"))
    hours = f"{cardinal(h, 'f')} {_form(h, ('hodina', 'hodiny', 'hodin'))}"
    return hours if mi == 0 else f"{hours} {cardinal(mi, 'f')} {_form(mi, ('minuta', 'minuty', 'minut'))}"


def _ordinal_in_context(m: re.Match) -> str:
    word = m.group(2)
    case = "nom_n" if word.endswith(("o", "í", "e")) else ("nom_f" if word.endswith("a") else "nom_m")
    if word in MASC_OVERRIDE or word.endswith(("den", "týden", "měsíc", "rok")):
        case = "nom_m"
    return ordinal(int(m.group(1)), case) + " "


def _cardinal_in_context(m: re.Match) -> str:
    n = _num(m.group("num"))
    nxt = m.group("next") or ""
    return cardinal(n, _gender_of(nxt)) + nxt


REPLACE = [   # literal / regex replacements before numbers (brands, abbreviations with digits)
    (r"\bO2\b", "ó dva"), (r"\bT-Mobile\b", "tý mobajl"), (r"\bVodafone\b", "vodafoun"),
    (r"\b5G\b", "pět gé"), (r"\b4G\b", "čtyři gé"), (r"\b3G\b", "tři gé"), (r"\bLTE\b", "el té é"),
    (r"\bWi-?Fi\b", "vajfaj"), (r"\b24/7\b", "nonstop"),
]
ABBREV = [   # after numbers
    (r"\btzn\.", "to znamená"), (r"\batd\.", "a tak dále"), (r"\bapod\.", "a podobně"), (r"\bnapř\.", "například"),
    (r"\btj\.", "to jest"), (r"\bresp\.", "respektive"), (r"\bčj\.", "číslo jednací"), (r"\bč\.\s*", "číslo "),
    (r"\bmj\.", "mimo jiné"), (r"\bpř\.", "příklad"), (r"\bcca\b\.?", "zhruba"), (r"\bDPH\b", "dé pé há"),
    (r"\bIČO\b", "ičo"), (r"\bSIM\b", "sim"), (r"\bSMS\b", "esemes"), (r"\bMMS\b", "emem es"), (r"\bPIN\b", "pin"),
    (r"\bPUK\b", "puk"), (r"\bGB\b", "gigabajtů"), (r"\bMB\b", "megabajtů"), (r"\bMbit/s\b", "megabitů za sekundu"),
    (r"\bKč\b", "korun"), (r"/\s*měs\.?", " měsíčně"), (r"\bměs\.", "měsíčně"), (r"\bhod\.", "hodin"),
    (r"\bmin\.", "minut"), (r"\bsek\.", "sekund"), (r"\bul\.", "ulice"), (r"\bpls\b", "prosím"), (r"&", " a "),
    (r"\bTV\b", "televize"), (r"\bID\b", "í dé"),
]
NUM = r"\d{1,3}(?:[  ]\d{3})+|\d+"
UNIT_RE = r"Kč|hal\.|%|GB\b|MB\b|TB\b|Mbit/s|Mb/s|Gbit/s|min\b|SMS\b|ks\b|km\b|EUR\b|€"


def normalize(text: str) -> str:
    t = text.replace(" ", " ")
    for pat, rep in REPLACE:
        t = re.sub(pat, rep, t)
    # phone numbers: +420 123 456 789 / 123 456 789 / 123456789
    t = re.sub(r"(\+\d{3}\s?)?(?<!\d)(\d{3})\s?(\d{3})\s?(\d{3})(?!\d)", _phone, t)
    # dates: 15. 3. 2025 / 15.3. / 15. března 2025
    t = re.sub(r"(?<!\d)(?P<d>[0-3]?\d)\.\s?(?P<m>1[0-2]|0?[1-9])\.(?:\s?(?P<y>(?:19|20)\d{2}))?(?!\d)", _date, t)
    t = re.sub(r"(?<!\d)(?P<d>[0-3]?\d)\.\s(?P<m>" + "|".join(MONTHS_GEN) + r")(?:\s(?P<y>(?:19|20)\d{2}))?", _date, t)
    # times: 14:30 / 9.05 h / 14:00
    t = re.sub(r"(?<!\d)(?P<h>[01]?\d|2[0-3])[:](?P<mi>[0-5]\d)(?!\d)", _time, t)
    t = re.sub(r"(?<!\d)(?P<h>[01]?\d|2[0-3])\.(?P<mi>[0-5]\d)\s?h(?:od\.?)?\b", _time, t)
    # amounts with units: 499 Kč, 199,90 Kč, 199,- Kč, 20 %, 2,5 %, 10 GB, 100 Mbit/s
    t = re.sub(r"(?P<num>" + NUM + r")(?:,(?P<frac>\d+|-))?\s?(?P<unit>" + UNIT_RE + r")", _amount, t)
    # ranges: 2-4, 2–4
    t = re.sub(r"(?<![\d,])(\d+)\s?[–-]\s?(\d+)(?![\d,])", lambda m: f"{cardinal(int(m.group(1)))} až {cardinal(int(m.group(2)))}", t)
    # ordinals before a lowercase word: 2. patro (neuter -o / -í), 2. linka (feminine -a), 2. den
    t = re.sub(r"(?<!\d)(\d{1,2})\.\s(?=([a-záčďéěíňóřšťúůýž]+))", _ordinal_in_context, t)
    # decimals without a unit: 1,5 -> jedna celá pět
    t = re.sub(r"(?<!\d)(\d+),(\d+)(?!\d)", lambda m: _decimal(int(m.group(1)), m.group(2)), t)
    # remaining cardinals (gender from the next word)
    t = re.sub(r"(?P<num>" + NUM + r")(?P<next>\s+[^\s\d]+(?:\s+[^\s\d]+)?)?", _cardinal_in_context, t)
    for pat, rep in ABBREV:
        if pat == pat.lower():   # lowercase abbreviations also at a sentence start: Např. -> Například
            t = re.sub(pat, lambda m, r=rep: r[0].upper() + r[1:] if m.group(0)[:1].isupper() else r, t, flags=re.I)
        else:
            t = re.sub(pat, rep, t)
    t = re.sub(r"\b[vV] (čtyři|čtrnáct|čtvrt)", lambda m: m.group(0)[0] + "e " + m.group(1), t)   # v -> ve
    t = re.sub(r"\d", lambda m: " " + DIGITS[int(m.group(0))] + " ", t)   # never let a digit through
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" ([,.;:!?])", r"\1", t)
    t = t.strip()
    if text.rstrip().endswith(".") and t and t[-1] not in ".!?…":   # "... atd." kept its sentence end
        t += "."
    return t


def sentences(text: str, max_chars: int = 220) -> list[str]:
    """Split normalised text into sentences for streaming; overlong ones at commas."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    out = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(", ", 0, max_chars)
            if cut <= 0:
                cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                break
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        out.append(p)
    return out
