"""Slot parsing and scoring for free-form slot strings reported by student runtimes.

Canonical values: TIMER -> seconds, ALARM -> minutes after midnight,
TEMPERATURE -> degrees, BRIGHTNESS -> percent, COLOR / CREATE_REMINDER -> text.
Numeric intents return None when nothing parseable is found.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

SLOT_VALUES: dict[str, tuple[str, ...]] = {
    "TIMER": ("10 seconds", "30 seconds", "1 minute"),
    "ALARM": ("6:00 AM", "8:00 AM", "9:00 PM"),
    "TEMPERATURE": ("18 degrees", "22 degrees", "26 degrees"),
    "BRIGHTNESS": ("20 percent", "60 percent", "100 percent"),
    "COLOR": ("Red", "Blue", "Green"),
    "CREATE_REMINDER": ("Drink water", "Study", "Exercise"),
}
SLOTTED = set(SLOT_VALUES)

_UNITS = {"TIMER": "s", "ALARM": "min", "TEMPERATURE": "deg", "BRIGHTNESS": "%"}
_SPANS = {"TIMER": 50.0, "ALARM": 900.0, "TEMPERATURE": 8.0, "BRIGHTNESS": 80.0}
_COLORS = {
    "red", "blue", "green", "yellow", "white", "orange", "purple", "pink",
    "cyan", "magenta", "violet",
}

# --------------------------------------------------------------------------
# number words
# --------------------------------------------------------------------------
_ONES_NAMES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_ONES = {w: i for i, w in enumerate(_ONES_NAMES)}
_TENS_NAMES = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_TENS = {w: i * 10 for i, w in enumerate(_TENS_NAMES) if w != "_"}
_TENS["fourty"] = 40
_DIGIT_WORDS = {w: i for i, w in enumerate(_ONES_NAMES[:10])}
_NUM_START = set(_ONES) | set(_TENS) | {"hundred", "thousand", "half"}
# category of the new token -> categories allowed to precede it
_ALLOWED: dict[str, set[str | None]] = {
    "unit": {None, "hundred", "thousand", "tens"},
    "teen": {None, "hundred", "thousand"},
    "tens": {None, "hundred", "thousand"},
    "hundred": {None, "unit", "teen"},
    "thousand": {None, "unit", "teen", "tens", "hundred"},
    "half": {None},
}
_DIGIT_RE = re.compile(r"^\d+(?:\.\d+)?$")
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def number_to_words(n: int) -> str:
    """Spell an integer 0..9999 in English, e.g. 22 -> 'twenty two'."""
    if int(n) != n or not 0 <= n <= 9999:
        raise ValueError(f"number_to_words supports integers 0..9999, got {n!r}")
    n = int(n)
    parts: list[str] = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        parts += [_ONES_NAMES[thousands], "thousand"]
    hundreds, rest = divmod(rest, 100)
    if hundreds:
        parts += [_ONES_NAMES[hundreds], "hundred"]
    if rest and rest < 20:
        parts.append(_ONES_NAMES[rest])
    elif rest:
        tens, unit = divmod(rest, 10)
        parts.append(_TENS_NAMES[tens])
        if unit:
            parts.append(_ONES_NAMES[unit])
    return " ".join(parts) or "zero"


def _category(tok: str) -> str:
    if tok in _ONES:
        return "unit" if _ONES[tok] < 10 else "teen"
    if tok in _TENS:
        return "tens"
    return tok  # hundred / thousand / half


def _word_run(toks: list[str], i: int) -> tuple[float, int] | None:
    """Parse a well-formed run of number words starting at i; returns (value, end)."""
    total = cur = 0.0
    last: str | None = None
    frac: list[int] | None = None
    j, n = i, len(toks)
    while j < n:
        t = toks[j]
        if frac is not None:
            if t not in _DIGIT_WORDS:
                break
            frac.append(_DIGIT_WORDS[t])
            j += 1
            continue
        if t == "and" and last in ("hundred", "thousand") and j + 1 < n and (
            toks[j + 1] in _ONES or toks[j + 1] in _TENS
        ):
            j += 1
            continue
        if t == "point" and last is not None and j + 1 < n and toks[j + 1] in _DIGIT_WORDS:
            frac, last = [], "point"
            j += 1
            continue
        if t not in _NUM_START:
            break
        cat = _category(t)
        if last not in _ALLOWED[cat]:
            break
        if last is not None and (
            (cat == "unit" and _ONES[t] == 0) or (cat == "hundred" and last == "unit" and cur == 0)
        ):
            break
        if cat in ("unit", "teen"):
            cur += _ONES[t]
        elif cat == "tens":
            cur += _TENS[t]
        elif cat == "hundred":
            cur = (cur or 1) * 100
        elif cat == "thousand":
            total += (cur or 1) * 1000
            cur = 0
        else:  # half
            cur = 0.5
        last = cat
        j += 1
    if j == i:
        return None
    value = total + cur
    if frac:
        value += float("0." + "".join(map(str, frac)))
    return value, j


def _find_numbers(toks: list[str]) -> list[tuple[float, int, int]]:
    """Find numbers (digits or words) in tokens as (value, start, end) spans."""
    out: list[tuple[float, int, int]] = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if _DIGIT_RE.match(t):
            v, j = float(t), i + 1
        elif t in _NUM_START:
            run = _word_run(toks, i)
            if run is None:
                i += 1
                continue
            v, j = run
        else:
            i += 1
            continue
        if toks[j:j + 3] == ["and", "a", "half"]:
            v, j = v + 0.5, j + 3
        elif toks[j:j + 2] == ["and", "half"]:
            v, j = v + 0.5, j + 2
        out.append((v, i, j))
        i = j
    return out


def words_to_number(text: str | None) -> float | None:
    """Convert a string that is only a number ('twenty two', '22.5') to a float, else None."""
    if text is None:
        return None
    toks = [t for t in _clean(text).split() if t not in ("a", "an")]
    nums = _find_numbers(toks)
    if len(nums) == 1 and nums[0][1] == 0 and nums[0][2] == len(toks):
        return nums[0][0]
    return None


# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------
def _clean(text: Any) -> str:
    """Lowercase, normalise symbols (%, degree sign, a.m., o'clock) and punctuation."""
    s = str(text).lower()
    s = s.replace("°", " degrees ").replace("º", " degrees ").replace("%", " percent ")
    s = re.sub(r"degrees\s*(?:celsius|c)\b", "degrees", s)
    s = re.sub(r"\b([ap])\s?\.\s?m\b\.?", r"\1m", s)
    s = re.sub(r"\bo['’]?\s?clock\b", " ", s)
    s = re.sub(r"['’`]", "", s)
    s = re.sub(r"[^a-z0-9:. ]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)  # keep dots only inside numbers
    s = re.sub(r"(?<!\d):|:(?!\d)", " ", s)  # keep colons only inside times
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)  # '9pm' -> '9 pm'
    return " ".join(s.split())


def _int_words(digits: str) -> str:
    n = int(digits)
    if n <= 9999:
        return number_to_words(n)
    return " ".join(_ONES_NAMES[int(d)] for d in digits)


def _time_words(m: re.Match[str]) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    out = " " + number_to_words(hour)
    if 0 < minute < 10:
        out += " oh " + number_to_words(minute)
    elif minute:
        out += " " + number_to_words(minute)
    return out + " "


def _decimal_words(m: re.Match[str]) -> str:
    frac = " ".join(_ONES_NAMES[int(d)] for d in m.group(2))
    return f" {_int_words(m.group(1))} point {frac} "


def _spell(text: Any) -> str:
    """Normalised lowercase text with digits spelled out ('6:00 AM' -> 'six am')."""
    s = _clean(text)
    s = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", _time_words, s)
    s = re.sub(r"(\d+)\.(\d+)", _decimal_words, s)
    s = re.sub(r"\d+", lambda m: f" {_int_words(m.group())} ", s)
    return " ".join(s.split())


def _plain(text: Any) -> str:
    return _clean(text)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _unit_seconds(tok: str) -> int | None:
    if tok in ("s", "sec", "secs", "second", "seconds"):
        return 1
    if tok in ("m", "min", "mins", "minute", "minutes"):
        return 60
    if tok in ("h", "hr", "hrs", "hour", "hours"):
        return 3600
    return None


def _parse_timer(toks: list[str]) -> float | None:
    unit_toks = [t for t in toks if _unit_seconds(t)]
    nums = _find_numbers(toks)
    if not nums:
        for t in toks:
            m = _CLOCK_RE.match(t)
            if m and not unit_toks:
                return float(int(m.group(1)) * 60 + int(m.group(2)))
        if unit_toks:  # "a minute", "an hour"
            return float(_unit_seconds(unit_toks[0]))
        return None
    total = 0.0
    for v, _, end in nums:
        k = end
        while k < len(toks) and toks[k] in ("a", "an"):
            k += 1
        unit = _unit_seconds(toks[k]) if k < len(toks) else None
        if unit is None:
            unit = _unit_seconds(unit_toks[0]) if len(nums) == 1 and unit_toks else 1
        total += v * unit
    return total


def _parse_alarm(toks: list[str]) -> int | None:
    meridiem = None
    for t in toks:
        if t in ("am", "morning"):
            meridiem = "am"
        elif t in ("pm", "afternoon", "evening", "night"):
            meridiem = "pm"
    nums = _find_numbers(toks)
    clock = next(((i, _CLOCK_RE.match(t)) for i, t in enumerate(toks) if _CLOCK_RE.match(t)), None)
    if clock and (not nums or clock[0] < nums[0][1]):
        hour, minute = int(clock[1].group(1)), int(clock[1].group(2))  # type: ignore[union-attr]
    elif nums:
        value, start, end = nums[0]
        if value != int(value):
            return None
        hour, minute = int(value), 0
        if _DIGIT_RE.match(toks[start]) and len(toks[start]) in (3, 4):
            hour, minute = divmod(hour, 100)  # military '0600'
        elif len(nums) > 1 and nums[1][1] == end and nums[1][0] == int(nums[1][0]) and nums[1][0] < 60:
            minute = int(nums[1][0])  # 'six thirty'
    elif "noon" in toks:
        return 720
    elif "midnight" in toks:
        return 0
    else:
        return None
    if not (0 <= hour <= 24 and 0 <= minute < 60):
        return None
    if meridiem and 1 <= hour <= 12:
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    elif meridiem and hour == 0:
        pass
    return (hour * 60 + minute) % 1440


def _parse_plain_number(toks: list[str]) -> float | None:
    nums = _find_numbers(toks)
    return nums[0][0] if nums else None


def parse_slot(intent: str, text: Any) -> float | str | None:
    """Canonical value of a free-form slot string for the given intent (None if empty)."""
    if text is None:
        return None
    cleaned = _clean(text)
    if not cleaned:
        return None
    toks = cleaned.split()
    key = str(intent).upper()
    if key == "TIMER":
        return _parse_timer(toks)
    if key == "ALARM":
        return _parse_alarm(toks)
    if key in ("TEMPERATURE", "BRIGHTNESS"):
        return _parse_plain_number(toks)
    if key == "COLOR":
        return next((t for t in toks if t in _COLORS), cleaned)
    if key == "CREATE_REMINDER":
        return " ".join(t for t in toks if t != "to") or None
    return cleaned


# --------------------------------------------------------------------------
# phonetics and distances
# --------------------------------------------------------------------------
_VOWELS = set("aeiou")


def _encode_word(word: str) -> str:
    """Simplified Metaphone code for one lowercase word (not full Metaphone)."""
    w = re.sub(r"(.)\1+", r"\1", re.sub(r"[^a-z]", "", word))
    if not w:
        return ""
    if w[:2] in ("kn", "gn", "pn", "wr"):
        w = w[1:]
    elif w[0] == "x":
        w = "s" + w[1:]
    elif w[:2] == "wh":
        w = "w" + w[2:]
    out: list[str] = []
    n = len(w)
    i = 0
    while i < n:
        c = w[i]
        nx = w[i + 1] if i + 1 < n else ""
        nx2 = w[i + 2] if i + 2 < n else ""
        if c in _VOWELS:
            if i == 0:
                out.append(c.upper())
        elif c == "b":
            if not (i == n - 1 and i > 0 and w[i - 1] == "m"):
                out.append("B")
        elif c == "c":
            if nx == "h":
                out.append("X")
                i += 1
            elif nx == "k":
                out.append("K")
                i += 1
            else:
                out.append("S" if nx in ("e", "i", "y") else "K")
        elif c == "d":
            if nx == "g":
                out.append("J")
                i += 1
            else:
                out.append("T")
        elif c == "g":
            if nx == "h":
                if i == 0:
                    out.append("K")
                i += 1
            elif nx == "n" and i + 1 == n - 1:
                pass
            else:
                out.append("J" if nx in ("e", "i", "y") else "K")
        elif c == "h":
            if nx in _VOWELS and (i == 0 or w[i - 1] not in "cspgt"):
                out.append("H")
        elif c == "p":
            if nx == "h":
                out.append("F")
                i += 1
            else:
                out.append("P")
        elif c == "q":
            out.append("K")
        elif c == "s":
            if nx == "h":
                out.append("X")
                i += 1
            elif nx == "c" and nx2 == "h":
                out.append("X")
                i += 2
            else:
                out.append("S")
        elif c == "t":
            if nx == "c" and nx2 == "h":
                out.append("X")
                i += 2
            elif nx == "h":
                out.append("0")
                i += 1
            else:
                out.append("T")
        elif c == "v":
            out.append("F")
        elif c in ("w", "y"):
            if nx in _VOWELS:
                out.append(c.upper())
        elif c == "x":
            out.append("KS")
        elif c == "z":
            out.append("S")
        else:  # f j k l m n r
            out.append(c.upper())
        i += 1
    return "".join(out)


def phonetic_key(text: Any) -> str:
    """Simplified Metaphone key of the digit-spelled text, words joined by spaces."""
    if text is None:
        return ""
    codes = (_encode_word(w) for w in _spell(text).split())
    return " ".join(c for c in codes if c)


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Edit distance between two strings or sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm_lev(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num_diff(intent: str, a: float, b: float) -> float:
    d = abs(a - b)
    if intent == "ALARM":
        d %= 1440
        d = min(d, 1440 - d)
    return float(d)


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if _is_num(a) and _is_num(b):
        return abs(a - b) < 1e-6
    return a == b


def slot_distance(intent: str, true_text: Any, pred_text: Any) -> dict[str, Any]:
    """Compare a predicted slot string with the true one; see module docs for keys."""
    key = str(intent).upper()
    true_c = parse_slot(key, true_text)
    pred_c = parse_slot(key, pred_text)
    true_s, pred_s = _spell(true_text), _spell(pred_text) if pred_text is not None else ""
    pred_empty = not pred_s

    abs_error: float | None = None
    rel_error: float | None = None
    if key in _UNITS and _is_num(true_c) and _is_num(pred_c):
        abs_error = _num_diff(key, true_c, pred_c)  # type: ignore[arg-type]
        rel_error = abs_error / _SPANS[key]

    if pred_empty:
        phonetic = char = 1.0
    else:
        phonetic = _norm_lev(phonetic_key(true_text), phonetic_key(pred_text))
        char = _norm_lev(true_s, pred_s)

    nearest: str | None = None
    if key in SLOT_VALUES and not pred_empty:
        cands = SLOT_VALUES[key]
        if key in _UNITS and _is_num(pred_c):
            nearest = min(cands, key=lambda s: _num_diff(key, parse_slot(key, s), pred_c))  # type: ignore[arg-type]
        else:
            nearest = min(
                cands,
                key=lambda s: (
                    _norm_lev(phonetic_key(s), phonetic_key(pred_text)),
                    _norm_lev(_spell(s), pred_s),
                ),
            )
    return {
        "exact": _same(true_c, pred_c),
        "parsed_pred": pred_c,
        "abs_error": abs_error,
        "unit": _UNITS.get(key),
        "rel_error": rel_error,
        "phonetic_dist": phonetic,
        "char_dist": char,
        "nearest_schema_value": nearest,
    }
