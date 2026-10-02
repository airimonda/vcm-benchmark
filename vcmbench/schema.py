"""The agreed label space: 19 intents (+ out of scope) and 93 command variations.

Students' runtimes name their intents differently ("SET_TIMER", "lights_on",
"TEMPERATURE_22", "unknown", ...). `normalize_prediction` maps whatever the Pi
printed onto this schema. Extra aliases can be added in the config file.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

INTENTS = [
    "PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "LIGHT_ON", "LIGHT_OFF", "WEATHER", "TIME", "CALL", "MESSAGE",
    "LIST_REMINDERS", "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS",
    "COLOR", "CREATE_REMINDER",
]
OOS = "OUT_OF_SCOPE"
NONE = "NO_RESPONSE"          # the Pi did not fire anything for this trial
REJECT = "REJECT"             # scoring class: OUT_OF_SCOPE or NO_RESPONSE
SLOTTED = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}

VARIATIONS_CSV = Path(__file__).with_name("variations.csv")


@dataclass(frozen=True)
class Variation:
    intent: str
    number: int
    value: str      # slot value, "" for fixed commands
    phrase: str     # the variation name used as the 93-way label


def load_variations(path: Path = VARIATIONS_CSV) -> list[Variation]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Variation(r["label"], int(r["variation"]), r["value"] or "", r["phrase"])
                for r in csv.DictReader(f)]


VARIATIONS = load_variations()
VARIATION_BY_PHRASE = {v.phrase.lower(): v for v in VARIATIONS}
assert len(VARIATIONS) == 93, len(VARIATIONS)


def _phrase_key(text: str) -> str:
    """Phrase comparison key: lowercase, punctuation dropped, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", "", str(text).lower().replace("_", " ")).split())


VARIATION_BY_KEY = {_phrase_key(v.phrase): v for v in VARIATIONS}
assert len(VARIATION_BY_KEY) == 93


def match_variation(value) -> Variation | None:
    """A 93-class prediction -> its Variation.

    Accepts the phrase itself ("Set the temperature to 22 degrees", any case,
    punctuation ignored) or `variation_id` = 0-based row number in
    vcmbench/variations.csv (as int or digit string).
    """
    if value is None:
        return None
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        i = int(value)
        return VARIATIONS[i] if 0 <= i < len(VARIATIONS) else None
    return VARIATION_BY_KEY.get(_phrase_key(value))

# Common names other runtimes use. Keys are already normalised (see _key).
BUILTIN_ALIASES = {
    "PLAY": "PLAY_MUSIC", "MUSIC": "PLAY_MUSIC", "PLAY_SONG": "PLAY_MUSIC", "START_MUSIC": "PLAY_MUSIC",
    "PAUSE_MUSIC": "PAUSE", "PAUSE_SONG": "PAUSE",
    "STOP_MUSIC": "STOP", "STOP_PLAYBACK": "STOP",
    "NEXT_SONG": "NEXT", "SKIP": "NEXT", "SKIP_SONG": "NEXT", "NEXT_TRACK": "NEXT",
    "VOLUMEUP": "VOLUME_UP", "INCREASE_VOLUME": "VOLUME_UP", "VOL_UP": "VOLUME_UP", "LOUDER": "VOLUME_UP",
    "VOLUMEDOWN": "VOLUME_DOWN", "DECREASE_VOLUME": "VOLUME_DOWN", "VOL_DOWN": "VOLUME_DOWN",
    "LOWER_VOLUME": "VOLUME_DOWN", "QUIETER": "VOLUME_DOWN",
    "LIGHTS_ON": "LIGHT_ON", "TURN_ON_LIGHTS": "LIGHT_ON", "TURN_ON_LIGHT": "LIGHT_ON", "LIGHTON": "LIGHT_ON",
    "LIGHTS_OFF": "LIGHT_OFF", "TURN_OFF_LIGHTS": "LIGHT_OFF", "TURN_OFF_LIGHT": "LIGHT_OFF", "LIGHTOFF": "LIGHT_OFF",
    "GET_WEATHER": "WEATHER", "WEATHER_QUERY": "WEATHER",
    "GET_TIME": "TIME", "TELL_TIME": "TIME", "TIME_QUERY": "TIME",
    "MAKE_CALL": "CALL", "PHONE_CALL": "CALL", "CALL_CONTACT": "CALL",
    "SEND_MESSAGE": "MESSAGE", "SEND_MSG": "MESSAGE", "TEXT": "MESSAGE",
    "LIST_REMINDER": "LIST_REMINDERS", "SHOW_REMINDERS": "LIST_REMINDERS", "GET_REMINDERS": "LIST_REMINDERS",
    "SET_TIMER": "TIMER", "START_TIMER": "TIMER", "COUNTDOWN": "TIMER",
    "SET_ALARM": "ALARM", "WAKE_UP": "ALARM",
    "SET_TEMPERATURE": "TEMPERATURE", "TEMP": "TEMPERATURE", "SET_TEMP": "TEMPERATURE",
    "SET_TEMPERATURE_REAL": "TEMPERATURE", "AIRCON": "TEMPERATURE", "THERMOSTAT": "TEMPERATURE",
    "SET_BRIGHTNESS": "BRIGHTNESS", "DIM": "BRIGHTNESS",
    "SET_COLOR": "COLOR", "CHANGE_COLOR": "COLOR", "SET_COLOUR": "COLOR", "COLOUR": "COLOR",
    "LIGHT_COLOR": "COLOR",
    "REMINDER": "CREATE_REMINDER", "SET_REMINDER": "CREATE_REMINDER", "ADD_REMINDER": "CREATE_REMINDER",
    "REMIND": "CREATE_REMINDER", "REMIND_ME": "CREATE_REMINDER",
    # rejections
    "OOS": OOS, "UNKNOWN": OOS, "NONE": OOS, "NULL": OOS, "REJECT": OOS, "REJECTED": OOS,
    "OTHER": OOS, "NOISE": OOS, "SILENCE": OOS, "BACKGROUND": OOS, "UNK": OOS, "NO_INTENT": OOS,
    "NOT_UNDERSTOOD": OOS, "FALLBACK": OOS, "OUT_OF_SCOPE": OOS,
}


def _key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")


def build_alias_table(extra: dict[str, str] | None = None) -> dict[str, str]:
    table = {i: i for i in INTENTS}
    table.update(BUILTIN_ALIASES)
    for k, v in (extra or {}).items():
        table[_key(k)] = v if v in INTENTS or v == OOS else _key(v)
    return table


def normalize_prediction(intent: str | None, slot: str | None,
                         aliases: dict[str, str]) -> tuple[str, str, bool]:
    """Map a raw (intent, slot) printed by the Pi onto the schema.

    Returns (intent, slot, known). `intent` is one of INTENTS, OOS, or
    "OTHER:<raw>" when no alias matches (known=False). A joint name such as
    "TEMPERATURE_22" or "SET_TIMER_30_SECONDS" is split into intent + slot
    when no separate slot was given.
    """
    if intent is None or str(intent).strip() == "":
        return OOS, "", True
    k = _key(intent)
    slot = (slot or "").strip()
    if k in aliases:
        return aliases[k], slot, True
    # longest alias that is a prefix of the name; the rest is the slot
    parts = k.split("_")
    for n in range(len(parts) - 1, 0, -1):
        head = "_".join(parts[:n])
        if head in aliases:
            tail = " ".join(parts[n:]).lower()
            return aliases[head], slot or tail, True
    return f"OTHER:{k}", slot, False


def resolve_prediction(intent: str | None, slot: str | None, variation, aliases: dict[str, str]
                       ) -> tuple[str, str, bool, str]:
    """Like normalize_prediction, but also accepts a 93-class output.

    `variation` (or an `intent` that is itself one of the 93 phrases) is turned
    into its intent + slot value. Returns (intent, slot, known, variation phrase
    or "").
    """
    v = match_variation(variation) if variation not in (None, "") else None
    if v is None and intent and " " in str(intent).strip():
        v = match_variation(intent)
    if v is not None:
        return v.intent, v.value, True, v.phrase
    if not intent and variation not in (None, ""):
        # not one of the 93 phrases: an alias like "OUT_OF_SCOPE", else flagged as unknown
        intent = str(variation)
    i, s, known = normalize_prediction(intent, slot, aliases)
    return i, s, known, ""


def variations_for(intent: str, value: str = "") -> list[Variation]:
    return [v for v in VARIATIONS if v.intent == intent and (not value or v.value.lower() == value.lower())]
