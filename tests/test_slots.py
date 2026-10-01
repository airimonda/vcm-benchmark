import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vcmbench.slots import (  # noqa: E402
    SLOT_VALUES,
    SLOTTED,
    levenshtein,
    number_to_words,
    parse_slot,
    phonetic_key,
    slot_distance,
    words_to_number,
)


def test_constants():
    assert SLOTTED == {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}
    assert all(len(v) == 3 for v in SLOT_VALUES.values())


@pytest.mark.parametrize("text,value", [
    ("twenty two", 22), ("twenty-two", 22), ("one hundred", 100), ("a hundred", 100),
    ("one hundred and five", 105), ("six", 6), ("zero", 0), ("nineteen", 19),
    ("two thousand five hundred", 2500), ("22", 22), ("22.5", 22.5), ("half", 0.5),
    ("one and a half", 1.5), ("twenty two point five", 22.5), ("ninety", 90),
])
def test_words_to_number(text, value):
    assert words_to_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["", None, "red", "a minute", "six thirty", "twenty two degrees"])
def test_words_to_number_none(text):
    assert words_to_number(text) is None


@pytest.mark.parametrize("n,text", [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (22, "twenty two"), (60, "sixty"),
    (100, "one hundred"), (105, "one hundred five"), (1260, "one thousand two hundred sixty"),
    (9999, "nine thousand nine hundred ninety nine"),
])
def test_number_to_words(n, text):
    assert number_to_words(n) == text
    assert words_to_number(text) == n


def test_number_to_words_range():
    for bad in (-1, 10000):
        with pytest.raises(ValueError):
            number_to_words(bad)


@pytest.mark.parametrize("text,seconds", [
    ("10 seconds", 10), ("30", 30), ("1 minute", 60), ("1 min", 60), ("60s", 60), ("90 sec", 90),
    ("a minute", 60), ("one minute", 60), ("sixty seconds", 60), ("1", 1), ("2 minutes", 120),
    ("one and a half minutes", 90), ("1 minute 30 seconds", 90), ("half a minute", 30),
    ("thirty seconds", 30), ("1.5 minutes", 90), ("1:30", 90),
])
def test_parse_timer(text, seconds):
    assert parse_slot("TIMER", text) == pytest.approx(seconds)


@pytest.mark.parametrize("text,minutes", [
    ("6:00 AM", 360), ("8:00 AM", 480), ("9:00 PM", 1260), ("9 PM", 1260), ("9pm", 1260),
    ("21:00", 1260), ("06:00", 360), ("six am", 360), ("6 am", 360), ("6", 360),
    ("12 am", 0), ("12 pm", 720), ("six o'clock", 360), ("6 a.m.", 360), ("nine pm", 1260),
    ("six thirty am", 390), ("6:30 pm", 1110), ("noon", 720), ("0600", 360), ("eight", 480),
])
def test_parse_alarm(text, minutes):
    assert parse_slot("ALARM", text) == minutes


@pytest.mark.parametrize("intent,text,value", [
    ("TEMPERATURE", "22", 22), ("TEMPERATURE", "22 degrees", 22), ("TEMPERATURE", "twenty two", 22),
    ("TEMPERATURE", "22°C", 22), ("TEMPERATURE", "18.5 degrees", 18.5),
    ("BRIGHTNESS", "100%", 100), ("BRIGHTNESS", "one hundred percent", 100),
    ("BRIGHTNESS", "60 percent", 60), ("BRIGHTNESS", "twenty percent", 20),
])
def test_parse_numeric(intent, text, value):
    assert parse_slot(intent, text) == pytest.approx(value)


def test_parse_text_intents():
    assert parse_slot("COLOR", "BLUE") == "blue"
    assert parse_slot("COLOR", "dark green!") == "green"
    assert parse_slot("COLOR", "Teal") == "teal"
    assert parse_slot("CREATE_REMINDER", "Drink water") == "drink water"
    assert parse_slot("CREATE_REMINDER", "  to  drink,  water. ") == "drink water"
    assert parse_slot("PLAY_MUSIC", " Hello,  World ") == "hello world"


@pytest.mark.parametrize("intent", ["TIMER", "ALARM", "TEMPERATURE", "COLOR", "CREATE_REMINDER", "X"])
def test_parse_empty(intent):
    assert parse_slot(intent, "") is None
    assert parse_slot(intent, None) is None
    assert parse_slot(intent, "  ") is None


def test_parse_numeric_unparseable():
    assert parse_slot("TIMER", "banana") is None
    assert parse_slot("ALARM", "banana") is None
    assert parse_slot("ALARM", "25:99") is None


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein([1, 2, 3], [1, 3]) == 1


def test_phonetic_key():
    assert phonetic_key("6:00 AM") == phonetic_key("six am")
    assert phonetic_key("6:00 AM").split()[1] == "AM"
    assert phonetic_key("22") == phonetic_key("twenty two")
    assert phonetic_key("100%") == phonetic_key("one hundred percent")
    assert phonetic_key("22°") == phonetic_key("twenty two degrees")
    assert phonetic_key("blue") == phonetic_key("blew")
    assert phonetic_key("phone") == phonetic_key("fone")
    assert phonetic_key("knight") == phonetic_key("night")
    assert phonetic_key("") == ""
    assert phonetic_key(None) == ""
    assert phonetic_key("Drink Water") == phonetic_key("drink  water!")


def test_slot_distance_temperature():
    d = slot_distance("TEMPERATURE", "22 degrees", "26")
    assert d["exact"] is False
    assert d["abs_error"] == 4 and d["unit"] == "deg"
    assert d["rel_error"] == pytest.approx(0.5)
    assert d["parsed_pred"] == 26
    assert d["nearest_schema_value"] == "26 degrees"
    assert slot_distance("TEMPERATURE", "22 degrees", "twenty two")["exact"] is True
    assert slot_distance("TEMPERATURE", "22 degrees", "22 degrees")["phonetic_dist"] == 0.0


def test_slot_distance_alarm():
    assert slot_distance("ALARM", "9:00 PM", "21:00")["exact"] is True
    d = slot_distance("ALARM", "6:00 AM", "11 pm")
    assert d["abs_error"] == 420 and d["unit"] == "min"
    assert d["rel_error"] == pytest.approx(420 / 900)
    assert d["exact"] is False
    assert slot_distance("ALARM", "6:00 AM", "6 am")["exact"] is True
    assert slot_distance("ALARM", "6:00 AM", "six o'clock")["exact"] is True
    assert slot_distance("ALARM", "6:00 AM", "6:00 PM")["abs_error"] == 720
    assert slot_distance("ALARM", "9:00 PM", "7 am")["nearest_schema_value"] == "6:00 AM"


def test_slot_distance_timer_brightness():
    assert slot_distance("TIMER", "1 minute", "60 seconds")["exact"] is True
    d = slot_distance("TIMER", "30 seconds", "1 min")
    assert d["abs_error"] == 30 and d["unit"] == "s"
    assert d["nearest_schema_value"] == "1 minute"
    d = slot_distance("BRIGHTNESS", "100 percent", "60%")
    assert d["abs_error"] == 40 and d["unit"] == "%" and d["rel_error"] == pytest.approx(0.5)
    assert slot_distance("BRIGHTNESS", "100 percent", "one hundred percent")["exact"] is True


def test_slot_distance_color():
    blew = slot_distance("COLOR", "Blue", "blew")
    assert blew["exact"] is False
    assert blew["phonetic_dist"] < 0.34
    assert blew["abs_error"] is None and blew["rel_error"] is None and blew["unit"] is None
    assert blew["nearest_schema_value"] == "Blue"
    far = slot_distance("COLOR", "Red", "Green")
    assert far["phonetic_dist"] > blew["phonetic_dist"]
    assert far["phonetic_dist"] > 0.34
    assert slot_distance("COLOR", "Blue", "BLUE")["exact"] is True


def test_slot_distance_reminder():
    assert slot_distance("CREATE_REMINDER", "Drink water", "drink water")["exact"] is True
    d = slot_distance("CREATE_REMINDER", "Drink water", "water")
    assert d["exact"] is False and 0 < d["phonetic_dist"] < 1
    assert d["nearest_schema_value"] == "Drink water"
    assert slot_distance("CREATE_REMINDER", "Study", "studying")["nearest_schema_value"] == "Study"


@pytest.mark.parametrize("pred", ["", None, "   "])
def test_slot_distance_empty_pred(pred):
    for intent, true in [("TIMER", "1 minute"), ("COLOR", "Red")]:
        d = slot_distance(intent, true, pred)
        assert d["exact"] is False
        assert d["phonetic_dist"] == 1.0 and d["char_dist"] == 1.0
        assert d["parsed_pred"] is None
        assert d["abs_error"] is None and d["rel_error"] is None
        assert d["nearest_schema_value"] is None


def test_slot_distance_unparseable_numeric():
    d = slot_distance("TIMER", "30 seconds", "banana")
    assert d["exact"] is False and d["abs_error"] is None and d["parsed_pred"] is None
    assert 0 < d["phonetic_dist"] <= 1.0
    assert d["nearest_schema_value"] in SLOT_VALUES["TIMER"]


def test_slot_distance_unslotted_intent():
    d = slot_distance("PLAY_MUSIC", "", "")
    assert d["abs_error"] is None and d["nearest_schema_value"] is None
