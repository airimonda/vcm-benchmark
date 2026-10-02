import json
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vcmbench import audio as A                      # noqa: E402
from vcmbench import metrics as M                    # noqa: E402
from vcmbench import pi as P                         # noqa: E402
from vcmbench import schema as S                     # noqa: E402
from vcmbench.report import render_markdown, score   # noqa: E402


# ---------------------------------------------------------------- parsing / schema

@pytest.mark.parametrize("line,intent,slot,infer", [
    ('{"intent": "TIMER", "slot": "30 seconds", "infer_ms": 85}', "TIMER", "30 seconds", 85.0),
    ("2026-10-02 INFO intent=TIMER slot=30 seconds conf=0.91", "TIMER", "30 seconds", None),
    ("command: LIGHT_ON", "LIGHT_ON", "", None),
    ("heard intent=COLOR, slot=blue, infer_ms=55.2", "COLOR", "blue", 55.2),
    ('{"command": "ALARM", "slots": {"time": "6:00 AM"}}', "ALARM", "6:00 AM", None),
    ('intent=ALARM slot="6:00 AM" latency_ms=120', "ALARM", "6:00 AM", 120.0),
])
def test_parser_commands(line, intent, slot, infer):
    e = P.LineParser().parse(line, 1.0)
    assert (e.kind, e.intent, e.slot, e.infer_ms) == ("command", intent, slot, infer)


def test_parser_wake_and_noise():
    p = P.LineParser()
    assert p.parse("[vcm] wake word detected (0.93)", 0).kind == "wake"
    assert p.parse('{"event": "wake"}', 0).kind == "wake"
    assert p.parse("loading model ...", 0) is None


def test_parser_custom_regex():
    p = P.LineParser(r"RESULT: (?P<intent>\w+) \((?P<slot>[^)]*)\)")
    e = p.parse("RESULT: TIMER (1 minute)", 0)
    assert (e.intent, e.slot) == ("TIMER", "1 minute")


@pytest.mark.parametrize("raw,slot,expect", [
    ("SET_TIMER", "", ("TIMER", "")), ("lights on", "", ("LIGHT_ON", "")),
    ("set_temperature_22", "", ("TEMPERATURE", "22")), ("COLOR_BLUE", "", ("COLOR", "blue")),
    ("unknown", "", ("OUT_OF_SCOPE", "")), ("", "", ("OUT_OF_SCOPE", "")),
])
def test_normalize(raw, slot, expect):
    assert S.normalize_prediction(raw, slot, S.build_alias_table())[:2] == expect


def test_normalize_unknown_and_custom_alias():
    assert S.normalize_prediction("FOO", "", S.build_alias_table())[0] == "OTHER:FOO"
    assert S.normalize_prediction("foo", "", S.build_alias_table({"foo": "PAUSE"}))[0] == "PAUSE"


def test_schema_counts():
    assert len(S.INTENTS) == 19 and len(S.VARIATIONS) == 93


# ---------------------------------------------------------------- metrics

def test_classification_report_basic():
    yt = ["A", "A", "B", "REJECT", "REJECT"]
    yp = ["A", "B", "B", "REJECT", "A"]
    r = M.classification_report(yt, yp)
    assert r["accuracy"] == pytest.approx(3 / 5)
    assert r["per_class"]["A"]["precision"] == pytest.approx(1 / 2)
    assert r["per_class"]["B"]["precision"] == pytest.approx(1 / 2)
    assert r["balanced_accuracy"] == pytest.approx((0.5 + 1 + 0.5) / 3)
    assert r["false_accept_rate"] == pytest.approx(0.5)
    assert r["misfire_rate"] == pytest.approx(1 / 3)
    p, rc = 0.5, 0.5
    assert r["per_class"]["A"]["f2"] == pytest.approx(5 * p * rc / (4 * p + rc))


def test_command_label():
    assert M.command_label("TEMPERATURE", "22", None, False) == "Temperature 22 degrees"
    assert M.command_label("COLOR", "", None, False) == "COLOR:?"
    assert M.command_label("ALARM", "9pm", "Wake me up at 9:00 PM", True) == "Wake me up at 9:00 PM"
    assert M.command_label(S.NONE, "", None, False) == "REJECT"


def _trial(intent, var, slot, pred, pslot, **kw):
    t = {"true_intent": intent, "true_variation": var, "true_slot": slot, "pred_intent": pred,
         "pred_slot": pslot, "n_command_events": 0 if pred == S.NONE else 1, "cmd_start": 1.0,
         "cmd_end": 2.0, "latency_s": 0.5, "infer_ms": 50.0, "audio_ms": 1000.0, "is_synthetic": False}
    t.update(kw)
    return t


def test_score_end_to_end():
    trials = [
        _trial("TEMPERATURE", "Temperature 22 degrees", "22 degrees", "TEMPERATURE", "26"),
        _trial("TEMPERATURE", "Temperature 22 degrees", "22 degrees", "TEMPERATURE", "22 degrees"),
        _trial("TIME", "Time", "", "TIME", ""),
        _trial("OUT_OF_SCOPE", "", "", S.NONE, ""),
        _trial("OUT_OF_SCOPE", "", "", "PAUSE", ""),
    ]
    m = score(trials, [], {"model": "test"}, None, 0, 10, {"student": "x"})
    assert m["intent_level"]["accuracy"] == pytest.approx(4 / 5)
    assert m["command_level"]["accuracy"] == pytest.approx(3 / 5)
    assert m["intent_level"]["false_accept_rate"] == pytest.approx(0.5)
    assert m["slots"]["TEMPERATURE"]["mean_abs_error"] == pytest.approx(2.0)
    md = render_markdown(m)
    assert "93 commands" in md and "false accept" in md


# ---------------------------------------------------------------- audio

def test_trim_and_patch():
    sr = A.SR
    speech = (0.3 * np.sin(np.arange(sr // 2) / 5)).astype(np.float32)
    x = np.concatenate([np.zeros(sr), speech, np.zeros(sr)])
    y = A.trim(x)
    assert 0.5 <= len(y) / sr <= 0.75
    out, off = A.patch(y, y, gap_s=0.8)
    assert off["cmd_start"] == pytest.approx(off["wake_end"] + 0.8)
    assert len(out) / sr == pytest.approx(off["total"])
    assert A.dbfs(A.normalize(y, -20)) == pytest.approx(-20, abs=1.5)


# ---------------------------------------------------------------- links with the real agent

def _wait(pred, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _fake_ssh_link(tmp_path, monkeypatch, log):
    """SshLink driving the real pi_agent.py through a fake `ssh` that runs the last argument locally."""
    fake = tmp_path / "fakessh"
    fake.write_text('#!/bin/sh\nfor a; do last="$a"; done\nexec sh -c "$last"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(P, "REMOTE_AGENT", str(ROOT / "pi_agent.py"))
    link = P.SshLink(P.LineParser(), "pi@x", [], tmp_path / "cm", [str(log)], [], "pytest", interval=0.2,
                     python=sys.executable)
    link.ssh = [str(fake), "pi@x"]
    return link


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a POSIX shell script")


@posix_only
def test_ssh_link_with_local_agent(tmp_path, monkeypatch):
    log = tmp_path / "live.log"
    log.write_text("")
    link = _fake_ssh_link(tmp_path, monkeypatch, log)
    link.start()
    try:
        assert link.rtt is not None and abs(link.offset) < 0.5
        time.sleep(0.3)
        with open(log, "a") as f:
            f.write("intent=TIMER slot=30 seconds infer_ms=40\n")
        assert _wait(lambda: not link.events.empty())
        ev = link.drain()[0]
        assert (ev.intent, ev.slot, ev.infer_ms) == ("TIMER", "30 seconds", 40.0)
        assert abs(ev.t - time.time()) < 2
        assert _wait(lambda: len(link.samples) >= 2)
        assert link.specs.get("type") == "specs"
    finally:
        link.stop()


@posix_only
def test_ssh_link_reconnects(tmp_path, monkeypatch):
    """Unattended run: a dropped agent is restarted and events flow again."""
    import benchmark as B
    log = tmp_path / "live.log"
    log.write_text("")
    link = _fake_ssh_link(tmp_path, monkeypatch, log)
    link.start()
    try:
        link.proc.kill()
        link.proc.wait()
        assert not link.alive()
        real_sleep = time.sleep
        monkeypatch.setattr(B.time, "sleep", lambda s: real_sleep(min(s, 0.5)))
        assert B.ensure_link(link, max_wait_s=30)
        assert link.alive()
        time.sleep(0.5)
        with open(log, "a") as f:
            f.write("intent=PAUSE\n")
        assert _wait(lambda: not link.events.empty())
    finally:
        link.stop()


def test_http_link_with_local_agent(tmp_path):
    log = tmp_path / "live.log"
    log.write_text("")
    link = P.HttpLink(P.LineParser(), port=0)
    link.start()
    port = link.server.server_address[1]
    proc = subprocess.Popen([sys.executable, str(ROOT / "pi_agent.py"), "--post", f"http://127.0.0.1:{port}",
                             "--log", str(log), "--interval", "0.2"])
    try:
        assert _wait(lambda: bool(link.specs) and link.rtt is not None)
        with open(log, "a") as f:
            f.write(json.dumps({"intent": "LIGHT_OFF", "infer_ms": 12}) + "\n")
        assert _wait(lambda: not link.events.empty())
        assert link.drain()[0].intent == "LIGHT_OFF"
        assert link.alive()
    finally:
        proc.terminate()
        proc.wait(5)
        link.stop()


def test_sim_run(tmp_path):
    """Whole wizard in simulation mode, no sound, non-interactive."""
    r = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--mode", "sim", "--no-audio", "--yes",
                        "--fresh", "--size", "quick", "--limit", "6", "--gap-min", "1", "--gap-max", "1.1",
                        "--wake-word", "Watson", "--seed", "5", "--runs-dir", str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    run = next(tmp_path.iterdir())
    m = json.loads((run / "metrics.json").read_text())
    assert m["pipeline"]["trials"] + m["false_wake"]["n"] == 6        # wake + no-wake trials
    assert (run / "trials.csv").exists() and (run / "report.md").exists()


@pytest.mark.parametrize("noise_db", [-60, -40, -30])
def test_main_burst_noisy_wake_take(noise_db):
    """Laptop mic noise above -50 dBFS used to make the whole 2.5 s take count as speech."""
    sr = A.SR
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3 * sr).astype(np.float32) * 10 ** (noise_db / 20)
    t = np.arange(sr // 2) / sr
    x[int(1.2 * sr): int(1.2 * sr) + len(t)] += 0.3 * np.sin(2 * np.pi * 220 * t) * np.hanning(len(t))
    x[800:1100] += 0.5                                   # key click right after Enter
    y = A.main_burst(x)
    assert 0.35 <= len(y) / sr <= 0.9


# ---------------------------------------------------------------- false wake + seed

def _plan(tmp_path, seed, size="quick"):
    import argparse
    import benchmark as B
    B.YES = True
    args = argparse.Namespace(holdout=None, size=size, seed=seed, limit=None, wake_gap=0.5,
                              gap_min=10.0, gap_max=15.0)
    take = A.normalize(np.sin(np.arange(4000) / 7).astype(np.float32))
    return B.build_trials(args, {}, tmp_path, [take])


@pytest.mark.skipif(not (ROOT / ".cache" / "holdout.parquet").exists(), reason="holdout not downloaded")
def test_plan_false_wake_and_seed(tmp_path):
    a = _plan(tmp_path / "a", seed=7)
    oos = sum(t["true_intent"] == S.OOS for t in a)
    nw = [t for t in a if t["kind"] == "no_wake"]
    assert len(nw) == oos == 10
    assert all(t["true_intent"] != S.OOS for t in nw)
    assert len({t["true_intent"] for t in nw}) == len(nw)          # one per intent
    assert all(t["wake_take"] == 0 and t["cmd_start"] == pytest.approx(t["wake_start"]) for t in nw)
    assert len(a) == 103 + 10
    pos = [i for i, t in enumerate(a) if t["kind"] == "no_wake"]
    assert pos != list(range(len(a) - 10, len(a)))                  # mixed in, not appended
    b = _plan(tmp_path / "b", seed=7)
    c = _plan(tmp_path / "c", seed=8)
    key = lambda p: [(t["clip_idx"], t["kind"]) for t in p]
    assert key(a) == key(b) and key(a) != key(c)


def test_score_false_wake():
    trials = [
        _trial("TIME", "Time", "", "TIME", ""),
        _trial("OUT_OF_SCOPE", "", "", S.NONE, ""),
        _trial("PAUSE", "Pause", "", S.NONE, "", kind="no_wake"),
        _trial("STOP", "Stop", "", "STOP", "", kind="no_wake"),
    ]
    m = score(trials, [], {}, None, 0, 10, {"seed": 3})
    assert m["false_wake"]["n"] == 2 and m["false_wake"]["false_wakes"] == 1
    assert m["false_wake"]["false_wake_rate"] == pytest.approx(0.5)
    assert m["intent_level"]["n"] == 2                               # not mixed into 19/93 scores
    assert m["intent_level"]["accuracy"] == 1.0
    assert "false wake rate" in render_markdown(m)


class _NoTimingPi(P.SimLink):
    """Simulated Pi whose lines lack infer_ms / audio_ms."""

    def respond(self, intent, slot, t_end, wake=True):
        self.events.put(P.Event("command", t_end + 0.2, intent, slot, raw=f"intent={intent}"))


@pytest.mark.parametrize("pi_cls,waived", [(P.SimLink, False), (_NoTimingPi, True)])
def test_sound_check_requires_timing(tmp_path, monkeypatch, pi_cls, waived):
    import argparse
    import benchmark as B
    monkeypatch.setattr(B, "YES", True)
    link = pi_cls(P.LineParser(), accuracy=1.0)
    link.rng.random = lambda: 0.5                         # always wake, always right
    trials = [{"order": 0, "kind": "wake", "transcript": "Time", "true_intent": "TIME", "true_slot": "",
               "true_variation": "Time", "audio_file": "x.wav", "cmd_end": 0.1}]
    cfg = {"wake_word": "Watson"}
    B.sound_check(argparse.Namespace(no_audio=True), cfg, link, None, tmp_path, trials,
                  S.build_alias_table())
    assert cfg["timing_waived"] is waived


def test_breakdowns_overall_real_synthetic():
    trials = [
        _trial("TIME", "Time", "", "TIME", ""),
        _trial("TIME", "Time", "", "PAUSE", ""),
        _trial("TIMER", "Timer 1 minute", "1 minute", "TIMER", "60 seconds", is_synthetic=True),
        _trial("OUT_OF_SCOPE", "", "", "STOP", ""),
        _trial("PAUSE", "Pause", "", "PAUSE", "", kind="no_wake", is_synthetic=True),
    ]
    m = score(trials, [], {}, None, 0, 10, {})
    b = m["breakdowns"]
    assert list(b) == ["overall", "real voice", "synthetic voice"]
    assert b["overall"]["n"] == 4 and b["overall"]["accuracy"] == pytest.approx(2 / 4)
    assert b["real voice"]["n"] == 3 and b["real voice"]["accuracy"] == pytest.approx(1 / 3)
    assert b["real voice"]["false_accept_rate"] == 1.0
    assert b["synthetic voice"]["accuracy"] == 1.0 and b["synthetic voice"]["slot_exact_rate"] == 1.0
    assert b["synthetic voice"]["false_wakes"] == 1 and b["overall"]["false_wakes"] == 1
    md = render_markdown(m)
    assert "Overall vs real vs synthetic" in md and "synthetic voice" in md
    assert md.index("## At a glance") < md.index("# Detailed metrics") < md.index("## Classification")


# ---------------------------------------------------------------- 93-class output

@pytest.mark.parametrize("line,expect", [
    ('{"variation": "Set the temperature to 22 degrees", "infer_ms": 50, "audio_ms": 1500}',
     ("TEMPERATURE", "22 degrees", "Set the temperature to 22 degrees")),
    ('{"intent": "Wake me up at 9:00 PM", "infer_ms": 5, "audio_ms": 1000}',
     ("ALARM", "9:00 PM", "Wake me up at 9:00 PM")),
    ('{"variation_id": 0, "infer_ms": 5, "audio_ms": 1000}', ("PLAY_MUSIC", "", "Play music")),
    ("variation=change color to blue infer_ms=40 audio_ms=1500", ("COLOR", "Blue", "Change color to Blue")),
    ('{"variation": "OUT_OF_SCOPE", "infer_ms": 5, "audio_ms": 1000}', ("OUT_OF_SCOPE", "", "")),
    ("intent=TIMER slot=30 seconds infer_ms=40 audio_ms=1500", ("TIMER", "30 seconds", "")),
])
def test_93_class_output(line, expect):
    e = P.LineParser().parse(line, 0)
    i, s, known, v = S.resolve_prediction(e.intent, e.slot, e.variation, S.build_alias_table())
    assert (i, s, v) == expect and known
    assert e.infer_ms is not None and e.audio_ms is not None


def test_93_class_unknown_phrase_is_flagged():
    i, _, known, v = S.resolve_prediction("", "", "Change the lights to Blue", S.build_alias_table())
    assert i.startswith("OTHER:") and not known and v == ""


def test_score_exact_wording():
    t1 = _trial("TEMPERATURE", "Temperature 22 degrees", "22 degrees", "TEMPERATURE", "22 degrees",
                pred_variation="Set the temperature to 22 degrees")       # right command, other wording
    t2 = _trial("TIME", "Time", "", "TIME", "", pred_variation="Time")
    t3 = _trial("PAUSE", "Pause", "", "STOP", "", pred_variation="Stop")
    m = score([t1, t2, t3], [], {}, None, 0, 10, {})
    assert m["command_level"]["accuracy"] == pytest.approx(2 / 3)          # same rule as everyone
    assert m["exact_wording"]["accuracy"] == pytest.approx(1 / 3)          # wording must match too
    assert ("Pause", "Stop") in [c for c, _ in m["command_level"]["confusions"]]
    assert "93-class output" in render_markdown(m)
