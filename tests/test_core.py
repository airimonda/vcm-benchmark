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


def test_ssh_link_with_local_agent(tmp_path, monkeypatch):
    """SshLink driving pi_agent.py through `sh -c` instead of ssh."""
    log = tmp_path / "live.log"
    log.write_text("")
    monkeypatch.setattr(P, "REMOTE_AGENT", str(ROOT / "pi_agent.py"))
    link = P.SshLink(P.LineParser(), "x", [], tmp_path / "cm", [str(log)], [], "pytest", interval=0.2,
                     python=sys.executable)
    link.ssh = ["sh", "-c"]
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


@pytest.mark.skipif(not (ROOT / ".cache" / "holdout.parquet").exists(), reason="holdout not downloaded")
def test_sim_run(tmp_path):
    """Whole wizard in simulation mode, no sound, non-interactive."""
    r = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--mode", "sim", "--no-audio", "--yes",
                        "--fresh", "--size", "quick", "--limit", "6", "--gap-min", "1", "--gap-max", "1.1",
                        "--wake-word", "Watson", "--runs-dir", str(tmp_path), "--holdout",
                        str(ROOT / ".cache" / "holdout.parquet")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    run = next(tmp_path.iterdir())
    m = json.loads((run / "metrics.json").read_text())
    assert m["pipeline"]["trials"] == 6
    assert (run / "trials.csv").exists() and (run / "report.md").exists()
