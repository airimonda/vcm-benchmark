import json
import os
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
                        "--wake-word", "hey pi", "--seed", "5", "--runs-dir", str(tmp_path)],
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
    cfg = {"wake_word": "hey pi"}
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


# ---------------------------------------------------------------- mic check

def _speech(level, n=A.SR):
    return (level * np.sin(np.arange(n) / 5) * np.hanning(n)).astype(np.float32)


@pytest.mark.parametrize("level,verdict", [(0.3, "ok"), (0.03, "weak"), (0.004, "not heard"), (1.5, "clipping")])
def test_mic_levels(level, verdict):
    rng = np.random.default_rng(0)
    ambient = (0.002 * rng.standard_normal(2 * A.SR)).astype(np.float32)
    rec = np.concatenate([ambient, np.clip(_speech(level) + ambient[: A.SR], -1, 1)])
    assert A.mic_levels(ambient, rec)["verdict"] == verdict


def test_mic_check_flow(tmp_path, monkeypatch):
    import benchmark as B
    monkeypatch.setattr(B, "YES", True)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    A.save(tmp_path / "t.wav", _speech(0.3))
    rng = np.random.default_rng(1)

    class FakeLink(P.SshLink):
        def __init__(self):
            self.specs = {"audio_inputs": "card 2: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]"}

        def record(self, seconds, device="default"):
            noise = (0.002 * rng.standard_normal(int(seconds * A.SR))).astype(np.float32)
            if seconds > 2:
                noise[: A.SR] += _speech(0.3)
            return noise

    class FakePlayer:
        def play(self, x):
            return 0.0

    cfg = {}
    trials = [{"kind": "no_wake", "audio_file": "t.wav"}]
    B.mic_check(None, cfg, FakeLink(), FakePlayer(), tmp_path, trials)
    assert cfg["mic_check"]["verdict"] == "ok" and cfg["pi_mic"] == "default"
    assert B.pi_mics(FakeLink().specs)[1][0] == "plughw:2,0"


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


# ---------------------------------------------------------------- class-number order

def test_id_orders_from_manifest(tmp_path):
    orders = S.id_orders()
    assert len(orders["manifest"]) == 94 and orders["manifest"][93] == S.OOS
    assert S.lookup_id(0, orders["manifest"])[2] == "Play music"
    assert S.lookup_id(0, orders["alphabetical"])[2] == "Adjust brightness to 100 percent"
    assert S.lookup_id(93, orders["alphabetical"])[0] == S.OOS
    assert S.lookup_id(94, orders["manifest"]) is None
    f = tmp_path / "labels.txt"
    f.write_text("OUT_OF_SCOPE\nTime\nPlay music\n", encoding="utf-8")
    o = S.id_orders(label_file=str(f))["file"]
    assert S.lookup_id(0, o)[0] == S.OOS and S.lookup_id(2, o)[2] == "Play music"
    j = tmp_path / "labels.json"
    j.write_text('{"Time": 1, "Play music": 0}', encoding="utf-8")
    assert S.read_label_file(j) == ["Play music", "Time"]


def test_wrong_id_order_is_detected_and_rescored(tmp_path):
    alpha = S.id_orders()["alphabetical"]
    phrases = ["Time", "Weather", "Lights on", "Pause"]
    trials = []
    for p in phrases:                                  # the model numbers its classes alphabetically
        v = S.match_variation(p)
        t = _trial(v.intent, v.phrase, v.value, "", "")
        t.update(pred_variation_id=alpha.index(v.phrase), n_command_events=1, transcript=p)
        trials.append(t)
    m = score([dict(t) for t in trials], [], {}, None, 0, 10, {"id_order": "manifest"})
    ic = m["id_order_check"]
    assert ic["better"] == "alphabetical" and ic["matches"]["alphabetical"] == 4
    assert "--id-order alphabetical" in render_markdown(m)
    m2 = score([dict(t) for t in trials], [], {}, None, 0, 10, {"id_order": "alphabetical"})
    assert m2["command_level"]["accuracy"] == 1.0 and m2["id_order_check"]["better"] is None


def test_rescore_cli(tmp_path):
    import json as _json
    alpha = S.id_orders()["alphabetical"]
    v = S.match_variation("Time")
    t = _trial(v.intent, v.phrase, "", S.NONE, "")
    t.update(pred_variation_id=alpha.index("Time"), n_command_events=1, transcript="Time", play_t0=1.0)
    (tmp_path / "trials.jsonl").write_text(_json.dumps(t) + "\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(_json.dumps({"id_order": "manifest", "mode": "sim"}), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--rescore", str(tmp_path),
                        "--id-order", "alphabetical"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    m = _json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert m["intent_level"]["accuracy"] == 1.0
    assert _json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["id_order"] == "alphabetical"


@pytest.mark.parametrize("raw,target,opts", [
    ("pi@192.168.1.20", "pi@192.168.1.20", []),
    ("ssh pi@192.168.1.20", "pi@192.168.1.20", []),
    ("ssh pi@host -p 2222", "pi@host", ["-p", "2222"]),
    ("ssh -i ~/.ssh/k pi@raspberrypi.local", "pi@raspberrypi.local", ["-i", "~/.ssh/k"]),
    ("mypi", "mypi", []),
])
def test_parse_ssh_target(raw, target, opts):
    import benchmark as B
    assert B.parse_ssh_target(raw) == (target, opts)


# ---------------------------------------------------------------- finding the Pi (fake ssh, no network)

@posix_only
def test_find_pi_over_ssh(tmp_path, monkeypatch):
    import argparse
    import benchmark as B
    fake = tmp_path / "ssh"
    fake.write_text('#!/bin/sh\n'
                    'for a; do case "$a" in good@pi) echo VCM_DIR; exit 0;; '
                    'noperm@pi) echo "Permission denied (publickey,password)." >&2; exit 255;; esac; done\n'
                    'echo "ssh: Could not resolve hostname x: nodename nor servname provided" >&2; exit 255\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))                     # no real ~/.ssh/config
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "config").write_text("Host mypi\n  HostName 10.0.0.5\nHost work\n  HostName work.example\n")
    args = argparse.Namespace(host=None)
    cands = [c[0] for c in B.ssh_candidates(args, {"host": "noperm@pi"})]
    assert cands[0] == "noperm@pi" and "mypi" in cands and "work" not in cands
    found, problems = B.find_pi_over_ssh(args, {"host": "noperm@pi"})
    assert found is None
    assert "password" in problems["noperm@pi"] and "not found" in problems["mypi"]
    found, _ = B.find_pi_over_ssh(argparse.Namespace(host="ssh good@pi"), {})
    assert found == ("good@pi", [])


@posix_only
def test_pi_push_picks_reachable_address_and_stops(tmp_path):
    """Pi-sends mode: the agent tries each laptop address, uses the one that answers, and exits
    by itself when the laptop says the test is over."""
    logdir = tmp_path / "vcm_benchmark"
    logdir.mkdir()
    link = P.HttpLink(P.LineParser(), port=0)
    link.start()
    urls = f"http://127.0.0.1:9,http://127.0.0.1:{link.port}"          # first one is dead
    proc = subprocess.Popen([sys.executable, str(ROOT / "pi_agent.py"), "--post", urls,
                             "--log-dir", str(logdir), "--interval", "0.2"],
                            stderr=subprocess.PIPE, text=True)
    try:
        assert _wait(lambda: bool(link.specs), timeout=15)
        (logdir / "s1_20261002-120000.log").write_text("", encoding="utf-8")
        time.sleep(1.0)
        with open(logdir / "s1_20261002-120000.log", "a", encoding="utf-8") as f:
            f.write('{"intent": "TIME", "infer_ms": 5, "audio_ms": 1000}\n')
        assert _wait(lambda: not link.events.empty(), timeout=10)
        assert link.log_file.endswith("s1_20261002-120000.log")
        link.stop()
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------- any log format

@pytest.mark.parametrize("line,intent,slot,infer", [
    ('{"id": "t1", "heard": {"intent": "TEMPERATURE", "slots": {"temperature": "18 degrees"}}}',
     "TEMPERATURE", "18 degrees", None),                                   # nested JSON
    ("  TEMPERATURE  {'temperature': '18'}  ->  accepted=True  say='Setting 18'", "TEMPERATURE", "18", None),
    ("2026-10-02 12:00:01,123 INFO Heard: set_temperature_22 (p=0.97) took 85 ms", "TEMPERATURE", "22", 85.0),
    ("\x1b[32m[RESULT]\x1b[0m TIMER 30 seconds | conf 0.88 | model 42ms", "TIMER", "30 seconds", 42.0),
    ("Decoded command: Set the temperature to 22 degrees (52 ms)", "TEMPERATURE", "22 degrees", None),
    ("{'intent': 'LIGHT_ON', 'infer_ms': 7, 'audio_ms': 900}", "LIGHT_ON", "", 7.0),   # Python dict
    ('{"pred": 12, "infer_ms": 5, "audio_ms": 1000}', "LIGHT_OFF", "", 5.0),          # class number
    ('2026-10-02 12:00 {"intent": "PAUSE", "infer_ms": 4, "audio_ms": 900}', "PAUSE", "", 4.0),
])
def test_any_log_format(line, intent, slot, infer):
    e = P.LineParser().parse(line, 0)
    i, s, known, _ = S.resolve_prediction(e.intent, e.slot, e.variation, S.build_alias_table(),
                                          S.id_orders()["manifest"])
    assert (i, s, e.infer_ms) == (intent, slot, infer) and known


@pytest.mark.parametrize("line", [
    '{"mark": true, "turn_id": "t1", "correct": true, "intended_intent": null, "t": 1}',
    "loaded intents: TIMER ALARM PAUSE STOP",
    "cpu TEMP=55C load 0.3",
    "time: 12.3s elapsed, stop requested",
])
def test_non_command_lines_ignored(line):
    e = P.LineParser().parse(line, 0)
    assert e is None or e.kind != "command"


def test_multiline_json():
    p = P.LineParser()
    for part in ["{", '  "intent": "PAUSE",', '  "infer_ms": 9, "audio_ms": 800']:
        assert p.parse(part, 0) is None
    e = p.parse("}", 0)
    assert (e.intent, e.infer_ms, e.audio_ms) == ("PAUSE", 9.0, 800.0)


def test_auto_log_lock_uses_one_file():
    link = P.SimLink(P.LineParser())
    link.handle({"type": "log", "path": "/a/debug.log", "line": "cpu 55C", "auto": True}, 1.0)
    link.handle({"type": "log", "path": "/a/convo.jsonl", "line": '{"intent": "TIME"}', "auto": True}, 1.0)
    link.handle({"type": "log", "path": "/a/stdout.log", "line": "TIME -> ok", "auto": True}, 1.0)
    evs = link.drain()
    assert [e.intent for e in evs] == ["TIME"] and link.log_file == "/a/convo.jsonl"
    assert link.ignored == {"/a/stdout.log": 1}
    link.handle({"type": "log", "path": "/b/x.log", "line": "intent=PAUSE"}, 1.0)        # explicit source wins
    link.handle({"type": "log", "path": "/a/convo.jsonl", "line": '{"intent": "STOP"}', "auto": True}, 1.0)
    assert [e.intent for e in link.drain()] == ["PAUSE"]
@posix_only
def test_setup_ssh_key_and_known_hosts(tmp_path, monkeypatch):
    """Password login -> one-time key setup: creates a key if missing and installs it once;
    afterwards BatchMode (password-less) login works. Fake ssh / ssh-keygen, no network."""
    import argparse
    import benchmark as B
    bindir, pi_auth = tmp_path / "bin", tmp_path / "pi_authorized_keys"
    bindir.mkdir()
    (bindir / "ssh-keygen").write_text('#!/bin/sh\nwhile [ "$1" != "-f" ]; do shift; done\n'
                                       'echo PRIV > "$2"; echo "ssh-ed25519 AAAAtest vcm-benchmark" > "$2.pub"\n')
    (bindir / "ssh").write_text(
        '#!/bin/sh\n'
        f'AUTH="{pi_auth}"\n'
        'case "$*" in\n'
        '  *authorized_keys*) read k; grep -qxF "$k" "$AUTH" 2>/dev/null || echo "$k" >> "$AUTH"; exit 0;;\n'
        '  *BatchMode=yes*) [ -s "$AUTH" ] && exit 0; echo "Permission denied (publickey,password)." >&2; exit 255;;\n'
        'esac\nexit 0\n')
    for f in bindir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(B.Path, "home", classmethod(lambda cls: tmp_path))
    ok, why = B.setup_ssh_key("student@mypi.local", [])
    assert ok, why
    assert (tmp_path / ".ssh" / "id_ed25519.pub").exists()
    assert B.setup_ssh_key("student@mypi.local", [])[0]                  # again: no duplicate line
    assert pi_auth.read_text().count("AAAAtest") == 1
    (tmp_path / ".ssh" / "known_hosts").write_text(
        "mypi.local ssh-ed25519 AAAA\n[raspi4.lan]:2222 ssh-ed25519 AAAA\n|1|hashed= ssh-ed25519 AAAA\n"
        "work.example.com ssh-ed25519 AAAA\n")
    cands = B.ssh_candidates(argparse.Namespace(host=None), {})
    names = [c[0] for c in cands]
    assert any(n.endswith("@mypi.local") for n in names)
    assert any(n.endswith("@raspi4.lan") and o == ["-p", "2222"] for n, o, _ in cands)
    assert not any("work.example" in n for n in names)
