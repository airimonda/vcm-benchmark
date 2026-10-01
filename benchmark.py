#!/usr/bin/env python3
"""VCM live benchmark: play the holdout set to your Raspberry Pi and score what it does.

    python benchmark.py              # guided run (recommended)
    python benchmark.py --mode sim   # try the whole pipeline without a Pi
    python benchmark.py --resume runs/<run-id>

Steps:
  1. Connect to your Pi (SSH, Pi-pushes-to-laptop, or manual) and read its specs.
  2. Record your wake word on the laptop mic.
  3. Build "wake word + command" audio for every holdout clip.
  4. Sound check: two warm-up trials to set the volume and check that the Pi's
     output is understood.
  5. Play the trials 10-15 s apart; the Pi sleeps between turns.
  6. Score: 19-intent and 93-command metrics, slot-value distances, Pi
     resources and latency. Saved to runs/<run-id>/.
  7. Optionally delete the generated audio.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vcmbench import audio as A                                  # noqa: E402
from vcmbench import schema as S                                 # noqa: E402
from vcmbench.dataset import load_holdout                        # noqa: E402
from vcmbench.pi import (HttpLink, LineParser, ManualLink, SimLink, SshLink,  # noqa: E402
                         laptop_ips)
from vcmbench.report import score, write_outputs                 # noqa: E402

SAVED_SETTINGS = ROOT / "bench_settings.json"     # remembered answers (git-ignored)
CACHE = ROOT / ".cache"

# ------------------------------------------------------------------ console helpers

YES = False   # --yes: accept every default


def banner(n: int | str, title: str) -> None:
    print(f"\n{'=' * 70}\n STEP {n}: {title}\n{'=' * 70}")


def info(msg: str) -> None:
    print("  " + msg.replace("\n", "\n  "))


def ask(prompt: str, default: str = "") -> str:
    if YES:
        return default
    d = f" [{default}]" if default != "" else ""
    try:
        a = input(f"  > {prompt}{d}: ").strip()
    except EOFError:
        a = ""
    return a or default


def yesno(prompt: str, default: bool = True) -> bool:
    a = ask(prompt + (" (Y/n)" if default else " (y/N)"), "").lower()
    return default if not a else a.startswith("y")


def choose(prompt: str, options: list[tuple[str, str]], default: str) -> str:
    """options: [(key, description)]; returns the key."""
    for k, d in options:
        print(f"    [{k}] {d}")
    keys = {k for k, _ in options}
    while True:
        a = ask(prompt, default)
        if a in keys:
            return a
        print("    Please type one of: " + ", ".join(k for k, _ in options))


def wait_enter(msg: str = "Press Enter when ready") -> None:
    ask(msg)


# ------------------------------------------------------------------ step 1: Pi

def setup_pi(args, cfg: dict, run_dir: Path):
    banner(1, "Connect to your Raspberry Pi")
    info("Your Pi runs YOUR voice assistant as usual. This laptop plays the commands out loud,\n"
         "and reads what your assistant printed (its log) plus CPU/temperature/RAM from the Pi.\n"
         "Tip: run your assistant in a quiet/mock mode (no music playing, short replies),\n"
         "because the Pi's own speaker can drown out the next command.")
    mode = args.mode or cfg.get("mode")
    if not mode:
        mode = {"1": "ssh", "2": "http", "3": "manual", "4": "sim"}[choose("Connection", [
            ("1", "SSH from this laptop (Wi-Fi/LAN IP, raspberrypi.local, Tailscale, USB cable, ...)"),
            ("2", "Pi sends to this laptop (use when the laptop cannot reach the Pi)"),
            ("3", "Manual: no connection, I type what the Pi did after each command"),
            ("4", "Simulation: no Pi, just try the pipeline"),
        ], "1")]
    cfg["mode"] = mode

    if mode in ("ssh", "http"):
        info("\nWhere does your assistant print its result? Give the log file(s) on the Pi,\n"
             "e.g. ~/myassistant/logs/live.log, or a command whose output to follow,\n"
             "e.g. `journalctl --user -u myassistant -f -n 0 -o cat` (systemd service).\n"
             "Each recognised command must appear as ONE line, for example:\n"
             '   {"intent": "TIMER", "slot": "30 seconds", "infer_ms": 85}\n'
             "   intent=TIMER slot=30 seconds\n"
             "Optional: a line containing 'wake word' when the wake word fires (gives wake-rate).\n"
             "See README.md > 'What your Pi must print'.")
        logs = args.log or cfg.get("logs") or []
        log_cmds = args.log_cmd or cfg.get("log_cmds") or []
        if not logs and not log_cmds:
            a = ask("Log file on the Pi (or start with '!' for a command)", "")
            if a.startswith("!"):
                log_cmds = [a[1:].strip()]
            elif a:
                logs = [a]
        cfg["logs"], cfg["log_cmds"] = logs, log_cmds
        if not logs and not log_cmds:
            info("No log given: the Pi's answers cannot be read. Switching to manual entry,\n"
                 "but Pi metrics are still collected.")
            cfg["manual_answers"] = True
        cfg["proc"] = args.proc or cfg.get("proc") or ask(
            "Text that identifies your assistant's process (e.g. 'main.py' or 'python.*assistant';"
            " blank = whole Pi only)", "")

    parser = LineParser(cfg.get("command_regex"), cfg.get("wake_regex"))
    if mode == "ssh":
        target = args.host or cfg.get("host") or ask("SSH target (user@host or ~/.ssh/config alias)",
                                                     "pi@raspberrypi.local")
        cfg["host"] = target
        link = SshLink(parser, target, args.ssh_opt or cfg.get("ssh_opts") or [], CACHE / "ssh",
                       cfg["logs"], cfg["log_cmds"], cfg["proc"] or None,
                       python=cfg.get("pi_python", "python3"))
        info(f"Connecting to {target} (type the Pi password if asked) ...")
        ok, out = link.check()
        if not ok:
            info(f"SSH failed: {out}\n"
                 "Check: Pi on, same network (or Tailscale up), `ssh {target}` works in a terminal.\n"
                 "Tip: `ssh-copy-id {target}` once, so no password is needed.")
            raise SystemExit(1)
        link.upload_agent()
        specs = link.fetch_specs()
    elif mode == "http":
        port = int(args.port or cfg.get("port") or 8765)
        cfg["port"] = port
        link = HttpLink(parser, port)
        link.start()
        ips = laptop_ips()
        argv = " ".join(link.agent_args(cfg["logs"], [f"'{c}'" for c in cfg["log_cmds"]],
                                        f"'{cfg['proc']}'" if cfg["proc"] else None, 1.0))
        info("\nOn the Pi:\n"
             "  1. Copy pi_agent.py from this repo to the Pi (git clone, scp, USB stick, ...).\n"
             "  2. Run (keep it running during the test):\n")
        for ip in ips or ["<LAPTOP_IP>"]:
            info(f"     python3 pi_agent.py --post http://{ip}:{port} {argv}")
        info("\n  Use the laptop IP that the Pi can reach (Tailscale IP if you use Tailscale).\n"
             "  macOS may ask to allow incoming connections for Python: allow it.\n"
             "Waiting for the Pi ...")
        while not link.specs:
            time.sleep(0.5)
        specs = link.specs
        info("Pi connected.")
    elif mode == "manual":
        link = ManualLink(parser)
        specs = {"model": ask("Pi model (e.g. Raspberry Pi 4 Model B 4GB)", "Raspberry Pi"),
                 "ram_mb": ask("RAM in MB", ""), "os": ask("OS", ""), "hostname": "manual"}
        link.specs = specs
    else:
        link = SimLink(parser, seed=args.seed)
        specs = link.specs
    if mode == "manual" or cfg.get("manual_answers"):
        link.manual = True

    print()
    info("Pi specs:")
    for k in ("model", "hostname", "os", "kernel", "cpu_model", "cores", "max_freq_mhz", "ram_mb",
              "disk_free_gb", "python", "throttled"):
        if specs.get(k) not in (None, ""):
            info(f"  {k:14s} {specs[k]}")
    pk = {k: v for k, v in (specs.get("packages") or {}).items() if v}
    if pk:
        info("  packages       " + ", ".join(f"{k} {v}" for k, v in pk.items()))
    if specs.get("audio_inputs"):
        info("  microphones    " + " / ".join(l.strip() for l in str(specs["audio_inputs"]).splitlines()
                                             if l.startswith("card"))[:200])
    (run_dir / "pi_specs.json").write_text(json.dumps(specs, indent=2))
    return link


# ------------------------------------------------------------------ step 2: wake word

def record_wake(args, cfg: dict, run_dir: Path) -> tuple[str, list]:
    banner(2, "Record your wake word")
    word = args.wake_word or cfg.get("wake_word") or ask("Your wake word, as you say it", "Watson")
    cfg["wake_word"] = word
    wake_dir = run_dir / "wake"
    if args.wake_files:
        takes = [A.normalize(A.trim(A.load(Path(p)))) for p in args.wake_files]
        info(f"Using {len(takes)} wake word file(s) from --wake-files.")
        for i, t in enumerate(takes):
            A.save(wake_dir / f"wake_{i + 1}.wav", t)
        return word, takes
    if cfg["mode"] == "sim" and args.no_audio:
        return word, [A.normalize(0.1 * __import__("numpy").random.default_rng(0)
                                  .standard_normal(8000).astype("float32"))]

    info("The laptop records you saying the wake word a few times. Each take is reused for\n"
         "different commands. Speak at normal volume, about 30 cm from the laptop, quiet room.\n"
         "(macOS: allow microphone access for your terminal if asked.)")
    inputs = A.list_devices("input")
    for i, name, d in inputs:
        info(f"  [{i}] {name}{'  (default)' if d else ''}")
    dev = args.mic if args.mic is not None else int(ask("Microphone number", str(next(
        (i for i, _, d in inputs if d), inputs[0][0] if inputs else 0))))
    n = int(args.wake_takes or ask("How many takes", "3"))
    player = A.Player(cfg.get("speaker"))
    takes = []
    while len(takes) < n:
        wait_enter(f"Take {len(takes) + 1}/{n}: press Enter, then say '{word}' once")
        raw = A.record(2.5, dev)
        x = A.trim(raw)
        peak = float(abs(raw).max())
        dur = len(x) / A.SR
        if peak < 0.02 or dur < 0.15:
            info(f"Too quiet or nothing heard (peak {peak:.3f}). Move closer / speak louder. Again.")
            continue
        if peak > 0.99:
            info("Clipped (too loud). Move back a little. Again.")
            continue
        if dur > 2.2:
            info(f"That was {dur:.1f} s long: say only the wake word. Again.")
            continue
        x = A.normalize(x)
        info(f"Got {dur:.2f} s. Playing it back ...")
        player.play(x)
        if yesno("Keep this take?"):
            takes.append(x)
            A.save(wake_dir / f"wake_{len(takes)}.wav", x)
    return word, takes


# ------------------------------------------------------------------ step 3: build trials

def build_trials(args, cfg: dict, run_dir: Path, takes: list) -> list[dict]:
    banner(3, "Build the test audio (wake word + command)")
    clips = load_holdout(args.holdout or cfg.get("holdout"), CACHE)
    info(f"Holdout set: {len(clips)} clips ({sum(c.intent != S.OOS for c in clips)} commands over "
         f"{len({c.variation for c in clips if c.variation})} variations, "
         f"{sum(c.intent == S.OOS for c in clips)} out-of-scope).")
    size = args.size or cfg.get("size")
    if not size:
        est = lambda k: k * (4 + (args.gap_min + args.gap_max) / 2) / 60
        oos = sum(c.intent == S.OOS for c in clips)
        size = {"1": "full", "2": "quick"}.get(choose("Test size", [
            ("1", f"full: all {len(clips)} clips (~{est(len(clips)):.0f} min)"),
            ("2", f"quick: 1 clip per variation + all out-of-scope ({93 + oos} clips, ~{est(93 + oos):.0f} min)"),
        ], "1"))
    cfg["size"] = size
    rng = random.Random(args.seed)
    if size == "quick":
        seen, pick = set(), []
        for c in sorted(clips, key=lambda c: rng.random()):
            if c.intent == S.OOS or c.variation not in seen:
                pick.append(c)
                seen.add(c.variation)
        clips = pick
    order = clips[:]
    rng.shuffle(order)
    if args.limit:
        order = order[: args.limit]

    gap = float(args.wake_gap if args.wake_gap is not None else cfg.get("wake_gap") or ask(
        "Pause between wake word and command, seconds (time your Pi needs to start listening "
        "after the wake word, e.g. after its chime)", "0.8"))
    cfg["wake_gap"] = gap
    out_dir = run_dir / "audio"
    trials = []
    for k, c in enumerate(order):
        cmd = A.normalize(A.trim(c.audio))
        wi = k % len(takes)
        x, off = A.patch(takes[wi], cmd, gap)
        path = out_dir / f"trial_{k:03d}.wav"
        A.save(path, x)
        trials.append({"order": k, "clip_idx": c.idx, "transcript": c.transcript,
                       "true_intent": c.intent, "true_variation": c.variation, "true_slot": c.slot_value,
                       "speaker_id": c.speaker_id, "is_synthetic": c.is_synthetic,
                       "wake_take": wi + 1, "audio_file": str(path.relative_to(run_dir)), **off})
    (run_dir / "plan.json").write_text(json.dumps(trials, indent=2))
    info(f"Wrote {len(trials)} trial files to {out_dir}")
    return trials


# ------------------------------------------------------------------ step 4/5: play and collect

def choose_speaker(args, cfg: dict) -> A.Player | None:
    if args.no_audio:
        return None
    outs = A.list_devices("output")
    info("Speakers:")
    for i, name, d in outs:
        info(f"  [{i}] {name}{'  (default)' if d else ''}")
    dev = args.speaker if args.speaker is not None else int(ask("Speaker number", str(next(
        (i for i, _, d in outs if d), outs[0][0] if outs else 0))))
    cfg["speaker"] = dev
    return A.Player(dev, volume=float(cfg.get("volume", 1.0)))


def manual_answer(prompt_extra: str = "") -> tuple[str, str]:
    """Ask the student what the Pi did. Returns (raw intent, raw slot)."""
    if YES:
        return "", ""
    while True:
        a = ask("What did the Pi do? intent [slot] (number or name; Enter = nothing, o = out of scope, ? = list)"
                + prompt_extra, "")
        if a == "?":
            for i, name in enumerate(S.INTENTS, 1):
                print(f"      {i:2d} {name}")
            continue
        if not a:
            return S.NONE, ""
        if a.lower() in ("o", "oos"):
            return S.OOS, ""
        head, _, slot = a.partition(" ")
        if head.isdigit() and 1 <= int(head) <= len(S.INTENTS):
            head = S.INTENTS[int(head) - 1]
        return head, slot.strip()


def play_trial(t: dict, player: A.Player | None, link, run_dir: Path) -> float:
    if player:
        t0 = player.play(A.load(run_dir / t["audio_file"]))
    else:
        t0 = time.time()
        time.sleep(0.05)
    t["play_t0"] = t0
    t["cmd_end_abs"] = t0 + t["cmd_end"]
    if isinstance(link, SimLink):
        link.respond(t["true_intent"], t["true_slot"], t["cmd_end_abs"])
    return t0


def collect(t: dict, link, aliases: dict, wait_until: float, events_pool: list) -> None:
    """Wait for the Pi until `wait_until` (laptop time), then assign events to this trial."""
    if link.manual:
        raw_i, raw_s = manual_answer()
        evs = [] if raw_i == S.NONE else [S_Event(raw_i, raw_s)]
        while time.time() < wait_until:      # keep the 10-15 s spacing
            time.sleep(0.05)
        link.drain()
    else:
        while time.time() < wait_until:
            events_pool.extend(link.drain())
            time.sleep(0.05)
        events_pool.extend(link.drain())
        evs = [e for e in events_pool if t["play_t0"] - 0.2 <= e.t < wait_until]
        for e in evs:
            events_pool.remove(e)
        # events that came before this trial started are strays (late answers); drop them
        events_pool[:] = [e for e in events_pool if e.t >= wait_until]
    cmds = [e for e in evs if e.kind == "command"]
    t["n_command_events"] = len(cmds)
    t["wake_logged"] = any(e.kind == "wake" for e in evs)
    if cmds:
        e = cmds[0]
        intent, slot, known = S.normalize_prediction(e.intent, e.slot, aliases)
        t.update(pred_intent=intent, pred_slot=slot, pred_raw=e.raw or f"{e.intent} {e.slot}".strip(),
                 infer_ms=e.infer_ms, audio_ms=e.audio_ms,
                 latency_s=(e.t - t["cmd_end_abs"]) if not link.manual else None)
    else:
        t.update(pred_intent=S.NONE, pred_slot="", pred_raw="", infer_ms=None, audio_ms=None, latency_s=None)


class S_Event:  # tiny stand-in for manual answers
    kind = "command"

    def __init__(self, intent, slot):
        self.intent, self.slot, self.raw, self.t = intent, slot, f"(typed) {intent} {slot}".strip(), 0.0
        self.infer_ms = self.audio_ms = None


def sound_check(args, cfg: dict, link, player, run_dir: Path, trials: list[dict], aliases: dict) -> None:
    banner(4, "Sound check")
    info("Place the laptop speaker about 1 m from the Pi's microphone. Set the laptop volume\n"
         "to a normal speaking level. Start your assistant on the Pi now if it is not running.\n"
         "Two warm-up commands play now; they are NOT scored.")
    warm = [dict(t) for t in trials if t["true_intent"] in ("TIME", "TEMPERATURE")][:2] or [dict(trials[0])]
    while True:
        if not YES:
            wait_enter("Press Enter to play the warm-up commands")
        pool: list = []
        for t in warm:
            info(f"Playing: '{cfg['wake_word']}' ... '{t['transcript']}'  (expect {t['true_intent']} "
                 f"{t['true_slot']})".rstrip())
            n_lines = len(link.raw_lines)
            play_trial(t, player, link, run_dir)
            collect(t, link, aliases, time.time() + (2 if args.no_audio else 8), pool)
            if not link.manual:
                new = link.raw_lines[n_lines:]
                info(f"  Pi log lines received: {len(new)}")
                for _, line in new[-6:]:
                    info(f"    | {line[:120]}")
            got = f"{t['pred_intent']} {t['pred_slot']}".strip()
            info(f"  -> understood as: {got}" + ("  (OK)" if t["pred_intent"] == t["true_intent"] else ""))
            if t["pred_intent"].startswith("OTHER:"):
                info(f"  '{t['pred_intent'][6:]}' is not one of the 19 intents. Add an alias, e.g. in "
                     f"bench_settings.json: \"aliases\": {{\"{t['pred_intent'][6:]}\": \"TIMER\"}}")
        a = choose("Next", [("c", "continue to the test"), ("r", "replay (after changing the volume / position)"),
                            ("x", "my Pi printed a line but it was not understood: enter a regex"),
                            ("q", "quit")], "c")
        if a == "c":
            return
        if a == "q":
            raise SystemExit(0)
        if a == "x":
            info("Python regex with named groups: (?P<intent>...) required, (?P<slot>...),"
                 " (?P<infer_ms>...) optional.\n  Example: RESULT: (?P<intent>\\w+) \\((?P<slot>[^)]*)\\)")
            rx = ask("command regex", cfg.get("command_regex") or "")
            if rx:
                cfg["command_regex"] = rx
                link.parser = LineParser(rx, cfg.get("wake_regex"))


def run_trials(args, cfg: dict, link, player, run_dir: Path, trials: list[dict], aliases: dict) -> tuple[float, float]:
    banner(5, "Run the test")
    done_path = run_dir / "trials.jsonl"
    done = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            r = json.loads(line)
            done[r["order"]] = r
    todo = [t for t in trials if t["order"] not in done]
    avg = sum(t["total"] for t in todo) / max(len(todo), 1)
    est = len(todo) * (avg + (args.gap_min + args.gap_max) / 2) / 60
    info(f"{len(todo)} trials to play (~{est:.0f} min). {len(done)} already done.\n"
         "Keep the room quiet and do not move the laptop or the Pi.\n"
         "Press Ctrl+C at any time to pause (you can then resume, or stop and score what is done).")
    if not YES:
        wait_enter("Press Enter to start")
    rng = random.Random(args.seed + 1)
    t_start = cfg.get("t_start") or time.time()
    cfg["t_start"] = t_start
    pool: list = []
    with open(done_path, "a") as f:
        i = 0
        while i < len(todo):
            t = todo[i]
            try:
                if hasattr(link, "alive") and not link.alive():
                    info("!! Lost the connection to the Pi. Ctrl+C to pause, fix it, then resume.")
                play_trial(t, player, link, run_dir)
                gap = rng.uniform(args.gap_min, args.gap_max)
                collect(t, link, aliases, t["cmd_end_abs"] + gap, pool)
            except KeyboardInterrupt:
                a = choose("\n  Paused", [("r", "resume (replay this command)"), ("s", "skip this command"),
                                          ("q", "stop now and score what is done")], "r")
                if a == "q":
                    break
                if a == "s":
                    i += 1
                continue
            ok = "OK " if t["pred_intent"] == t["true_intent"] or (
                t["true_intent"] == S.OOS and t["pred_intent"] in (S.OOS, S.NONE)) else "-- "
            lat = f"{t['latency_s']:.2f}s" if t.get("latency_s") is not None else ""
            print(f"  {ok}[{len(done) + i + 1:3d}/{len(trials)}] '{t['transcript'][:38]:38s}' "
                  f"expected {t['true_intent']:15s} got {t['pred_intent']} {t['pred_slot']} {lat}")
            if isinstance(link, SshLink) and (len(done) + i) % 20 == 19:
                link.sync_clock(4)
            f.write(json.dumps(t, default=str) + "\n")
            f.flush()
            i += 1
    if isinstance(link, (SshLink, HttpLink, SimLink)):
        (run_dir / "pi_samples.jsonl").write_text("\n".join(json.dumps(s) for s in link.samples))
    return t_start, time.time()


# ------------------------------------------------------------------ step 6/7

def model_profile(args, cfg: dict, link) -> dict | None:
    path = args.model or cfg.get("model_path")
    if path is None and not YES:
        path = ask("Optional: path to your ONNX model for parameter/FLOP counts "
                   "(on this laptop, or 'pi:~/path/model.onnx'; blank = skip)", "")
    if not path:
        return None
    cfg["model_path"] = path
    try:
        from vcmbench.flops import format_si, onnx_profile
    except ImportError as e:
        info(f"FLOP count skipped: {e}")
        return None
    local = Path(path).expanduser()
    if path.startswith("pi:"):
        if not isinstance(link, SshLink):
            info("Fetching a model from the Pi needs the SSH connection; skipped.")
            return None
        local = CACHE / "model_from_pi.onnx"
        r = link.run(f"cat {path[3:]}", timeout=120)
        if r.returncode != 0:
            info("Could not read the model from the Pi: " + r.stderr.decode(errors="replace"))
            return None
        local.write_bytes(r.stdout)
    try:
        p = onnx_profile(local)
    except Exception as e:  # a broken/unsupported model must not lose the results
        info(f"FLOP count failed: {e}")
        return None
    p["flops_si"] = format_si(p["flops"]) + "FLOP"
    p["path"] = path
    return p


def cleanup(args, run_dir: Path) -> None:
    banner(7, "Clean up")
    targets = [p for p in (run_dir / "audio", run_dir / "wake") if p.exists()]
    if not targets:
        return
    size = sum(f.stat().st_size for p in targets for f in p.rglob("*") if f.is_file()) / 1e6
    info(f"Generated audio: {', '.join(str(p) for p in targets)} ({size:.0f} MB).\n"
         "Your results (report.md, metrics.json, trials.csv) are kept either way.\n"
         "Keep the audio if you want to re-run or resume this test.")
    if args.delete_audio or (not YES and yesno("Delete the generated audio (wake word recordings and trial files)?",
                                               False)):
        for p in targets:
            shutil.rmtree(p)
        info("Deleted.")
    if (CACHE / "holdout.parquet").exists() and not YES and yesno(
            "Also delete the downloaded holdout set (re-downloaded next time)?", False):
        (CACHE / "holdout.parquet").unlink()


def main() -> None:
    global YES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ssh", "http", "manual", "sim"], help="how to reach the Pi")
    ap.add_argument("--host", help="SSH target, e.g. pi@192.168.1.20 or an ~/.ssh/config alias")
    ap.add_argument("--ssh-opt", action="append", help="extra ssh option, e.g. --ssh-opt=-p2222")
    ap.add_argument("--port", type=int, help="laptop port for --mode http (default 8765)")
    ap.add_argument("--log", action="append", help="log file on the Pi with your assistant's output")
    ap.add_argument("--log-cmd", action="append", help="command on the Pi whose output to follow")
    ap.add_argument("--proc", help="regex for your assistant's process command line (CPU/RAM)")
    ap.add_argument("--student", help="your name or student number (goes in the report)")
    ap.add_argument("--wake-word", help="your wake word")
    ap.add_argument("--wake-files", nargs="+", help="use these wav files instead of recording")
    ap.add_argument("--wake-takes", type=int, help="number of wake word takes to record (default 3)")
    ap.add_argument("--wake-gap", type=float, help="seconds between wake word and command (default 0.8)")
    ap.add_argument("--mic", type=int, help="input device number")
    ap.add_argument("--speaker", type=int, help="output device number")
    ap.add_argument("--holdout", help="'hf' (default), a holdout .parquet, or a dataset/holdout folder")
    ap.add_argument("--size", choices=["full", "quick"], help="full = all 196 clips, quick = 103")
    ap.add_argument("--limit", type=int, help="play only the first N trials (debugging)")
    ap.add_argument("--gap-min", type=float, default=10.0, help="min seconds between commands")
    ap.add_argument("--gap-max", type=float, default=15.0, help="max seconds between commands")
    ap.add_argument("--model", help="ONNX model for FLOP/parameter counts (laptop path or pi:PATH)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--resume", help="resume / re-score an earlier run folder")
    ap.add_argument("--fresh", action="store_true", help="ignore saved settings from last time")
    ap.add_argument("--no-audio", action="store_true", help="sim mode only: do not play sound")
    ap.add_argument("--delete-audio", action="store_true", help="delete generated audio at the end")
    ap.add_argument("--yes", action="store_true", help="accept all defaults (non-interactive)")
    args = ap.parse_args()
    YES = args.yes
    if args.no_audio and args.mode != "sim":
        ap.error("--no-audio only works with --mode sim")

    print("\nVCM live benchmark\n"
          "The laptop plays the class holdout set ('wake word, command'), your Raspberry Pi\n"
          "listens, and this script scores what your assistant did. Answer the questions;\n"
          "press Enter to accept the [default].")

    if args.resume:
        run_dir = Path(args.resume)
        cfg = json.loads((run_dir / "config.json").read_text())
        trials = json.loads((run_dir / "plan.json").read_text())
        info(f"Resuming {run_dir}")
    else:
        cfg = {}
        if SAVED_SETTINGS.exists() and not args.fresh:
            saved = json.loads(SAVED_SETTINGS.read_text())
            if yesno(f"Reuse your settings from last time (mode {saved.get('mode')}, "
                     f"host {saved.get('host', '-')}, wake word {saved.get('wake_word')})?"):
                cfg = {k: v for k, v in saved.items() if k not in ("t_start",)}
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.runs_dir) / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["started"] = stamp
        trials = None
    cfg["student"] = args.student or cfg.get("student") or ask("Your name or student number", "")

    link = setup_pi(args, cfg, run_dir)
    aliases = S.build_alias_table(cfg.get("aliases"))
    try:
        if trials is None:
            word, takes = record_wake(args, cfg, run_dir)
            trials = build_trials(args, cfg, run_dir, takes)
        save_cfg(cfg, run_dir)
        player = choose_speaker(args, cfg)
        if isinstance(link, SshLink):
            link.start()          # agent streams metrics + log lines from now on
            time.sleep(1.0)
        if not (args.yes and cfg["mode"] == "sim"):
            sound_check(args, cfg, link, player, run_dir, trials, aliases)
            aliases = S.build_alias_table(cfg.get("aliases"))
        save_cfg(cfg, run_dir)
        t_start, t_end = run_trials(args, cfg, link, player, run_dir, trials, aliases)
    finally:
        link.stop()
    save_cfg(cfg, run_dir)

    banner(6, "Results")
    done = [json.loads(l) for l in (run_dir / "trials.jsonl").read_text().splitlines()]
    samples = link.samples
    if not samples and (run_dir / "pi_samples.jsonl").exists():
        samples = [json.loads(l) for l in (run_dir / "pi_samples.jsonl").read_text().splitlines() if l]
    prof = model_profile(args, cfg, link) if not args.yes else None
    meta = {k: cfg.get(k) for k in ("student", "started", "mode", "host", "wake_word", "wake_gap", "size")}
    meta["holdout"] = cfg.get("holdout") or "huggingface"
    meta["gap_s"] = [args.gap_min, args.gap_max]
    m = score(done, samples, link.specs, prof, t_start, t_end, meta)
    report = write_outputs(run_dir, m, done, samples)
    print()
    print(report.read_text())
    info(f"Saved: {report}, metrics.json, trials.csv, pi_metrics.csv in {run_dir}")
    cleanup(args, run_dir)


def save_cfg(cfg: dict, run_dir: Path) -> None:
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    if cfg.get("mode") == "sim":
        return                         # a dry run must not overwrite real settings
    keep = {k: v for k, v in cfg.items() if k not in ("started", "t_start", "size")}
    SAVED_SETTINGS.write_text(json.dumps(keep, indent=2))


if __name__ == "__main__":
    main()
