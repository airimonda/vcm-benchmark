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
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vcmbench import audio as A                                  # noqa: E402
from vcmbench import schema as S                                 # noqa: E402
from vcmbench.dataset import load_holdout                        # noqa: E402
from vcmbench.pi import (HttpLink, LineParser, ManualLink, SimLink, SshLink,  # noqa: E402
                         laptop_ips)
from vcmbench.report import score, write_outputs                 # noqa: E402

IS_WINDOWS = platform.system() == "Windows"
for _stream in (sys.stdout, sys.stderr):        # Pi log lines may hold characters cp1252 can't print
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

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
        if not shutil.which("ssh"):
            info("No `ssh` command found. Windows 10/11: Settings > System > Optional features >\n"
                 "Add a feature > 'OpenSSH Client', then open a new terminal. Or use connection 2 or 3.")
            raise SystemExit(1)
        link = SshLink(parser, target, args.ssh_opt or cfg.get("ssh_opts") or [], CACHE / "ssh",
                       cfg["logs"], cfg["log_cmds"], cfg["proc"] or None,
                       python=cfg.get("pi_python", "python3"))
        info(f"Connecting to {target} (type the Pi password if asked) ...")
        ok, out = link.check()
        if not ok:
            info(f"SSH failed: {out}\n"
                 "Check: Pi on, same network (or Tailscale up), `ssh {target}` works in a terminal.\n"
                 + key_tip(target))
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
        link = SimLink(parser, seed=args.seed or 0)
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
    (run_dir / "pi_specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    return link


# ------------------------------------------------------------------ step 2: wake word

def record_wake(args, cfg: dict, run_dir: Path) -> tuple[str, list]:
    banner(2, "Record your wake word")
    word = args.wake_word or cfg.get("wake_word") or ask("Your wake word, as you say it", "Watson")
    cfg["wake_word"] = word
    wake_dir = run_dir / "wake"
    if args.wake_files:
        takes = [A.normalize(A.main_burst(A.load(Path(p)))) for p in args.wake_files]
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
        raw = A.record(3.0, dev)
        x = A.main_burst(raw)
        peak = float(abs(raw).max())
        dur = len(x) / A.SR
        if peak < 0.02 or dur < 0.15:
            info(f"Too quiet or nothing heard (peak {peak:.3f}). Move closer / speak louder. Again.")
            continue
        if peak > 0.99:
            info("Clipped (too loud). Move back a little. Again.")
            continue
        if dur > 1.8:
            info(f"Heard {dur:.1f} s of sound: say only the wake word, and keep the room quiet "
                 f"(noise level {A.dbfs(raw):.0f} dBFS). Again.")
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
    clips = all_clips = load_holdout(args.holdout or cfg.get("holdout"), CACHE)
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
    seed = choose_seed(args, cfg)
    rng = random.Random(seed)
    if size == "quick":
        seen, pick = set(), []
        for c in sorted(clips, key=lambda c: rng.random()):
            if c.intent == S.OOS or c.variation not in seen:
                pick.append(c)
                seen.add(c.variation)
        clips = pick
    # False-wake check: in-scope commands played WITHOUT the wake word, as many as the
    # out-of-scope clips. The Pi should stay asleep. One clip per intent where possible.
    n_fw = sum(c.intent == S.OOS for c in clips)
    pool = [c for c in all_clips if c.intent != S.OOS]
    rng.shuffle(pool)
    no_wake, used = [], set()
    for c in pool + pool:                       # second pass fills up if intents run out
        if len(no_wake) == n_fw:
            break
        if c.intent not in used or len(used) == len(S.INTENTS):
            if not any(c is d for d in no_wake):
                no_wake.append(c)
                used.add(c.intent)
    order = [(c, "wake") for c in clips] + [(c, "no_wake") for c in no_wake]
    rng.shuffle(order)                          # wake and no-wake trials mixed at random
    if args.limit:
        order = order[: args.limit]

    gap = float(args.wake_gap if args.wake_gap is not None else cfg.get("wake_gap") or ask(
        "Pause between wake word and command, seconds (time your Pi needs to start listening "
        "after the wake word, e.g. after its chime)", "0.8"))
    cfg["wake_gap"] = gap
    out_dir = run_dir / "audio"
    trials = []
    wake_n = 0
    for k, (c, kind) in enumerate(order):
        cmd = A.normalize(A.trim(c.audio))
        if kind == "wake":
            wi = wake_n % len(takes)
            wake_n += 1
            x, off = A.patch(takes[wi], cmd, gap)
        else:
            wi = -1
            x, off = A.patch(A.np.zeros(0, dtype="float32"), cmd, 0.0)
        path = out_dir / f"trial_{k:03d}.wav"
        A.save(path, x)
        trials.append({"order": k, "kind": kind, "clip_idx": c.idx, "transcript": c.transcript,
                       "true_intent": c.intent, "true_variation": c.variation, "true_slot": c.slot_value,
                       "speaker_id": c.speaker_id, "is_synthetic": c.is_synthetic, "accent_group": c.accent_group,
                       "wake_take": wi + 1, "audio_file": path.relative_to(run_dir).as_posix(), **off})
    (run_dir / "plan.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    nw = sum(t["kind"] == "no_wake" for t in trials)
    info(f"Wrote {len(trials)} trial files to {out_dir}: {len(trials) - nw} with the wake word, "
         f"{nw} without it (false-wake check), shuffled with seed {seed}.")
    return trials


def choose_seed(args, cfg: dict) -> int:
    """The shuffle seed decides the playing order and which clips are used for the
    false-wake check. Same seed + same test size = same test."""
    if args.seed is not None:
        seed = args.seed
    else:
        suggestion = random.randint(1, 99999)
        info("Shuffle seed: decides the order of the commands and which ones play without the\n"
             "wake word. Keep the suggested random number, or type your own (e.g. the class's\n"
             "agreed seed, or an earlier run's seed to repeat it exactly).")
        while True:
            a = ask("Shuffle seed", str(suggestion))
            if a.lstrip("-").isdigit():
                seed = int(a)
                break
            info("Please type a whole number.")
    cfg["seed"] = seed
    return seed


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
        link.respond(t["true_intent"], t["true_slot"], t["cmd_end_abs"], t.get("kind", "wake") == "wake")
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


def pi_mics(specs: dict) -> list[tuple[str, str]]:
    """[(ALSA device, description)] from the Pi's `arecord -l` output in its specs."""
    out = [("default", "the Pi's default input (shared with your assistant if it uses PulseAudio/PipeWire)")]
    for line in str(specs.get("audio_inputs") or "").splitlines():
        m = re.match(r"card (\d+): .*?\[(.*?)\].*?device (\d+)", line)
        if m:
            out.append((f"plughw:{m.group(1)},{m.group(3)}", m.group(2)))
    return out


def mic_check(args, cfg: dict, link, player, run_dir: Path, trials: list[dict]) -> None:
    """Record on the Pi while the laptop plays a command: does the Pi's mic actually hear it?"""
    if not isinstance(link, SshLink) or player is None:
        return
    banner("4a", "Mic check: does the Pi hear the laptop?")
    info("The Pi records 2 s of room noise, then records again while the laptop plays a command\n"
         "(without the wake word, so your assistant should not react). Keep the room quiet.")
    clip_t = next((t for t in trials if t.get("kind") == "no_wake"), trials[0])
    clip = A.load(run_dir / clip_t["audio_file"])
    device = cfg.get("pi_mic", "default")
    while True:
        if not YES:
            wait_enter(f"Press Enter to run the mic check (Pi input: {device})")
        try:
            ambient = link.record(2, device)
            box: dict = {}

            def rec():
                try:
                    box["x"] = link.record(len(clip) / A.SR + 4, device)
                except Exception as e:  # reported below
                    box["err"] = e

            th = threading.Thread(target=rec)
            th.start()
            time.sleep(3.0 if IS_WINDOWS else 2.0)     # let arecord start on the Pi
            player.play(clip)
            th.join()
            if "err" in box:
                raise box["err"]
            r = A.mic_levels(ambient, box["x"])
        except Exception as e:
            msg = str(e)
            info(f"!! Recording on the Pi failed: {msg[:300]}")
            if "busy" in msg.lower():
                info("   Your assistant is holding the microphone. Choose 'default' (shared through\n"
                     "   PulseAudio/PipeWire), or stop your assistant for the check and start it again after.")
            elif "not found" in msg.lower() or "No such file" in msg:
                info("   arecord is missing on the Pi: sudo apt install alsa-utils")
            r = None
        if r:
            cfg["mic_check"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
            info(f"Room noise {r['noise_dbfs']:.0f} dBFS, laptop speech {r['speech_dbfs']:.0f} dBFS, "
                 f"signal-to-noise {r['snr_db']:.0f} dB, peak {r['peak']:.2f}")
            info({"ok": "OK: the Pi hears the laptop clearly.",
                  "weak": "WEAK: the Pi hears it, but not by much. Turn the laptop volume up a little or move\n"
                          "it closer (aim for 20 dB or more).",
                  "not heard": "NOT HEARD: the Pi barely hears the laptop. Check the Pi mic (choose another\n"
                               "input below), turn the laptop volume up, or move it closer.",
                  "clipping": "CLIPPING: too loud for the Pi mic, the sound distorts. Turn the laptop volume\n"
                              "down or move it further away."}[r["verdict"]])
        ok = bool(r) and r["verdict"] in ("ok", "weak")
        opts = ([("c", "continue")] if ok else []) + [("r", "repeat the check"), ("d", "choose another Pi microphone")]
        opts += [("s", "skip the mic check")] if not ok else []
        a = choose("Next", opts, "c" if ok else ("s" if YES else "r"))
        if a in ("c", "s"):
            cfg["pi_mic"] = device
            return
        if a == "d":
            mics = pi_mics(link.specs)
            for i, (dev, desc) in enumerate(mics, 1):
                info(f"  [{i}] {dev:14s} {desc}")
            pick = ask("Number, or an ALSA device name", "1")
            device = mics[int(pick) - 1][0] if pick.isdigit() and 1 <= int(pick) <= len(mics) else pick


def sound_check(args, cfg: dict, link, player, run_dir: Path, trials: list[dict], aliases: dict) -> None:
    banner(4, "Sound check")
    info("Place the laptop speaker about 1 m from the Pi's microphone. Set the laptop volume\n"
         "to a normal speaking level. Start your assistant on the Pi now if it is not running.\n"
         "Two warm-up commands play now; they are NOT scored.")
    wake_trials = [t for t in trials if t.get("kind", "wake") == "wake"]
    warm = [dict(t) for t in wake_trials if t["true_intent"] in ("TIME", "TEMPERATURE")][:2] or [dict(wake_trials[0])]
    while True:
        if not YES:
            wait_enter("Press Enter to play the warm-up commands")
        pool: list = []
        timing_ok = True
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
            if t["n_command_events"] and not link.manual:
                missing = [k for k in ("infer_ms", "audio_ms") if t.get(k) is None]
                if missing:
                    timing_ok = False
                    info(f"  !! The line has no {' / '.join(missing)}. Both are REQUIRED on every command line:\n"
                         "     infer_ms = time your model took for this command (features + model), in ms\n"
                         "     audio_ms = length of the audio it processed, in ms\n"
                         '     e.g. {"intent": "TIME", "slot": "", "infer_ms": 85, "audio_ms": 1500}\n'
                         "     See README.md > 'What your Pi must print'. Add them and replay.")
                else:
                    info(f"  timing: infer_ms {t['infer_ms']:.0f}, audio_ms {t['audio_ms']:.0f} "
                         f"(real-time factor {t['infer_ms'] / t['audio_ms']:.3f})" if t["audio_ms"] else
                         "  !! audio_ms is 0")
            if t["pred_intent"].startswith("OTHER:"):
                info(f"  '{t['pred_intent'][6:]}' is not one of the 19 intents. Add an alias, e.g. in "
                     f"bench_settings.json: \"aliases\": {{\"{t['pred_intent'][6:]}\": \"TIMER\"}}")
        options = [("c", "continue to the test")] if timing_ok else []
        options += [("r", "replay (after changing the volume / position / your Pi's output)"),
                    ("x", "my Pi printed a line but it was not understood: enter a regex"),
                    ("q", "quit")]
        if not timing_ok:
            options.append(("i", "continue WITHOUT infer_ms/audio_ms (report marks timing as missing)"))
        a = choose("Next", options, "c" if timing_ok else ("i" if YES else "r"))
        if a in ("c", "i"):
            cfg["timing_waived"] = a == "i"
            return
        if a == "q":
            raise SystemExit(0)
        if a == "x":
            info("Python regex with named groups: (?P<intent>...), (?P<infer_ms>...), (?P<audio_ms>...) "
                 "required, (?P<slot>...) for slots.\n"
                 "  Example: RESULT: (?P<intent>\\w+) \\((?P<slot>[^)]*)\\) (?P<infer_ms>[0-9.]+)ms/(?P<audio_ms>[0-9.]+)ms")
            rx = ask("command regex", cfg.get("command_regex") or "")
            if rx:
                cfg["command_regex"] = rx
                link.parser = LineParser(rx, cfg.get("wake_regex"))


def run_trials(args, cfg: dict, link, player, run_dir: Path, trials: list[dict], aliases: dict) -> tuple[float, float]:
    banner(5, "Run the test")
    done_path = run_dir / "trials.jsonl"
    done = {}
    if done_path.exists():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["order"]] = r
    todo = [t for t in trials if t["order"] not in done]
    avg = sum(t["total"] for t in todo) / max(len(todo), 1)
    est = len(todo) * (avg + (args.gap_min + args.gap_max) / 2) / 60
    finish = dt.datetime.now() + dt.timedelta(minutes=est)
    info(f"{len(todo)} trials to play (~{est:.0f} min, done around {finish:%H:%M}). {len(done)} already done.\n"
         "Running unattended now. Ctrl+C pauses (resume / skip / stop and score).")
    rng = random.Random(int(cfg.get("seed") or 0) + 1)
    t_start = cfg.get("t_start") or time.time()
    cfg["t_start"] = t_start
    pool: list = []
    with open(done_path, "a", encoding="utf-8") as f:
        i = 0
        while i < len(todo):
            t = todo[i]
            try:
                if not ensure_link(link):
                    info("!! The Pi did not come back. Stopping; results so far are scored.\n"
                         f"   Fix the connection, then continue with: python benchmark.py --resume {run_dir}")
                    break
                if not play_with_retry(t, player, link, run_dir):
                    info("!! The speaker failed repeatedly. Stopping; results so far are scored.\n"
                         f"   Fix the audio output, then continue with: python benchmark.py --resume {run_dir}")
                    break
                gap = rng.uniform(args.gap_min, args.gap_max)
                collect(t, link, aliases, t["cmd_end_abs"] + gap, pool)
                if not link.manual and not link.alive():
                    info(f"!! Connection dropped during '{t['transcript']}'; this command will be replayed.")
                    continue
            except KeyboardInterrupt:
                a = choose("\n  Paused", [("r", "resume (replay this command)"), ("s", "skip this command"),
                                          ("q", "stop now and score what is done")], "r")
                if a == "q":
                    break
                if a == "s":
                    i += 1
                continue
            lat = f"{t['latency_s']:.2f}s" if t.get("latency_s") is not None else ""
            if t.get("kind") == "no_wake":
                ok = "OK " if t["n_command_events"] == 0 else "-- "
                expected = "(no wake word) nothing"
                got = "nothing" if t["n_command_events"] == 0 else f"FALSE WAKE: {t['pred_intent']}"
            else:
                ok = "OK " if t["pred_intent"] == t["true_intent"] or (
                    t["true_intent"] == S.OOS and t["pred_intent"] in (S.OOS, S.NONE)) else "-- "
                expected, got = t["true_intent"], f"{t['pred_intent']} {t['pred_slot']} {lat}"
            print(f"  {ok}[{len(done) + i + 1:3d}/{len(trials)}] '{t['transcript'][:38]:38s}' "
                  f"expected {expected:22s} got {got}")
            if isinstance(link, SshLink) and (len(done) + i) % 20 == 19:
                link.sync_clock(4)
            f.write(json.dumps(t, default=str) + "\n")
            f.flush()
            i += 1
    if isinstance(link, (SshLink, HttpLink, SimLink)):
        (run_dir / "pi_samples.jsonl").write_text("\n".join(json.dumps(s) for s in link.samples), encoding="utf-8")
    return t_start, time.time()


def ensure_link(link, max_wait_s: float = 600) -> bool:
    """Unattended recovery: if the Pi link is down, reconnect / wait up to max_wait_s."""
    if link.manual or link.alive():
        return True
    info("!! Lost the connection to the Pi. Reconnecting ...")
    end = time.time() + max_wait_s
    delay = 5.0
    while time.time() < end:
        if isinstance(link, SshLink):
            try:
                link.restart()
            except Exception as e:                       # keep trying until the deadline
                info(f"   reconnect failed: {e}")
        time.sleep(delay)
        if link.alive():
            info("   Reconnected.")
            return True
        delay = min(delay * 2, 60)
    return False


def play_with_retry(t: dict, player, link, run_dir: Path, tries: int = 3) -> bool:
    for _ in range(tries):
        try:
            play_trial(t, player, link, run_dir)
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:                           # PortAudio errors, device unplugged, ...
            info(f"!! Audio error ({e}); retrying in 5 s ...")
            time.sleep(5)
            if player:
                try:
                    player.__init__(player.device, player.volume)
                except Exception:
                    player.__init__(None, player.volume)   # fall back to the default speaker
    return False


@contextmanager
def keep_awake():
    """Stop the laptop from sleeping during the unattended run (best effort)."""
    proc, prev = None, None
    system = platform.system()
    try:
        if system == "Darwin":
            proc = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        elif system == "Linux" and shutil.which("systemd-inhibit"):
            proc = subprocess.Popen(["systemd-inhibit", "--what=idle:sleep", "--who=vcm-benchmark",
                                     "--why=benchmark running", "sleep", "infinity"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Windows":
            import ctypes
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            prev = ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    except Exception as e:
        info(f"(could not keep the laptop awake: {e}; turn off sleep yourself)")
    try:
        yield
    finally:
        if proc:
            proc.terminate()
        if system == "Windows" and prev is not None:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def notify(title: str, msg: str) -> None:
    """Bell + desktop notification when the unattended run ends (best effort)."""
    print("\a", end="", flush=True)
    try:
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{title}"'],
                           timeout=5, capture_output=True)
        elif IS_WINDOWS:
            ps = ("Add-Type -AssemblyName System.Windows.Forms; $n = New-Object System.Windows.Forms.NotifyIcon; "
                  "$n.Icon = [System.Drawing.SystemIcons]::Information; $n.Visible = $true; "
                  f"$n.ShowBalloonTip(10000, '{title}', '{msg}', 'Info'); Start-Sleep 8; $n.Dispose()")
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif shutil.which("notify-send"):
            subprocess.run(["notify-send", title, msg], timeout=5, capture_output=True)
    except Exception:
        pass


def key_tip(target: str | None) -> str:
    """How to set up password-less SSH, per laptop OS."""
    t = target or "user@pi"
    if IS_WINDOWS:
        return ("      Set up an SSH key once (PowerShell):\n"
                "        ssh-keygen -t ed25519      (press Enter at every question)\n"
                f'        type $env:USERPROFILE\\.ssh\\id_ed25519.pub | ssh {t} '
                '"mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"')
    return (f"      Set up an SSH key once: `ssh-keygen -t ed25519` (if you have none), "
            f"then `ssh-copy-id {t}`.")


def approve(args, cfg: dict, link, trials: list[dict], run_dir: Path) -> dict | None:
    """Ask everything that is left, show the plan, and get one go-ahead.
    After this nothing is asked until the report is printed."""
    banner("5a", "Approve the unattended run")
    prof = model_profile(args, cfg, link)
    if prof:
        info(f"Model: {prof['params']:,} parameters, {prof['flops_si']} per inference.")
    if args.delete_audio:
        cfg["delete_audio"] = True
    else:
        cfg["delete_audio"] = yesno("When the test ends, delete the generated audio "
                                    "(your wake word recordings and the trial files)?", False)
    path = run_dir / "trials.jsonl"
    n = len(trials) - (len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0)
    avg = sum(t["total"] for t in trials) / max(len(trials), 1)
    est = n * (avg + (args.gap_min + args.gap_max) / 2) / 60
    finish = dt.datetime.now() + dt.timedelta(minutes=est)
    info(f"\nPlan:\n"
         f"  Pi            {link.specs.get('model') or '?'} ({cfg['mode']}"
         f"{', ' + cfg['host'] if cfg.get('host') else ''})\n"
         f"  wake word     '{cfg['wake_word']}', pause {cfg['wake_gap']} s before the command\n"
         f"  shuffle seed  {cfg.get('seed')}  ({sum(t.get('kind') == 'no_wake' for t in trials)} commands "
         f"play without the wake word)\n"
         f"  commands      {n} ({cfg.get('size')}), {args.gap_min:g}-{args.gap_max:g} s apart\n"
         f"  duration      ~{est:.0f} min, done around {finish:%H:%M}\n"
         f"  at the end    report saved in {run_dir}; generated audio "
         f"{'DELETED' if cfg['delete_audio'] else 'kept'}\n")
    if link.manual:
        info("Manual mode: you type what the Pi did after every command, so this run is NOT unattended.")
    else:
        info("After you approve, the run needs no input. The laptop is kept awake; keep it\n"
             "plugged in, lid open, volume unchanged, and the room quiet. If the Pi connection\n"
             "drops, it reconnects and replays that command; if the speaker fails, it retries.")
        if isinstance(link, SshLink) and not link.batch_ok():
            info("NOTE: your SSH login asks for a password, so an automatic reconnect cannot log in.\n"
                 + key_tip(cfg.get("host")))
    if not yesno("Start now?", True):
        info(f"Not started. Continue later with: python benchmark.py --resume {run_dir}")
        raise SystemExit(0)
    return prof


# ------------------------------------------------------------------ step 6/7

def model_profile(args, cfg: dict, link) -> dict | None:
    path = args.model or cfg.get("model_path")
    if path is None and not YES and not link.manual:
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


def cleanup(run_dir: Path, delete: bool) -> None:
    banner(7, "Clean up")
    targets = [p for p in (run_dir / "audio", run_dir / "wake") if p.exists()]
    if not targets:
        return
    size = sum(f.stat().st_size for p in targets for f in p.rglob("*") if f.is_file()) / 1e6
    if delete:
        for p in targets:
            shutil.rmtree(p)
        info(f"Deleted the generated audio ({size:.0f} MB), as chosen before the run.")
    else:
        quoted = ", ".join('"%s"' % p for p in targets)
        cmd = (f"Remove-Item -Recurse {quoted}" if IS_WINDOWS
               else "rm -r " + " ".join(str(p) for p in targets))
        info(f"Kept the generated audio ({size:.0f} MB). Delete it any time with:\n  {cmd}")


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
    ap.add_argument("--seed", type=int, help="shuffle seed (order + false-wake clips); asked if not given")
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
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        trials = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        info(f"Resuming {run_dir}")
    else:
        cfg = {}
        if SAVED_SETTINGS.exists() and not args.fresh:
            saved = json.loads(SAVED_SETTINGS.read_text(encoding="utf-8"))
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
            mic_check(args, cfg, link, player, run_dir, trials)
            sound_check(args, cfg, link, player, run_dir, trials, aliases)
            aliases = S.build_alias_table(cfg.get("aliases"))
        prof = approve(args, cfg, link, trials, run_dir)
        save_cfg(cfg, run_dir)
        with keep_awake():
            t_start, t_end = run_trials(args, cfg, link, player, run_dir, trials, aliases)
    finally:
        link.stop()
    save_cfg(cfg, run_dir)

    banner(6, "Results")
    done = [json.loads(l) for l in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = link.samples
    if not samples and (run_dir / "pi_samples.jsonl").exists():
        samples = [json.loads(l) for l in (run_dir / "pi_samples.jsonl").read_text(encoding="utf-8").splitlines() if l]
    meta = {k: cfg.get(k) for k in ("student", "started", "mode", "host", "wake_word", "wake_gap", "size",
                                     "seed", "pi_mic", "mic_check")}
    meta["holdout"] = cfg.get("holdout") or "huggingface"
    meta["gap_s"] = [args.gap_min, args.gap_max]
    m = score(done, samples, link.specs, prof, t_start, t_end, meta)
    report = write_outputs(run_dir, m, done, samples)
    print()
    print(report.read_text(encoding="utf-8"))
    info(f"Saved: {report}, metrics.json, trials.csv, pi_metrics.csv in {run_dir}")
    cleanup(run_dir, bool(cfg.get("delete_audio")))
    acc = m["intent_level"]["accuracy"]
    notify("VCM benchmark finished", f"{len(done)} commands, intent accuracy {100 * acc:.1f}%")


def save_cfg(cfg: dict, run_dir: Path) -> None:
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if cfg.get("mode") == "sim":
        return                         # a dry run must not overwrite real settings
    keep = {k: v for k, v in cfg.items() if k not in ("started", "t_start", "size", "seed")}
    SAVED_SETTINGS.write_text(json.dumps(keep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
