"""Talking to the student's Raspberry Pi.

Four ways to connect (pick whatever works on your network):

* ssh       laptop -> Pi over SSH (Wi-Fi/LAN IP, raspberrypi.local, Tailscale,
            USB-ethernet gadget, an ~/.ssh/config alias, ...). The laptop copies
            pi_agent.py to the Pi and runs it; everything streams back over SSH.
* http      Pi -> laptop. For networks where the laptop cannot reach the Pi.
            You start pi_agent.py on the Pi yourself with --post http://LAPTOP:PORT.
* manual    no connection. After each trial you type what the Pi did.
* sim       no Pi at all; a fake Pi answers (to try the pipeline).

Every link produces the same things: `specs` (dict), a list of metric samples,
and an event queue of parsed log lines with laptop-clock timestamps.
"""
from __future__ import annotations

import json
import queue
import random
import re
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AGENT_PATH = Path(__file__).resolve().parent.parent / "pi_agent.py"
REMOTE_AGENT = "~/.vcm_bench/pi_agent.py"


# ------------------------------------------------------------------ parsing

_KEY = r"(?:intent|command|cmd|label|prediction|predicted|pred|action|result)"
_INTENT_RE = re.compile(_KEY + r"[\"']?\s*[=:]\s*[\"']?(?P<intent>[A-Za-z][A-Za-z0-9_\-]*)", re.I)
# slot value runs until a delimiter or the next "key=" / "key:"
_SLOT_RE = re.compile(r"\b(?:slot_value|slot|value)[\"']?\s*[=:]\s*[\"']?(?P<slot>.+?)[\"']?"
                      r"(?=\s*[,;|}\)\]]|\s+[A-Za-z_]+\s*[=:]|\s*$)", re.I)
_INFER_RE = re.compile(r"(?:infer(?:ence)?_?ms|latency_?ms|model_?ms)[\"']?\s*[=:]\s*(?P<v>[0-9.]+)", re.I)
_AUDIO_RE = re.compile(r"(?:audio|window|utterance)_?ms[\"']?\s*[=:]\s*(?P<v>[0-9.]+)", re.I)
DEFAULT_WAKE_REGEX = r"(?i)\b(wake[ _-]?word|wake detected|woke|hotword|wake=1|listening)\b"

_JSON_INTENT_KEYS = ("intent", "command", "cmd", "label", "prediction", "pred", "action")
_JSON_SLOT_KEYS = ("slot", "slot_value", "value", "slots")
_JSON_INFER_KEYS = ("infer_ms", "inference_ms", "latency_ms", "model_ms")
_JSON_AUDIO_KEYS = ("audio_ms", "window_ms", "utterance_ms")


@dataclass
class Event:
    kind: str                 # "command" or "wake"
    t: float                  # laptop clock
    intent: str = ""
    slot: str = ""
    infer_ms: float | None = None
    audio_ms: float | None = None
    raw: str = ""


class LineParser:
    """Turns one log line from the student's runtime into an Event (or None).

    JSON lines are read by key (intent/command/label, slot/value, infer_ms, ...);
    other lines are searched for `intent=X` / `command: X` / `prediction=X`,
    `slot=...`, `infer_ms=...`; a student-supplied `command_regex` (named groups
    `intent`, optional `slot`, `infer_ms`, `audio_ms`) replaces that search.
    """

    def __init__(self, command_regex: str | None = None, wake_regex: str | None = None):
        self.command_re = re.compile(command_regex, re.IGNORECASE) if command_regex else None
        self.wake_re = re.compile(wake_regex or DEFAULT_WAKE_REGEX)

    def parse(self, line: str, t: float) -> Event | None:
        s = line.strip()
        j = s.find("{")
        if j >= 0:
            try:
                obj = json.loads(s[j:])
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                ev = self._from_json(obj, t, s)
                if ev:
                    return ev
        if self.command_re is not None:          # the student's own regex
            m = self.command_re.search(s)
            if m and m.groupdict().get("intent"):
                g = m.groupdict()
                return Event("command", t, g["intent"].strip(), (g.get("slot") or "").strip(),
                             _f(g.get("infer_ms")), _f(g.get("audio_ms")), s)
        else:
            m = _INTENT_RE.search(s)
            if m:
                sm, im, am = _SLOT_RE.search(s), _INFER_RE.search(s), _AUDIO_RE.search(s)
                return Event("command", t, m.group("intent"), sm.group("slot").strip() if sm else "",
                             _f(im and im.group("v")), _f(am and am.group("v")), s)
        if self.wake_re.search(s):
            return Event("wake", t, raw=s)
        return None

    def _from_json(self, o: dict, t: float, raw: str) -> Event | None:
        kind = str(o.get("event") or o.get("type") or "").lower()
        intent = next((o[k] for k in _JSON_INTENT_KEYS if o.get(k) not in (None, "")), None)
        if intent is not None and not isinstance(intent, (dict, list)):
            slot = next((o[k] for k in _JSON_SLOT_KEYS if o.get(k) not in (None, "")), "")
            if isinstance(slot, dict):
                slot = " ".join(str(v) for v in slot.values())
            infer = next((o[k] for k in _JSON_INFER_KEYS if o.get(k) is not None), None)
            audio = next((o[k] for k in _JSON_AUDIO_KEYS if o.get(k) is not None), None)
            return Event("command", t, str(intent), str(slot), _f(infer), _f(audio), raw)
        if "wake" in kind or o.get("wake") in (True, 1, "1"):
            return Event("wake", t, raw=raw)
        return None


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ links

@dataclass
class PiLink:
    """Base class. Subclasses fill specs / samples / events."""
    parser: LineParser
    specs: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)       # metric dicts, "t" on laptop clock
    events: "queue.Queue[Event]" = field(default_factory=queue.Queue)
    raw_lines: list = field(default_factory=list)     # (laptop t, line) - all log lines seen
    errors: list = field(default_factory=list)
    offset: float = 0.0          # pi_clock - laptop_clock
    rtt: float | None = None
    manual: bool = False

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def handle(self, msg: dict, recv_t: float) -> None:
        typ = msg.get("type")
        pi_t = msg.get("t")
        t = pi_t - self.offset if isinstance(pi_t, (int, float)) and self.rtt is not None else recv_t
        if typ == "specs":
            self.specs = msg
        elif typ == "metrics":
            m = dict(msg, t=t)
            self.samples.append(m)
        elif typ == "log":
            line = str(msg.get("line", ""))
            self.raw_lines.append((t, line))
            ev = self.parser.parse(line, t)
            if ev:
                self.events.put(ev)
        elif typ == "clock":          # sent by the agent in --post mode
            self.offset = -float(msg["offset_s"])
            self.rtt = float(msg["rtt_s"])
        elif typ == "error":
            self.errors.append(msg)

    def drain(self) -> list[Event]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def agent_args(self, logs: list[str], log_cmds: list[str], proc: str | None,
                   interval: float) -> list[str]:
        a = ["--interval", str(interval)]
        for p in logs:
            a += ["--log", p]
        for c in log_cmds:
            a += ["--log-cmd", c]
        if proc:
            a += ["--proc", proc]
        return a


class SshLink(PiLink):
    def __init__(self, parser: LineParser, target: str, ssh_opts: list[str], control_dir: Path,
                 logs: list[str], log_cmds: list[str], proc: str | None, interval: float = 1.0,
                 python: str = "python3"):
        super().__init__(parser)
        self.target = target
        self.python = python
        # one shared SSH connection, so a password is asked at most once
        control_dir.mkdir(parents=True, exist_ok=True)
        self.ssh = ["ssh", "-o", "ControlMaster=auto", "-o", f"ControlPath={control_dir}/cm-%C",
                    "-o", "ControlPersist=900", "-o", "ServerAliveInterval=10", *ssh_opts, target]
        self.args = self.agent_args(logs, log_cmds, proc, interval)
        self.proc: subprocess.Popen | None = None
        self._ping_sent: dict[str, float] = {}
        self._lock = threading.Lock()

    def run(self, remote_cmd: str, input_bytes: bytes | None = None, timeout: float = 30) -> subprocess.CompletedProcess:
        return subprocess.run(self.ssh + [remote_cmd], input=input_bytes, capture_output=True,
                              timeout=timeout)

    def check(self) -> tuple[bool, str]:
        try:
            r = self.run("echo ok", timeout=60)
        except subprocess.TimeoutExpired:
            return False, "timed out"
        return r.returncode == 0, (r.stdout + r.stderr).decode(errors="replace").strip()

    def upload_agent(self) -> None:
        r = self.run("mkdir -p ~/.vcm_bench && cat > ~/.vcm_bench/pi_agent.py",
                     input_bytes=AGENT_PATH.read_bytes())
        if r.returncode != 0:
            raise RuntimeError("could not copy pi_agent.py: " + r.stderr.decode(errors="replace"))

    def fetch_specs(self) -> dict:
        r = self.run(f"{self.python} {REMOTE_AGENT} --specs-only", timeout=60)
        for line in r.stdout.decode(errors="replace").splitlines():
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("type") == "specs":
                self.specs = msg
        if not self.specs:
            raise RuntimeError("pi_agent.py gave no specs: " + r.stderr.decode(errors="replace")[-500:])
        return self.specs

    def start(self) -> None:
        cmd = " ".join([self.python, "-u", REMOTE_AGENT] + [shlex.quote(a) for a in self.args])
        self.proc = subprocess.Popen(self.ssh + [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._err_reader, daemon=True).start()
        self.sync_clock()

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            recv = time.time()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "pong":
                sent = self._ping_sent.pop(str(msg.get("id")), None)
                if sent is not None:
                    rtt = recv - sent
                    if self.rtt is None or rtt <= self.rtt * 1.5:
                        self.offset = msg["t"] - (sent + recv) / 2
                        self.rtt = rtt if self.rtt is None else min(rtt, self.rtt)
                continue
            self.handle(msg, recv)

    def _err_reader(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in self.proc.stderr:
            self.errors.append({"where": "ssh", "msg": raw.decode(errors="replace").strip()})

    def ping(self) -> None:
        if not self.proc or not self.proc.stdin:
            return
        pid = str(random.getrandbits(32))
        with self._lock:
            self._ping_sent[pid] = time.time()
            try:
                self.proc.stdin.write(f"ping {pid}\n".encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def sync_clock(self, n: int = 8) -> None:
        self.rtt = None
        for _ in range(n):
            self.ping()
            time.sleep(0.15)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"quit\n")
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        subprocess.run(self.ssh[:-1] + ["-O", "exit", self.target], capture_output=True)


class HttpLink(PiLink):
    """Laptop runs a small HTTP server; the Pi's agent POSTs to it."""

    def __init__(self, parser: LineParser, port: int = 8765):
        super().__init__(parser)
        self.port = port
        self.server: ThreadingHTTPServer | None = None
        self.last_seen = 0.0

    def start(self) -> None:
        link = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the console clean
                pass

            def do_POST(self):
                now = time.time()
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                link.last_seen = now
                if self.path.rstrip("/") == "/ping":
                    out = json.dumps({"t": time.time()}).encode()
                else:
                    try:
                        msgs = json.loads(body)
                    except ValueError:
                        msgs = []
                    for m in msgs if isinstance(msgs, list) else [msgs]:
                        if isinstance(m, dict):
                            link.handle(m, now)
                    out = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def alive(self) -> bool:
        return time.time() - self.last_seen < 10

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()


class ManualLink(PiLink):
    def __init__(self, parser: LineParser):
        super().__init__(parser, manual=True)

    def alive(self) -> bool:
        return True


class SimLink(PiLink):
    """A fake Pi for trying the pipeline: answers each trial after ~0.6 s."""

    def __init__(self, parser: LineParser, accuracy: float = 0.85, seed: int = 0):
        super().__init__(parser)
        self.accuracy = accuracy
        self.rng = random.Random(seed)
        self.rtt = 0.0
        self.specs = {"type": "specs", "hostname": "simulated-pi", "model": "Simulated Raspberry Pi",
                      "cores": 4, "ram_mb": 4096, "os": "simulation"}
        self._cpu_time = 0.0

    def alive(self) -> bool:
        return True

    def respond(self, truth_intent: str, truth_slot: str, t_cmd_end: float) -> None:
        from .schema import INTENTS, OOS
        r = self.rng.random()
        now = time.time()
        latency = self.rng.uniform(0.25, 0.9)
        self._cpu_time += self.rng.uniform(0.3, 0.6)
        self.samples.append({"type": "metrics", "t": now, "temp_c": 50 + self.rng.random() * 8,
                             "cpu_pct": self.rng.uniform(15, 45), "freq_mhz": 1800, "load1": 0.8,
                             "mem_used_mb": 900, "mem_avail_mb": 3000, "throttled": "0x0",
                             "proc": {"pids": [1], "cpu_pct": self.rng.uniform(20, 90),
                                      "cpu_time_s": self._cpu_time, "rss_mb": 180, "threads": 6}})
        if r < 0.05:
            return                                    # missed the wake word
        self.events.put(Event("wake", t_cmd_end - 1.0, raw="wake word detected"))
        if r < self.accuracy:
            intent, slot = truth_intent, truth_slot
        elif r < self.accuracy + 0.05 and truth_slot:
            intent, slot = truth_intent, "22 degrees" if truth_intent == "TEMPERATURE" else "something"
        else:
            intent, slot = self.rng.choice(INTENTS + [OOS]), ""
        self.events.put(Event("command", t_cmd_end + latency, intent, slot,
                              infer_ms=self.rng.uniform(40, 120), audio_ms=1000.0,
                              raw=f"intent={intent} slot={slot}"))


def laptop_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))
