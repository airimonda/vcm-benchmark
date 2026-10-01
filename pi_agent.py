#!/usr/bin/env python3
"""pi_agent.py - telemetry agent for the Raspberry Pi voice-command benchmark.

Runs ON THE PI. Python 3.7+, standard library only. It reports hardware specs,
live metrics (temperature, CPU, memory, ...), new lines of your assistant's log
files, and answers clock "ping" probes from the laptop.

Usually the laptop starts it for you over ssh and reads JSON lines from stdout.
If your network only lets the Pi talk to the laptop, run it by hand in --post
mode and it will push everything to the laptop over HTTP:

    python3 pi_agent.py --post http://LAPTOP_IP:8765 \\
        --log ~/myassistant/log.txt --proc "python.*assistant"

Options:
    --specs-only      print one "specs" JSON line and exit
    --log PATH        tail this file (repeatable); only NEW lines are reported
    --log-cmd CMD     run CMD in a shell (repeatable) and report each output line
                      (stdout+stderr) as a log with path "cmd:CMD"; restarted
                      2 s after it exits, e.g. "journalctl --user -u vcm -f -n 0 -o cat"
    --proc PATTERN    regex matched against process command lines; CPU, memory
                      and threads of matching processes are summed
    --interval SECS   metrics period (default 1.0)
    --post URL        push messages to URL/event (no stdin); clock sync via
                      URL/ping

Output (stdout mode): one JSON object per line, each with "type" and "t"
(Pi time.time()). Types: specs, metrics, log, pong, clock (post mode), error.
stdin commands (stdout mode): "ping <id>" -> pong reply; "quit" or EOF -> exit.
Ctrl-C or SIGTERM also exit cleanly.

Missing files (e.g. on macOS or non-Pi Linux) just give null fields.
"""
from __future__ import print_function

import argparse
import collections
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

try:
    from urllib.request import Request, urlopen
except ImportError:  # pragma: no cover
    Request = urlopen = None

MAX_QUEUE = 5000
PACKAGES = ["onnxruntime", "tflite_runtime", "tensorflow", "torch", "numpy",
            "sounddevice", "pyaudio", "vosk"]

STOP = threading.Event()
_out_lock = threading.Lock()
_queue = collections.deque()
_queue_lock = threading.Lock()
_post_mode = False


# ---------------------------------------------------------------- output
def emit(msg):
    msg.setdefault("t", time.time())
    if _post_mode:
        with _queue_lock:
            _queue.append(msg)
            while len(_queue) > MAX_QUEUE:
                _queue.popleft()
        return
    line = json.dumps(msg)
    with _out_lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except (IOError, OSError, ValueError):
            STOP.set()


def emit_error(where, msg):
    emit({"type": "error", "where": where, "msg": str(msg)})


# ---------------------------------------------------------------- helpers
def read_text(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


def run_cmd(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=3, universal_newlines=True)
        if p.returncode != 0:
            return None
        return p.stdout.strip() or None
    except Exception:
        return None


def meminfo():
    out = {}
    txt = read_text("/proc/meminfo")
    if not txt:
        return out
    for line in txt.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])  # kB
    return out


def kb_to_mb(v):
    return None if v is None else round(v / 1024.0, 1)


def pkg_version(name):
    try:
        from importlib import metadata
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
        except Exception:
            pass
        for alt in (name.replace("_", "-"),):
            try:
                return metadata.version(alt)
            except Exception:
                pass
        return None
    except ImportError:
        pass
    try:
        import pkg_resources
        for n in (name, name.replace("_", "-")):
            try:
                return pkg_resources.get_distribution(n).version
            except Exception:
                pass
        return None
    except ImportError:
        pass
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def get_throttled():
    out = run_cmd(["vcgencmd", "get_throttled"])
    if out and "=" in out:
        return out.split("=", 1)[1].strip()
    return None


def read_temp():
    txt = read_text("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(txt.strip()) / 1000.0, 1)
    except Exception:
        pass
    out = run_cmd(["vcgencmd", "measure_temp"])
    if out:
        m = re.search(r"([\d.]+)", out)
        if m:
            return float(m.group(1))
    return None


def read_freq_mhz(name="scaling_cur_freq"):
    txt = read_text("/sys/devices/system/cpu/cpu0/cpufreq/" + name)
    try:
        return round(int(txt.strip()) / 1000.0, 1)
    except Exception:
        return None


# ---------------------------------------------------------------- specs
def cpu_model():
    txt = read_text("/proc/cpuinfo") or ""
    found = {}
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            found.setdefault(k.strip(), v.strip())
    for key in ("Model name", "model name", "Hardware"):
        if found.get(key):
            return found[key]
    return None


def os_name():
    txt = read_text("/etc/os-release")
    if txt:
        for line in txt.splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    try:
        return platform.platform()
    except Exception:
        return None


def build_specs():
    mi = meminfo()
    model = read_text("/proc/device-tree/model")
    if model:
        model = model.replace("\x00", "").strip()
    try:
        disk = round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 2)
    except Exception:
        disk = None
    up = None
    txt = read_text("/proc/uptime")
    try:
        up = float(txt.split()[0])
    except Exception:
        pass
    maxf = read_freq_mhz("cpuinfo_max_freq")
    return {
        "type": "specs",
        "t": time.time(),
        "hostname": socket.gethostname(),
        "model": model or None,
        "os": os_name(),
        "kernel": platform.release() or None,
        "arch": platform.machine() or None,
        "cpu_model": cpu_model(),
        "cores": os.cpu_count(),
        "max_freq_mhz": maxf,
        "ram_mb": kb_to_mb(mi.get("MemTotal")),
        "swap_mb": kb_to_mb(mi.get("SwapTotal")),
        "disk_free_gb": disk,
        "python": platform.python_version(),
        "packages": dict((p, pkg_version(p)) for p in PACKAGES),
        "audio_inputs": run_cmd(["arecord", "-l"]),
        "audio_outputs": run_cmd(["aplay", "-l"]),
        "throttled": get_throttled(),
        "uptime_s": up,
    }


# ---------------------------------------------------------------- metrics
def read_cpu_stat():
    """Return list of (busy, total) for [all, cpu0, cpu1, ...] or None."""
    txt = read_text("/proc/stat")
    if not txt:
        return None
    res = []
    for line in txt.splitlines():
        if line.startswith("cpu"):
            parts = line.split()
            try:
                vals = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            total = sum(vals[:8])
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            res.append((total - idle, total))
    return res or None


def pct(prev, cur):
    db, dt = cur[0] - prev[0], cur[1] - prev[1]
    if dt <= 0:
        return None
    return round(100.0 * db / dt, 1)


def find_pids(regex, own):
    pids = []
    try:
        names = os.listdir("/proc")
    except Exception:
        return pids
    for n in names:
        if not n.isdigit() or int(n) in own:
            continue
        try:
            with open("/proc/%s/cmdline" % n, "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        if cmd and regex.search(cmd):
            pids.append(int(n))
    return sorted(pids)


def proc_stat(pid):
    """Return (cpu_ticks, rss_mb, threads) or None."""
    try:
        with open("/proc/%d/stat" % pid, "r") as f:
            s = f.read()
        rest = s[s.rindex(")") + 2:].split()
        ticks = int(rest[11]) + int(rest[12])  # utime, stime
        rss = 0.0
        threads = int(rest[17])
        with open("/proc/%d/status" % pid, "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                    break
        return ticks, rss, threads
    except Exception:
        return None


def metrics_loop(interval, pattern):
    try:
        regex = re.compile(pattern) if pattern else None
    except re.error as e:
        emit_error("proc", "bad regex: %s" % e)
        regex = None
    own = set([os.getpid(), os.getppid()])
    try:
        hz = os.sysconf("SC_CLK_TCK")
    except Exception:
        hz = 100
    prev_cpu = read_cpu_stat()
    prev_ticks = {}
    prev_wall = time.time()
    while not STOP.wait(interval):
        try:
            now = time.time()
            m = {"type": "metrics", "t": now}
            m["temp_c"] = read_temp()
            cur = read_cpu_stat()
            if cur and prev_cpu and len(cur) == len(prev_cpu):
                m["cpu_pct"] = pct(prev_cpu[0], cur[0])
                m["cpu_pct_per_core"] = [pct(a, b) for a, b in zip(prev_cpu[1:], cur[1:])]
            else:
                m["cpu_pct"] = None
                m["cpu_pct_per_core"] = None
            prev_cpu = cur
            m["freq_mhz"] = read_freq_mhz()
            try:
                m["load1"] = os.getloadavg()[0]
            except Exception:
                m["load1"] = None
            mi = meminfo()
            avail = mi.get("MemAvailable")
            total = mi.get("MemTotal")
            m["mem_avail_mb"] = kb_to_mb(avail)
            m["mem_used_mb"] = kb_to_mb(total - avail) if total and avail is not None else None
            m["throttled"] = get_throttled()
            if regex is None:
                m["proc"] = None
            else:
                pids = find_pids(regex, own)
                ticks_now = {}
                rss = 0.0
                threads = 0
                for pid in pids:
                    st = proc_stat(pid)
                    if st:
                        ticks_now[pid] = st[0]
                        rss += st[1]
                        threads += st[2]
                dwall = now - prev_wall
                cpu = None
                if ticks_now and dwall > 0 and prev_ticks:
                    d = 0
                    for pid, tk in ticks_now.items():
                        d += tk - prev_ticks.get(pid, tk)
                    cpu = round(d / float(hz) / dwall * 100.0, 1)
                elif ticks_now and dwall > 0:
                    cpu = None
                if not ticks_now:
                    m["proc"] = {"pids": [], "cpu_pct": None, "cpu_time_s": None,
                                 "rss_mb": None, "threads": None}
                else:
                    m["proc"] = {"pids": sorted(ticks_now), "cpu_pct": cpu,
                                 "cpu_time_s": round(sum(ticks_now.values()) / float(hz), 2),
                                 "rss_mb": round(rss, 1), "threads": threads}
                prev_ticks = ticks_now
            prev_wall = now
            emit(m)
        except Exception as e:
            emit_error("metrics", e)


# ---------------------------------------------------------------- log tail
def tail_loop(path):
    real = os.path.expanduser(path)
    fh = None
    ino = None
    pos = 0
    buf = b""
    first = True
    try:
        while not STOP.is_set():
            try:
                st = os.stat(real)
            except OSError:
                if fh:
                    fh.close()
                    fh = None
                buf = b""
                first = False  # once seen missing, a new file is read from start
                STOP.wait(0.02)
                continue
            if fh is not None and st.st_ino != ino:
                fh.close()
                fh = None
            if fh is None:
                try:
                    fh = open(real, "rb")
                except OSError:
                    STOP.wait(0.02)
                    continue
                ino = st.st_ino
                buf = b""
                if first:
                    pos = fh.seek(0, os.SEEK_END)
                    first = False
                else:
                    pos = 0
            elif st.st_size < pos:
                fh.seek(0)
                pos = 0
                buf = b""
            data = fh.read()
            if data:
                pos += len(data)
                now = time.time()
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", "replace").rstrip("\r")
                    emit({"type": "log", "path": path, "line": text, "t": now})
            else:
                STOP.wait(0.02)
    except Exception as e:
        emit_error("log:" + path, e)


# ---------------------------------------------------------------- log commands
_children = []
_children_lock = threading.Lock()


def kill_children():
    with _children_lock:
        procs = list(_children)
    for pr in procs:
        try:
            os.killpg(pr.pid, signal.SIGKILL)
        except Exception:
            try:
                pr.kill()
            except Exception:
                pass


def cmd_loop(cmd):
    path = "cmd:" + cmd
    while not STOP.is_set():
        pr = None
        try:
            pr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True)
            with _children_lock:
                _children.append(pr)
            if STOP.is_set():
                kill_children()
            for raw in iter(pr.stdout.readline, b""):
                emit({"type": "log", "path": path,
                      "line": raw.decode("utf-8", "replace").rstrip("\r\n"),
                      "t": time.time()})
            pr.wait()
            if not STOP.is_set():
                emit_error("log-cmd:" + cmd, "exited with code %s" % pr.returncode)
        except Exception as e:
            emit_error("log-cmd:" + cmd, e)
        finally:
            if pr is not None:
                with _children_lock:
                    if pr in _children:
                        _children.remove(pr)
        STOP.wait(2.0)


# ---------------------------------------------------------------- stdin
def stdin_loop():
    try:
        while not STOP.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line == "quit":
                break
            if line.startswith("ping"):
                parts = line.split(None, 1)
                emit({"type": "pong", "id": parts[1] if len(parts) > 1 else "",
                      "t": time.time()})
    except Exception as e:
        emit_error("stdin", e)
    STOP.set()


# ---------------------------------------------------------------- post mode
def http_post(url, obj):
    data = json.dumps(obj).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=3)
    try:
        return resp.read()
    finally:
        resp.close()


def sender_loop(base):
    url = base.rstrip("/") + "/event"
    backoff = 0.0
    while True:
        stopping = STOP.is_set()
        with _queue_lock:
            batch = []
            while _queue and len(batch) < 500:
                batch.append(_queue.popleft())
        if batch:
            try:
                http_post(url, batch)
                backoff = 0.0
            except Exception:
                with _queue_lock:
                    _queue.extendleft(reversed(batch))
                    while len(_queue) > MAX_QUEUE:
                        _queue.popleft()
                backoff = min(5.0, max(0.5, backoff * 2))
                if stopping:
                    return
                STOP.wait(backoff)
                continue
            if _queue:
                continue
        if stopping:
            return
        STOP.wait(0.2)


def clock_loop(base):
    url = base.rstrip("/") + "/ping"
    n = 0
    while not STOP.is_set():
        n += 1
        try:
            t0 = time.time()
            body = http_post(url, {"id": "%s-%d" % (socket.gethostname(), n), "t": t0})
            t1 = time.time()
            server_t = float(json.loads(body.decode("utf-8"))["t"])
            emit({"type": "clock", "offset_s": server_t - (t0 + t1) / 2.0,
                  "rtt_s": t1 - t0})
        except Exception:
            STOP.wait(2.0)
            continue
        STOP.wait(15.0)


# ---------------------------------------------------------------- main
def _on_signal(signum, frame):
    STOP.set()


def main(argv=None):
    global _post_mode
    ap = argparse.ArgumentParser(description="Raspberry Pi benchmark agent")
    ap.add_argument("--specs-only", action="store_true")
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--log-cmd", action="append", default=[])
    ap.add_argument("--proc")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--post")
    args = ap.parse_args(argv)
    _post_mode = bool(args.post) and not args.specs_only

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        specs = build_specs()
    except Exception as e:
        specs = {"type": "specs", "t": time.time()}
        emit_error("specs", e)
    if args.specs_only:
        emit(specs)
        return 0

    threads = []

    def start(fn, *a):
        th = threading.Thread(target=fn, args=a)
        th.daemon = True
        th.start()
        threads.append(th)

    emit(specs)
    sender = None
    if _post_mode:
        sender = threading.Thread(target=sender_loop, args=(args.post,))
        sender.daemon = True
        sender.start()
        start(clock_loop, args.post)
    else:
        start(stdin_loop)
    for p in args.log:
        start(tail_loop, p)
    for c in args.log_cmd:
        start(cmd_loop, c)
    start(metrics_loop, max(0.05, args.interval), args.proc)

    while not STOP.wait(0.2):
        pass
    kill_children()
    if sender is not None:
        sender.join(timeout=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
