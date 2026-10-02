import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# pi_agent.py runs on the Raspberry Pi (Linux); its process handling is POSIX-only.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Pi-side agent, POSIX only")

AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pi_agent.py")


def read_until(proc, pred, timeout=8):
    end = time.time() + timeout
    seen = []
    while time.time() < end:
        line = proc.stdout.readline()
        if not line:
            break
        msg = json.loads(line)
        seen.append(msg)
        if pred(seen):
            return seen
    return seen


def test_specs_only():
    out = subprocess.run([sys.executable, AGENT, "--specs-only"], stdout=subprocess.PIPE,
                         universal_newlines=True, timeout=20)
    assert out.returncode == 0
    msg = json.loads(out.stdout.strip().splitlines()[0])
    assert msg["type"] == "specs"
    assert "t" in msg and "packages" in msg and msg["cores"]


def test_stdout_mode(tmp_path):
    log = tmp_path / "a.log"
    log.write_text("old line\n")
    p = subprocess.Popen([sys.executable, AGENT, "--log", str(log), "--interval", "0.2",
                          "--proc", "pi_agent"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, universal_newlines=True)
    try:
        seen = read_until(p, lambda s: any(m["type"] == "metrics" for m in s))
        assert seen[0]["type"] == "specs"
        assert any(m["type"] == "metrics" for m in seen)
        with open(str(log), "a") as f:
            f.write("hello\nwor")
            f.flush()
            time.sleep(0.2)
            f.write("ld\n")
        p.stdin.write("ping 7\n")
        p.stdin.flush()
        seen = read_until(p, lambda s: any(m["type"] == "pong" for m in s) and
                          sum(m["type"] == "log" for m in s) >= 2)
        logs = [m["line"] for m in seen if m["type"] == "log"]
        assert logs == ["hello", "world"]
        pong = [m for m in seen if m["type"] == "pong"][0]
        assert pong["id"] == "7"
        p.stdin.close()
        p.wait(timeout=3)
        assert p.returncode == 0
    finally:
        if p.poll() is None:
            p.kill()


class _H(BaseHTTPRequestHandler):
    events = []
    pings = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/event":
            _H.events.extend(body)
            out = b"{}"
        else:
            _H.pings.append(body)
            out = json.dumps({"t": time.time() + 5}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_post_mode(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log = tmp_path / "b.log"
    log.write_text("")
    p = subprocess.Popen([sys.executable, AGENT, "--post", "http://127.0.0.1:%d" % srv.server_port,
                          "--log", str(log), "--interval", "0.2"], stdout=subprocess.PIPE,
                         universal_newlines=True)
    try:
        time.sleep(1.0)
        with open(str(log), "a") as f:
            f.write("posted line\n")
        end = time.time() + 6
        while time.time() < end:
            types = set(e["type"] for e in _H.events)
            if {"specs", "metrics", "log", "clock"} <= types:
                break
            time.sleep(0.1)
        types = set(e["type"] for e in _H.events)
        assert {"specs", "metrics", "log", "clock"} <= types
        clock = [e for e in _H.events if e["type"] == "clock"][0]
        assert 4 < clock["offset_s"] < 6
        assert _H.pings
        p.terminate()
        p.wait(timeout=5)
    finally:
        if p.poll() is None:
            p.kill()
        srv.shutdown()


def test_log_cmd():
    cmd = "%s -c \"print('hi')\"" % sys.executable
    p = subprocess.Popen([sys.executable, AGENT, "--log-cmd", cmd, "--interval", "5"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         universal_newlines=True)
    try:
        seen = read_until(p, lambda s: any(m["type"] == "error" for m in s))
        logs = [m for m in seen if m["type"] == "log"]
        assert logs and logs[0]["line"] == "hi" and logs[0]["path"] == "cmd:" + cmd
        assert any(m["type"] == "error" and m["where"].startswith("log-cmd") for m in seen)
        # restarts after 2 s
        seen = read_until(p, lambda s: any(m["type"] == "log" for m in s), timeout=6)
        assert any(m["type"] == "log" and m["line"] == "hi" for m in seen)
        p.stdin.close()
        p.wait(timeout=3)
    finally:
        if p.poll() is None:
            p.kill()


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs Linux /proc (the Pi)")
def test_find_log_writers(tmp_path):
    """Without --proc, the agent measures whichever process has the log file open."""
    sys.path.insert(0, os.path.dirname(AGENT))
    import pi_agent
    log = tmp_path / "vcm_benchmark.log"
    writer = subprocess.Popen([sys.executable, "-c",
                               "import time,sys; f=open(sys.argv[1],'a'); f.write('x\\n'); f.flush(); time.sleep(30)",
                               str(log)])
    try:
        deadline = time.time() + 5
        pids = []
        while time.time() < deadline and writer.pid not in pids:
            pids = pi_agent.find_log_writers([str(log)], set([os.getpid()]))
            time.sleep(0.1)
        assert writer.pid in pids
    finally:
        writer.kill()
