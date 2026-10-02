# VCM live benchmark (AI231 ME2)

One script that every student runs to test their Raspberry Pi voice assistant on the
same class holdout set, in the same way.

The laptop plays "**wake word, command**" out loud. Your Pi hears it and does whatever your
assistant does. The laptop reads what your assistant printed and scores it against the
agreed **19 intents** and **93 commands** (the "Option B" variations). It also records
your Pi's temperature, CPU, RAM, latency and real-time factor.

```
laptop speaker  ──(sound)──▶  Pi mic ─▶ your assistant ─▶ log line "intent=TIMER slot=30 seconds"
      ▲                                                          │
      └──────────── benchmark.py ◀── pi_agent.py (SSH or HTTP) ◀─┘  + CPU / temp / RAM every second
```

## Install (laptop: macOS or Windows)

Python 3.10 or newer. On Windows install it from python.org and tick **"Add python.exe to PATH"**.

macOS (Terminal):

```bash
git clone https://github.com/airimonda/vcm-benchmark.git && cd vcm-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/airimonda/vcm-benchmark.git; cd vcm-benchmark
py -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. No git? Download the ZIP from GitHub
(Code > Download ZIP) and unzip it.

On the Pi you need nothing extra: `pi_agent.py` uses only the Python standard library.
(Linux laptops: also `sudo apt install libportaudio2`.)

### Per-OS notes

| | macOS | Windows |
|---|---|---|
| Run the script | `python benchmark.py` | `python benchmark.py` (or `py benchmark.py`) |
| Microphone permission | allow Terminal when asked (System Settings > Privacy > Microphone) | Settings > Privacy > Microphone > let desktop apps use it |
| SSH client | built in | built in on Windows 10/11; if `ssh` is missing: Settings > System > Optional features > OpenSSH Client |
| Password-less SSH (needed for unattended reconnects) | `ssh-keygen -t ed25519`, then `ssh-copy-id user@pi` | `ssh-keygen -t ed25519`, then `type $env:USERPROFILE\.ssh\id_ed25519.pub \| ssh user@pi "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"` |
| SSH password prompts | asked once per run (shared connection) | asked at each step (Windows ssh has no shared connection), so set up the key |
| `--mode http` (Pi to laptop) | allow incoming connections for Python when asked | allow Python in the Windows Defender Firewall prompt (Private networks) |
| Kept awake during the run | `caffeinate` | Windows power request (no setting needed); still keep it plugged in |
| Done notification | Notification Center | tray balloon |

## Before you start

1. **Your Pi is connected** to the laptop's network (same Wi-Fi, hotspot, Tailscale, cable ...).
2. **Your assistant logs each command it recognises** to a file on the Pi. The benchmark reads it
   directly, whatever the format:
   * the newest `.log` in `~/vcm_benchmark/` (recommended name: `<id>_<date-time>.log`), or
   * otherwise **any log file your assistant writes** under your home folder (`.log`, `.jsonl`,
     `.txt`, `.out`) or its systemd journal: the file that shows a command first is used, and
     the report names it.

   Any format works: JSON (also nested, e.g. `{"heard": {"intent": ..., "slots": {...}}}`),
   `intent=TIMER slot=30 seconds`, or plain text like `TIMER 30 seconds | model 42 ms`. Required:
   the intent (or one of the 93 phrases / its class number), and for timing `infer_ms` and
   `audio_ms` (see [What your Pi must print](#what-your-pi-must-print)). Minimal version:

   ```python
   import datetime, json, os
   os.makedirs(os.path.expanduser("~/vcm_benchmark"), exist_ok=True)
   log = open(os.path.expanduser(f"~/vcm_benchmark/{STUDENT_ID}_{datetime.datetime.now():%Y%m%d-%H%M%S}.log"),
              "a", buffering=1)
   # after each decision:
   print(json.dumps({"intent": intent, "slot": slot, "infer_ms": infer_ms, "audio_ms": audio_ms}),
         file=log, flush=True)
   ```

3. **Your assistant is running**, and the laptop speaker is about 1 m from the Pi's mic.

How the script reaches the Pi (no setup questions unless needed):

1. It tries to log in over SSH by itself, without a password: the Pi you used last time, Pi
   entries in your `~/.ssh/config`, Pi-like hosts you have logged in to before, and
   `raspberrypi.local`. It picks the one that has `~/vcm_benchmark/`.
2. If that fails, it shows what went wrong for each address (not found, no answer, needs a
   password, SSH off ...) and offers:
   * **log in with your username and password**: type `user@host` (or paste your whole
     `ssh ...` command). The script then offers to **set up password-less login** (recommended):
     it creates an SSH key on the laptop if there is none and adds it to the Pi, so you type the
     password **once**. From then on nothing is asked, the Pi is found automatically next time,
     and dropped connections reconnect during the unattended run. (Without it, Windows asks the
     password at every step.)
   * **let the Pi send its data**: paste one line in a terminal on the Pi; it downloads the small
     agent from this repo and sends to the laptop. Works on networks where the laptop cannot
     reach the Pi.

It finds your assistant's process by itself (the one with the log file open) to report its CPU
and RAM.

## Run

```bash
python benchmark.py                 # guided run; answer the questions
python benchmark.py --mode sim --no-audio --size quick --limit 10 --gap-min 1 --gap-max 2
                                    # dry run without a Pi, to see what happens
```

The script walks you through these steps:

1. **Your Pi**: log in over SSH and read its specs (model, CPU, RAM, OS, Python packages,
   microphones).
2. **Record your wake word** 3 times on the laptop microphone (or `--wake-files a.wav b.wav`).
3. **Build the test audio**: each holdout clip gets one of your wake word takes in front of it,
   with a pause in between (default 0.8 s; set it to what your Pi needs after its chime).
   Wake word and command are levelled to the same loudness. For the **false-wake check**, as many
   in-scope commands as there are out-of-scope clips (10) are added **without** the wake word,
   one per intent; your Pi should ignore them. All trials are **shuffled** with a **seed you
   choose** (a random one is suggested; type the same seed again to repeat a test exactly, or use
   a seed the class agrees on so everyone hears the same order).
4. **Mic check** (SSH connection): the Pi records 2 s of room noise with `arecord`, then records
   again while the laptop plays a command without the wake word. The laptop measures room noise,
   laptop speech level, signal-to-noise ratio and clipping, and says **OK / weak / not heard /
   clipping** with what to change (volume, distance, or another Pi microphone; you can pick one
   from the Pi's `arecord -l` list). Repeat until it passes. If your assistant holds the mic
   exclusively ("device busy"), use the `default` input or stop the assistant for the check.
   The result goes in the report header.
   **Sound check**: two warm-up commands, not scored. You see the log lines your Pi printed and
   how they were understood. Fix the volume or the output format here.
5. **Approve**: every remaining question comes now: ONNX model path (optional), and whether to
   delete the generated audio at the end. You see the plan (number of commands, duration,
   finish time) and answer **Start now?** once.
6. **Test, unattended**: from here on nothing is asked. Commands play 10-15 s apart (random), so
   your assistant goes back to sleep between turns. You can walk away:
   * the laptop is kept awake (macOS `caffeinate`, Linux `systemd-inhibit`, Windows API);
     keep it plugged in with the lid open;
   * if the Pi connection drops, the script reconnects (up to 10 min) and replays that command;
     for this, SSH must log in without a password (`ssh-copy-id user@host`; the approval screen
     warns you if it can't);
   * if the speaker fails, it retries, then falls back to the default speaker;
   * if recovery fails, it stops, scores what is done, and tells you the `--resume` command.

   Ctrl+C still pauses (resume / skip / stop and score). Manual mode cannot run unattended.
7. **Results** are printed and saved in `runs/<date-time>/`. A desktop notification says it is done.
8. **Clean up**: the generated audio (your wake word recordings and the trial files) is deleted or
   kept as you chose in step 5. Results are always kept.

Test size: **full** = all 196 holdout clips (186 commands = 2 per variation, plus 10 out-of-scope)
+ 10 false-wake trials, about 60 min. **quick** = 1 clip per variation + the 10 out-of-scope clips
(103) + 10 false-wake trials, about 32 min.

The holdout set downloads automatically from Hugging Face
([airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
split `holdout`, 15 MB) and is cached in `.cache/`. Use `--holdout path/to/dataset/holdout` for a
local copy.

Your answers are saved in `bench_settings.json`, so the next run asks fewer questions
(`--fresh` to start over). Interrupted? `python benchmark.py --resume runs/<run-id>`.

## Options (advanced)

| Flag | What it does |
|---|---|
| `--host user@host` | skip the search and use this Pi (SSH) |
| `--mode http` | go straight to "the Pi sends its data" (paste one line on the Pi) |
| `--mode manual` | no network at all: after each command you type what the Pi did |
| `--mode sim` | no Pi, to try the script |
| `--log-dir DIR` | another log folder on the Pi (newest `.log` is read) |
| `--log FILE` | one fixed log file instead of the folder |
| `--log-cmd "CMD"` | follow a command's output, e.g. `journalctl --user -u myassistant -f -n 0 -o cat` |
| `--proc REGEX` | choose the process to measure yourself |
| `--ssh-opt=-p2222` | extra ssh option |

Over SSH, the script copies `pi_agent.py` to `~/.vcm_bench/` on the Pi and runs it there.

## What your Pi must print

Your assistant must append **one line per recognised command** to a log file on the Pi, in
`~/vcm_benchmark/` (the newest `.log` there is read; see [Before you start](#before-you-start)). Every line must carry **what your model decided**
(either intent + slot, or one of the 93 command phrases, see below) and two timing fields:

| Field | Meaning | Required |
|---|---|---|
| `intent` | what your model decided (one of the 19, or out of scope) | yes |
| `slot` | the slot value, free text (`22`, `twenty two`, `9pm`, `blue`) | for slotted intents |
| `infer_ms` | time your model took for this command: feature extraction + model, in milliseconds | **yes** |
| `audio_ms` | length of the audio your model processed for this command, in milliseconds | **yes** |

The benchmark reads almost any format: JSON (any nesting; also Python `{'intent': ...}` dicts and
objects spread over several lines), `key=value` / `key: value` text, and plain text that names
exactly one intent (written as a label: `TIMER`, `set_temperature_22`) or one of the 93 phrases.
Timing in plain text is picked up from e.g. `took 85 ms` / `model 42ms` and `audio 1500 ms` /
`window 1.5 s`. If a line is still not understood, the sound check shows it and you can give a
pattern (`x`). For example, all of these work:

```
{"intent": "TIMER", "slot": "30 seconds", "infer_ms": 85, "audio_ms": 1500}
intent=TIMER slot=30 seconds infer_ms=85 audio_ms=1500
```

Python example for your runtime:

```python
import json, os, time

t0 = time.perf_counter()
intent, slot = model.predict(audio)            # your features + model
infer_ms = (time.perf_counter() - t0) * 1000
audio_ms = len(audio) / sample_rate * 1000
log = open(os.path.expanduser("~/vcm_benchmark/myid_20261002-140000.log"), "a", buffering=1)  # once at start
print(json.dumps({"intent": intent, "slot": slot,
                  "infer_ms": round(infer_ms, 1), "audio_ms": round(audio_ms)}), file=log, flush=True)
```

### If your model outputs the 93 classes

Print the phrase it chose (as in the dataset manifest's `variation` column, or
[`vcmbench/variations.csv`](vcmbench/variations.csv); case and punctuation don't matter), or its
class number:

```
{"variation": "Set the temperature to 22 degrees", "infer_ms": 85, "audio_ms": 1500}
variation=Set the temperature to 22 degrees infer_ms=85 audio_ms=1500
{"variation_id": 64, "infer_ms": 85, "audio_ms": 1500}
{"variation": "OUT_OF_SCOPE", "infer_ms": 85, "audio_ms": 1500}
```

The phrase is turned into its intent and slot (here TEMPERATURE, 22 degrees), so every metric works
the same as for intent + slot models. The 93-command scores use **the same rule for everyone**
(intent and slot right, wording not judged), so all students stay comparable. On top of that, the
report shows an **exact wording** accuracy for 93-class models: the chosen phrase must be the one
that was spoken. A phrase that is not one of the 93 is scored wrong and listed in the report.

**Class numbers.** A number only means something with your model's class order. All orders below
are built from the holdout manifest's `variation` column; pick yours with `--id-order` (or `n` in
the sound check):

| `--id-order` | Class 0, 1, 2, ... = | Out of scope |
|---|---|---|
| `manifest` (default) | order of first appearance in the manifest (same as `variations.csv`) | 93 |
| `alphabetical` | names sorted A-Z: `sorted(set(...))`, sklearn `LabelEncoder`, pandas category | 93 |
| `alphabetical_oos` | the 93 names + `OUT_OF_SCOPE`, sorted A-Z together | where it sorts |
| `file` | your own label file (`--id-labels labels.txt`: one name per line, line 1 = class 0; or a JSON list / `{"name": id}`) | where `OUT_OF_SCOPE` is in the file |

You are not scored wrong silently:

* in the sound check, the script shows which phrase your number was read as; if that is not the
  spoken command but another order gives the right one, it says so;
* after the test, it checks every order against the right answers; if another order fits
  better, the report's **Check** list says so and gives the command to re-score.

Re-scoring needs no Pi and no audio:

```
python benchmark.py --rescore runs/<run-id> --id-order alphabetical
python benchmark.py --rescore runs/<run-id> --id-order file --id-labels my_labels.txt
```

The real-time factor in the report is `infer_ms / audio_ms`. The sound check refuses to start
the test until both fields are present (you can override it, and the report then marks timing
as missing).

* **Intent names** do not have to match exactly: `SET_TIMER`, `lights_on`, `get_weather`,
  `unknown`, `none`, ... are mapped to the 19 intents / out of scope. Joint names like
  `TEMPERATURE_22` or `COLOR_BLUE` are split into intent and slot. If your names are unusual,
  add them in `bench_settings.json`: `"aliases": {"AC_SET": "TEMPERATURE"}`. Unknown names are
  scored as wrong and listed in the report.
* Out of scope: print `intent=OUT_OF_SCOPE` (or `unknown`/`none`), or print nothing.
* Optional wake line, e.g. `wake word detected` or `{"event": "wake"}`: gives the wake detection rate.
* Print the line **when the decision is made** and flush it. Response latency is measured from
  the end of the spoken command to the moment the line appears.
* Totally different format? In the sound check choose `x` and enter a Python regex with named
  groups `intent`, `slot`, `infer_ms`, `audio_ms`, e.g.
  `RESULT: (?P<intent>\w+) \((?P<slot>[^)]*)\) (?P<infer_ms>[0-9.]+)ms/(?P<audio_ms>[0-9.]+)ms`.

The script measures **your assistant's** CPU and RAM by finding the process that has the log
file open. If your assistant opens the file only briefly per line, pass `--proc main.py` (a
pattern from its command line) instead.

## What the report contains

`runs/<id>/report.md` (also printed), `metrics.json` (everything), `trials.csv` (one row per
command: expected, what fired, raw log line, latency), `pi_metrics.csv` (one row per second),
`pi_specs.json`, `config.json`.

The report starts with **At a glance**: one small table with the key numbers for **overall**,
**real voice** and **synthetic voice** (intent and command accuracy, false accept, false reject,
false wake, slot exact match, latency p95), one line of Pi numbers (real-time factor, inference
time, max temperature, CPU and RAM of your assistant), and automatic warnings (unknown intent
names, missing timing fields, throttling, low response rate, extra fires). The **detailed
metrics** follow.

**Overall vs real vs synthetic voices**: every metric below at both label levels, per group. The
holdout's 10 out-of-scope clips are all real recordings, so synthetic voices have no false accept
rate ("-").

**Classification**, at the 19-intent level and at the 93-command level:
accuracy (with 95% Wilson interval), balanced accuracy, macro precision / recall / F1 / F2,
false accept rate, false reject rate, misfire rate, the most frequent confusions, per-intent
scores.

* REJECT = the clip was out of scope; on the prediction side it means the Pi said out of
  scope or did not respond.
* **False wake rate** = commands played without the wake word where the Pi fired anyway (count,
  rate, 95% interval, and which ones). Not part of the 19/93 scores.
* **False accept rate** = out-of-scope clips where the Pi fired a command. **False reject rate** =
  in-scope clips where it rejected or stayed silent. **Misfire rate** = in-scope clips where it
  fired the wrong command.
* 93-command level: the Pi predicts (intent, slot), not the wording, so a prediction matches a
  variation when intent and slot are both right. A wrong prediction counts against the first
  variation of its own (intent, slot), so it lowers that variation's precision.
* Only 10 out-of-scope clips: read the false accept rate with its confidence interval.

**Slot values** (TIMER, ALARM, TEMPERATURE, BRIGHTNESS, COLOR, CREATE_REMINDER; trials where the
intent was right):

* exact-match rate;
* **absolute error = Manhattan (L1) distance** in the slot's unit: seconds (timer), minutes on a
  24 h circle (alarm: 9 PM vs 6 AM = 540 min), degrees, percent;
* **relative error** = absolute error / spread of the 3 schema values (so slots can be compared);
* **phonetic distance**: normalised edit distance between simplified-Metaphone codes of the
  spelled-out values (0 = sounds the same, 1 = nothing in common), e.g. "blue" vs "blew" = 0;
* **character distance**: the same on the spelled-out text.

**Raspberry Pi** (mean / p95 / max over the test):

* response latency (end of command audio to Pi output; p50, p95, p99), clock-synced between
  laptop and Pi;
* inference time and **real-time factor** (infer_ms / audio_ms), from the timing fields every
  command line carries;
* CPU temperature, CPU clock (shows throttling), throttling flags (`vcgencmd get_throttled`);
* CPU use of the whole Pi and of your process; RAM of your process (RSS) and of the Pi; load average;
* runtime CPU-seconds per second of speech, and your process's CPU share of the test wall time
  (a hardware-independent "how heavy is my assistant" number);
* **FLOPs, parameters and model size** if you give your ONNX model (`--model model.onnx` or
  `--model pi:~/path/model.onnx`), plus effective GFLOP/s = FLOPs / mean inference time;
* response rate, wake detection rate, extra fires (more than one command per utterance),
  test wall time.

Not measured (needs hardware): power draw. If you have a USB power meter, note its reading
by hand.

## Tips for a fair test

* Same setup for everyone: laptop speaker about **1 m** from the Pi mic, quiet room, laptop volume
  at a normal talking level (set it in the sound check, then do not change it).
* Turn off music playback or long spoken replies on the Pi during the test (mock mode), or the
  Pi's own speaker covers the next command.
* Do not use the laptop for other audio during the test.

## Files

```
benchmark.py         the guided benchmark (run this on the laptop)
pi_agent.py          runs on the Pi: specs, metrics, log following (stdlib only, Python 3.7+)
vcmbench/            schema (19 intents, 93 variations), dataset, audio, Pi links, metrics,
                     slot distances, ONNX FLOP counter, report
tests/               pytest
```

## Privacy

The holdout set contains a classmate's voice and synthetic voices; it is the public class
dataset. Your wake word recordings stay on your laptop in `runs/<id>/wake/`; delete them at the
end of the run (step 7) if you do not want to keep them.
