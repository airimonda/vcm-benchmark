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

## Install (laptop)

Python 3.10 or newer.

```bash
git clone https://github.com/airimonda/vcm-benchmark.git && cd vcm-benchmark
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Linux laptops also need PortAudio: `sudo apt install libportaudio2`.
On the Pi you need nothing extra: `pi_agent.py` uses only the Python standard library.

## Run

```bash
python benchmark.py                 # guided run; answer the questions
python benchmark.py --mode sim --no-audio --size quick --limit 10 --gap-min 1 --gap-max 2
                                    # dry run without a Pi, to see what happens
```

The script walks you through these steps:

1. **Connect to your Pi** and read its specs (model, CPU, RAM, OS, Python packages, microphones).
2. **Record your wake word** 3 times on the laptop microphone (or `--wake-files a.wav b.wav`).
3. **Build the test audio**: each holdout clip gets one of your wake word takes in front of it,
   with a pause in between (default 0.8 s; set it to what your Pi needs after its chime).
   Wake word and command are levelled to the same loudness.
4. **Sound check**: two warm-up commands, not scored. You see the log lines your Pi printed and
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

Test size: **full** = all 196 holdout clips (186 commands = 2 per variation, plus 10 out-of-scope),
about 55 min. **quick** = 1 clip per variation + the 10 out-of-scope clips (103), about 30 min.

The holdout set downloads automatically from Hugging Face
([airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
split `holdout`, 15 MB) and is cached in `.cache/`. Use `--holdout path/to/dataset/holdout` for a
local copy.

Your answers are saved in `bench_settings.json`, so the next run asks fewer questions
(`--fresh` to start over). Interrupted? `python benchmark.py --resume runs/<run-id>`.

## Connect your Pi however you like

| Option | Use when | What you give |
|---|---|---|
| **1. SSH** (recommended) | The laptop can `ssh` to the Pi: same Wi-Fi/LAN, `raspberrypi.local`, Tailscale, USB-ethernet gadget, hotspot | `user@host` or an `~/.ssh/config` alias |
| **2. Pi to laptop (HTTP)** | The laptop cannot reach the Pi (e.g. router client isolation), but the Pi can reach the laptop | Run the printed `python3 pi_agent.py --post http://LAPTOP_IP:8765 ...` on the Pi |
| **3. Manual** | No network at all | After each command you type what the Pi did (`14 30 seconds`, `TIMER 30 seconds`, Enter = nothing) |
| **4. Simulation** | No Pi; just trying the script | nothing |

For SSH, the script copies `pi_agent.py` to `~/.vcm_bench/` on the Pi and runs it there. One SSH
connection is shared, so you type a password at most once (`ssh-copy-id user@host` avoids it).
Extra ssh options: `--ssh-opt=-p2222 --ssh-opt=-i~/.ssh/mykey`.

## What your Pi must print

Your assistant must write **one line per recognised command** to a log file (or to a systemd
journal / any command output you can follow). Any of these work out of the box:

```
{"intent": "TIMER", "slot": "30 seconds", "infer_ms": 85, "audio_ms": 1500}
intent=TIMER slot=30 seconds infer_ms=85
2026-10-02 12:00:01 INFO command: LIGHT_ON
prediction=set_temperature_22
intent=OUT_OF_SCOPE
```

* **Intent names** do not have to match exactly: `SET_TIMER`, `lights_on`, `get_weather`,
  `unknown`, `none`, ... are mapped to the 19 intents / out of scope. Joint names like
  `TEMPERATURE_22` or `COLOR_BLUE` are split into intent and slot. If your names are unusual,
  add them in `bench_settings.json`: `"aliases": {"AC_SET": "TEMPERATURE"}`. Unknown names are
  scored as wrong and listed in the report.
* **Slot** is free text: `22`, `22 degrees`, `twenty two`, `9pm`, `21:00`, `blue`, `drink water`.
* `infer_ms` (model time) and `audio_ms` (length of the audio window the model saw) are optional;
  with them the report shows inference time and real-time factor.
* Optional wake line, e.g. `wake word detected` or `{"event": "wake"}`: gives the wake detection rate.
* Print the line **when the decision is made** (flush the file: `print(..., flush=True)` or
  `logging` with a `FileHandler`). Response latency is measured from the end of the spoken
  command to the moment the line appears.
* Totally different format? In the sound check choose `x` and enter a Python regex with named
  groups, e.g. `RESULT: (?P<intent>\w+) \((?P<slot>[^)]*)\)`.

Tell the script how to find your process (e.g. `main.py`) to get **its** CPU and RAM, not just the
whole Pi's.

## What the report contains

`runs/<id>/report.md` (also printed), `metrics.json` (everything), `trials.csv` (one row per
command: expected, what fired, raw log line, latency), `pi_metrics.csv` (one row per second),
`pi_specs.json`, `config.json`.

**Classification**, at the 19-intent level and at the 93-command level:
accuracy (with 95% Wilson interval), balanced accuracy, macro precision / recall / F1 / F2,
false accept rate, false reject rate, misfire rate, the most frequent confusions, per-intent
scores, and the split between real and synthetic voices.

* REJECT = the clip was out of scope; on the prediction side it means the Pi said out of
  scope or did not respond.
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
* inference time and **real-time factor** (infer_ms / audio_ms), if your Pi prints them;
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
