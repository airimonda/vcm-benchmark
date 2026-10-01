"""Turn finished trials into metrics.json, report.md, trials.csv and a console summary."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import metrics as M
from .schema import NONE, OOS, SLOTTED
from .slots import slot_distance


def score(trials: list[dict], samples: list[dict], specs: dict, model_profile: dict | None,
          t_start: float, t_end: float, meta: dict) -> dict:
    yt_i, yp_i, yt_c, yp_c, slot_rows = [], [], [], [], []
    for t in trials:
        truth_oos = t["true_intent"] == OOS
        pred = t["pred_intent"]
        slot_ok = True
        if not truth_oos and t["true_intent"] in SLOTTED and pred == t["true_intent"]:
            d = slot_distance(pred, t["true_slot"], t.get("pred_slot") or "")
            t["slot_exact"] = bool(d["exact"])
            t["slot_abs_error"] = d.get("abs_error")
            t["slot_phonetic_dist"] = d.get("phonetic_dist")
            slot_ok = bool(d["exact"])
            slot_rows.append({"intent": pred, "slot": d})
        yt_i.append(M.intent_label(t["true_intent"]))
        yp_i.append(M.intent_label(pred))
        yt_c.append(M.REJECT if truth_oos else t["true_variation"])
        intent_ok = pred == t["true_intent"]
        yp_c.append(M.command_label(pred, t.get("pred_slot") or "",
                                    t["true_variation"] if intent_ok else None, slot_ok and intent_ok))
        t["correct_intent"] = yt_i[-1] == yp_i[-1]
        t["correct_command"] = yt_c[-1] == yp_c[-1]

    by_voice = {}
    for name, keep in (("real voice", lambda t: not t.get("is_synthetic")),
                       ("synthetic voice", lambda t: t.get("is_synthetic"))):
        sub = [t for t in trials if keep(t)]
        if sub:
            by_voice[name] = {"n": len(sub),
                              "intent_accuracy": sum(t["correct_intent"] for t in sub) / len(sub),
                              "command_accuracy": sum(t["correct_command"] for t in sub) / len(sub)}

    any_wake_log = any(t.get("wake_logged") for t in trials)
    responded = [t for t in trials if t["n_command_events"] > 0]
    pi = M.pi_report(samples, trials, t_start, t_end)
    if model_profile and model_profile.get("flops") and pi["infer_ms"].get("n"):
        pi["effective_gflops_per_s"] = model_profile["flops"] / (pi["infer_ms"]["mean"] / 1000) / 1e9
    return {
        "meta": meta,
        "pi_specs": {k: v for k, v in specs.items() if k not in ("type", "t")},
        "model": model_profile,
        "intent_level": M.classification_report(yt_i, yp_i),
        "command_level": M.classification_report(yt_c, yp_c),
        "slots": M.slot_report(slot_rows),
        "by_voice": by_voice,
        "pipeline": {
            "trials": len(trials),
            "response_rate": len(responded) / len(trials) if trials else float("nan"),
            "wake_detect_rate": (sum(bool(t.get("wake_logged") or t["n_command_events"]) for t in trials)
                                 / len(trials)) if any_wake_log and trials else None,
            "no_response": sum(t["pred_intent"] == NONE for t in trials),
            "extra_fires": sum(max(t["n_command_events"] - 1, 0) for t in trials),
            "unknown_intent_names": sorted({t["pred_intent"] for t in trials
                                            if t["pred_intent"].startswith("OTHER:")}),
        },
        "pi": pi,
    }


# ------------------------------------------------------------------ output

def _fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if pct:
        return f"{100 * v:.1f}%"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ci(ci):
    return "-" if ci is None or any(math.isnan(x) for x in ci) else f"[{100*ci[0]:.0f}-{100*ci[1]:.0f}%]"


def _table(rows: list[list[str]]) -> list[str]:
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return [line(rows[0]), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows[1:]]


def render_markdown(m: dict) -> str:
    out = [f"# VCM benchmark - {m['meta'].get('student') or 'student'} - {m['meta'].get('started')}", ""]
    meta = m["meta"]
    out += [f"Wake word: **{meta.get('wake_word')}** - trials: {m['pipeline']['trials']} - "
            f"connection: {meta.get('mode')} - holdout: {meta.get('holdout')}", ""]

    out += ["## Classification", ""]
    rows = [["metric", "19 intents (+reject)", "93 commands (+reject)"]]
    I, C = m["intent_level"], m["command_level"]
    for key, label, pct in [("accuracy", "accuracy", True), ("balanced_accuracy", "balanced accuracy", True),
                            ("macro_precision", "precision (macro)", True), ("macro_recall", "recall (macro)", True),
                            ("macro_f1", "F1 (macro)", True), ("macro_f2", "F2 (macro)", True),
                            ("false_accept_rate", "false accept rate (OOS fired)", True),
                            ("false_reject_rate", "false reject rate (in-scope silent/rejected)", True),
                            ("misfire_rate", "misfire rate (wrong command fired)", True)]:
        rows.append([label, _fmt(I[key], pct), _fmt(C[key], pct)])
    rows.append(["accuracy 95% CI", _ci(I["accuracy_ci95"]), _ci(C["accuracy_ci95"])])
    rows.append(["false accept 95% CI", f"{_ci(I['false_accept_ci95'])} ({I['false_accepts']}/{I['n_out_of_scope']})",
                 f"{_ci(C['false_accept_ci95'])}"])
    out += _table(rows) + [""]

    p = m["pipeline"]
    out += [f"Responses: {_fmt(p['response_rate'], True)} of trials fired a command; "
            f"no response: {p['no_response']}; extra fires: {p['extra_fires']}; "
            f"wake detect rate: {_fmt(p['wake_detect_rate'], True)}", ""]
    if p["unknown_intent_names"]:
        out += [f"**Unknown intent names from the Pi (scored wrong; add aliases):** "
                f"{', '.join(p['unknown_intent_names'])}", ""]
    if m["by_voice"]:
        out += _table([["voice", "n", "intent acc", "command acc"]] +
                      [[k, v["n"], _fmt(v["intent_accuracy"], True), _fmt(v["command_accuracy"], True)]
                       for k, v in m["by_voice"].items()]) + [""]

    out += ["## Slot values (slotted intents, intent right)", "",
            "abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); "
            "rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised "
            "edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.", ""]
    rows = [["intent", "n", "exact", "mean abs error", "mean rel error", "phonetic dist", "char dist"]]
    for k, v in m["slots"].items():
        rows.append([k, v["n"], _fmt(v["exact_rate"], True),
                     f"{_fmt(v.get('mean_abs_error'), nd=1)} {v.get('unit') or ''}".strip(),
                     _fmt(v.get("mean_rel_error")), _fmt(v.get("mean_phonetic_dist")), _fmt(v.get("mean_char_dist"))])
    out += (_table(rows) if len(rows) > 1 else ["(no slotted trials with the right intent)"]) + [""]

    out += ["## Raspberry Pi", ""]
    s = m["pi_specs"]
    pk = s.get("packages") or {}
    out += [f"- **{s.get('model') or '?'}** ({s.get('hostname')}), {s.get('cores')} cores "
            f"{s.get('cpu_model') or ''} up to {s.get('max_freq_mhz')} MHz, RAM {s.get('ram_mb')} MB, "
            f"{s.get('os')}, kernel {s.get('kernel')}, Python {s.get('python')}",
            "- packages: " + (", ".join(f"{k} {v}" for k, v in pk.items() if v) or "-"), ""]
    pi = m["pi"]
    d = lambda k, unit="", nd=1: (f"{_fmt(pi[k].get('mean'), nd=nd)} / {_fmt(pi[k].get('p95'), nd=nd)} / "
                                  f"{_fmt(pi[k].get('max'), nd=nd)} {unit}") if pi[k].get("n") else "-"
    rows = [["metric", "mean / p95 / max"],
            ["response latency (command end -> Pi output)", d("response_latency_s", "s", 3)],
            ["latency p50 / p99", f"{_fmt(pi['response_latency_s'].get('p50'))} / "
                                  f"{_fmt(pi['response_latency_s'].get('p99'))} s" if pi["response_latency_s"].get("n") else "-"],
            ["inference time (Pi-reported)", d("infer_ms", "ms")],
            ["real-time factor (infer / audio window)", d("rtf", "", 3)],
            ["CPU temperature", d("temp_c", "C")],
            ["CPU use, whole Pi", d("cpu_pct_system", "%")],
            ["CPU use, your runtime process", d("cpu_pct_process", "%")],
            ["RAM (RSS), your runtime process", d("rss_mb_process", "MB")],
            ["RAM used, whole Pi", d("mem_used_mb_system", "MB")],
            ["CPU clock", d("freq_mhz", "MHz", 0)],
            ["load average (1 min)", d("load1", "", 2)],
            ["runtime CPU-seconds per second of speech", _fmt(pi.get("cpu_seconds_per_speech_second"))],
            ["runtime CPU share of wall time", _fmt(pi.get("process_cpu_share_of_wall"), True)],
            ["throttling flags seen", ", ".join(pi["throttled_flags_seen"]) or "none"],
            ["test wall time", f"{pi['test_wall_time_s'] / 60:.1f} min"]]
    if m.get("model"):
        mp = m["model"]
        rows += [["model parameters", f"{mp.get('params'):,}"], ["model size", f"{mp.get('size_mb'):.2f} MB"],
                 ["model FLOPs per inference", mp.get("flops_si", "-")]]
        if pi.get("effective_gflops_per_s"):
            rows.append(["effective GFLOP/s (FLOPs / mean infer time)", f"{pi['effective_gflops_per_s']:.2f}"])
    out += _table(rows) + [""]

    out += ["## Most frequent confusions", ""]
    for lvl in ("intent_level", "command_level"):
        conf = m[lvl]["confusions"]
        out += [f"**{lvl.replace('_', ' ')}:** " +
                ("; ".join(f"{t} -> {p} ({n})" for (t, p), n in conf[:10]) or "none"), ""]
    out += ["## Per-intent scores", ""]
    rows = [["class", "n", "precision", "recall", "F1", "F2"]]
    for c, v in sorted(I["per_class"].items()):
        rows.append([c, v["support"], _fmt(v["precision"], True), _fmt(v["recall"], True),
                     _fmt(v["f1"], True), _fmt(v["f2"], True)])
    out += _table(rows) + [""]
    out += ["Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. "
            "Command level: a prediction matches a variation when intent and slot are right (the Pi does "
            "not predict the wording); wrong predictions count against the first variation of their "
            "(intent, slot). Macro scores average over classes present in the holdout. False accept rate "
            "rests on only the out-of-scope clips in the holdout, so read its confidence interval.", ""]
    return "\n".join(out)


TRIAL_COLUMNS = ["order", "clip_idx", "transcript", "true_intent", "true_variation", "true_slot",
                 "pred_intent", "pred_slot", "pred_raw", "correct_intent", "correct_command",
                 "slot_exact", "slot_abs_error", "slot_phonetic_dist", "n_command_events", "wake_logged",
                 "latency_s", "infer_ms", "audio_ms", "speaker_id", "is_synthetic", "wake_take", "audio_file"]


def write_outputs(run_dir: Path, m: dict, trials: list[dict], samples: list[dict]) -> Path:
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    md = render_markdown(m)
    (run_dir / "report.md").write_text(md)
    with open(run_dir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(trials)
    if samples:
        keys = ["t", "temp_c", "cpu_pct", "freq_mhz", "load1", "mem_used_mb", "mem_avail_mb", "throttled"]
        with open(run_dir / "pi_metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["proc_cpu_pct", "proc_rss_mb", "proc_cpu_time_s"])
            for s in samples:
                p = s.get("proc") or {}
                w.writerow([s.get(k) for k in keys] + [p.get("cpu_pct"), p.get("rss_mb"), p.get("cpu_time_s")])
    return run_dir / "report.md"
