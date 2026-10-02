"""Turn finished trials into metrics.json, report.md, trials.csv and a console summary."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import metrics as M
from . import schema as S
from .schema import NONE, OOS, SLOTTED
from .slots import slot_distance


def _voice(t: dict) -> str:
    return "synthetic voice" if t.get("is_synthetic") else "real voice"


def apply_id_order(trials: list[dict], meta: dict) -> dict | None:
    """For models that print a class number: map every number with the chosen order
    (meta["id_order"]) and check how well each known order fits the right answers."""
    with_id = [t for t in trials if t.get("pred_variation_id") is not None]
    if not with_id:
        return None
    orders = meta.get("id_orders") or S.id_orders()
    chosen = meta.get("id_order") or "manifest"
    if chosen not in orders:
        chosen = "manifest"
    for t in with_id:
        hit = S.lookup_id(int(t["pred_variation_id"]), orders[chosen])
        if hit is None:
            t.update(pred_intent=f"OTHER:ID_{t['pred_variation_id']}", pred_slot="", pred_variation="")
        else:
            t.update(pred_intent=hit[0], pred_slot=hit[1], pred_variation=hit[2])
    scored = [t for t in with_id if t.get("kind", "wake") == "wake"]

    def fits(order):
        n = 0
        for t in scored:
            hit = S.lookup_id(int(t["pred_variation_id"]), order)
            if hit and (hit[2] == t["true_variation"] or (hit[0] == OOS and t["true_intent"] == OOS)):
                n += 1
        return n
    fit = {name: fits(order) for name, order in orders.items()}
    best = max(fit, key=fit.get)
    return {"chosen": chosen, "n": len(scored), "matches": fit,
            "better": best if fit[best] > fit[chosen] else None}


def group_metrics(sub: list[dict], no_wake: list[dict]) -> dict:
    """All headline metrics for one group of (already scored) trials."""
    out = {"n": len(sub), "n_no_wake": len(no_wake)}
    if sub:
        I = M.classification_report([t["y_intent"][0] for t in sub], [t["y_intent"][1] for t in sub])
        C = M.classification_report([t["y_command"][0] for t in sub], [t["y_command"][1] for t in sub])
        slots = [t["slot_exact"] for t in sub if "slot_exact" in t]
        lat = M.summarize([t.get("latency_s") for t in sub])
        out.update({k: I[k] for k in ("accuracy", "accuracy_ci95", "balanced_accuracy", "macro_f1", "macro_f2",
                                      "false_accept_rate", "false_accepts", "n_out_of_scope",
                                      "false_reject_rate", "misfire_rate")})
        out.update(command_accuracy=C["accuracy"], command_balanced_accuracy=C["balanced_accuracy"],
                   command_macro_f1=C["macro_f1"], command_macro_f2=C["macro_f2"],
                   command_false_reject_rate=C["false_reject_rate"], command_misfire_rate=C["misfire_rate"],
                   slot_exact_rate=sum(slots) / len(slots) if slots else None, n_slot=len(slots),
                   latency_p50=lat.get("p50"), latency_p95=lat.get("p95"))
    if no_wake:
        fw = sum(t["false_wake"] for t in no_wake)
        out.update(false_wakes=fw, false_wake_rate=fw / len(no_wake))
    return out


def score(trials: list[dict], samples: list[dict], specs: dict, model_profile: dict | None,
          t_start: float, t_end: float, meta: dict) -> dict:
    no_wake = [t for t in trials if t.get("kind") == "no_wake"]
    trials = [t for t in trials if t.get("kind", "wake") == "wake"]
    for t in no_wake:
        t["false_wake"] = t["n_command_events"] > 0
    fw = sum(t["false_wake"] for t in no_wake)
    false_wake = {
        "n": len(no_wake), "false_wakes": fw,
        "false_wake_rate": fw / len(no_wake) if no_wake else float("nan"),
        "false_wake_ci95": M.wilson(fw, len(no_wake)),
        "wake_word_logged": sum(bool(t.get("wake_logged")) for t in no_wake),
        "fired": sorted(f"{t.get('transcript', '?')} -> {t['pred_intent']}" for t in no_wake if t["false_wake"]),
    }
    id_check = apply_id_order(trials + no_wake, meta)
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
                                    t["true_variation"] if intent_ok else None, slot_ok and intent_ok,
                                    t.get("pred_variation") or ""))
        t["y_intent"] = (yt_i[-1], yp_i[-1])
        t["y_command"] = (yt_c[-1], yp_c[-1])
        t["correct_intent"] = yt_i[-1] == yp_i[-1]
        t["correct_command"] = yt_c[-1] == yp_c[-1]

    # overall, then real vs synthetic voices; each group scored on its own
    breakdowns = {"overall": group_metrics(trials, no_wake)}
    for g in ("real voice", "synthetic voice"):
        breakdowns[g] = group_metrics([t for t in trials if _voice(t) == g], [t for t in no_wake if _voice(t) == g])

    # 93-class models: also score the exact phrase they chose (wording must match too)
    exact = None
    if any(t.get("pred_variation") for t in trials):
        yt = [M.REJECT if t["true_intent"] == OOS else t["true_variation"] for t in trials]
        yp = [t["pred_variation"] if t.get("pred_variation")
              else M.REJECT if t["pred_intent"] in (OOS, NONE)
              else f"{t['pred_intent']} (no phrase)" for t in trials]
        exact = M.classification_report(yt, yp)
        exact["lines_with_phrase"] = sum(bool(t.get("pred_variation")) for t in trials)
    any_wake_log = any(t.get("wake_logged") for t in trials)
    responded = [t for t in trials if t["n_command_events"] > 0]
    no_timing = sum(t.get("infer_ms") is None or not t.get("audio_ms") for t in responded)
    pi = M.pi_report(samples, trials, t_start, t_end)
    if model_profile and model_profile.get("flops") and pi["infer_ms"].get("n"):
        pi["effective_gflops_per_s"] = model_profile["flops"] / (pi["infer_ms"]["mean"] / 1000) / 1e9
    return {
        "meta": meta,
        "pi_specs": {k: v for k, v in specs.items() if k not in ("type", "t", "hostname")},  # no private names
        "model": model_profile,
        "intent_level": M.classification_report(yt_i, yp_i),
        "command_level": M.classification_report(yt_c, yp_c),
        "exact_wording": exact,
        "id_order_check": id_check,
        "false_wake": false_wake,
        "slots": M.slot_report(slot_rows),
        "breakdowns": breakdowns,
        "pipeline": {
            "trials": len(trials),
            "response_rate": len(responded) / len(trials) if trials else float("nan"),
            "wake_detect_rate": (sum(bool(t.get("wake_logged") or t["n_command_events"]) for t in trials)
                                 / len(trials)) if any_wake_log and trials else None,
            "no_response": sum(t["pred_intent"] == NONE for t in trials),
            "extra_fires": sum(max(t["n_command_events"] - 1, 0) for t in trials),
            "commands_missing_timing": no_timing,
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


def render_breakdowns(bd: dict) -> list[str]:
    """Overall vs real vs synthetic voices, every metric per group, at both label levels."""
    if not bd:
        return []
    groups = [(g, v) for g, v in bd.items() if v["n"] or v["n_no_wake"]]
    out = ["## Overall vs real vs synthetic voices", "",
           "Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 "
           "out-of-scope clips are all real recordings (none are synthetic), so there is no false accept "
           "rate for synthetic voices.", ""]
    rate = lambda v, key, k, n: f"{_fmt(v.get(key), True)} ({v.get(k)}/{v.get(n)})" if v.get(n) else "-"
    lines = [
        ("clips (with wake word)", lambda v: v["n"]),
        ("**19 intents** accuracy", lambda v: f"{_fmt(v.get('accuracy'), True)} {_ci(v.get('accuracy_ci95'))}"),
        ("balanced accuracy", lambda v: _fmt(v.get("balanced_accuracy"), True)),
        ("F1 (macro)", lambda v: _fmt(v.get("macro_f1"), True)),
        ("F2 (macro)", lambda v: _fmt(v.get("macro_f2"), True)),
        ("false accept rate", lambda v: rate(v, "false_accept_rate", "false_accepts", "n_out_of_scope")),
        ("false reject rate", lambda v: _fmt(v.get("false_reject_rate"), True)),
        ("misfire rate", lambda v: _fmt(v.get("misfire_rate"), True)),
        ("**93 commands** accuracy", lambda v: _fmt(v.get("command_accuracy"), True)),
        ("balanced accuracy", lambda v: _fmt(v.get("command_balanced_accuracy"), True)),
        ("F1 (macro)", lambda v: _fmt(v.get("command_macro_f1"), True)),
        ("F2 (macro)", lambda v: _fmt(v.get("command_macro_f2"), True)),
        ("misfire rate", lambda v: _fmt(v.get("command_misfire_rate"), True)),
        ("slot exact (intent right)", lambda v: f"{_fmt(v.get('slot_exact_rate'), True)} (n={v.get('n_slot')})"
                                                if v.get("n_slot") else "-"),
        ("latency p50 / p95", lambda v: f"{_fmt(v.get('latency_p50'), nd=2)} / {_fmt(v.get('latency_p95'), nd=2)} s"
                                        if v.get("latency_p50") is not None else "-"),
        ("false wake rate (no wake word)", lambda v: rate(v, "false_wake_rate", "false_wakes", "n_no_wake")),
    ]
    rows = [["metric"] + [g for g, _ in groups]]
    for label, f in lines:
        rows.append([label] + [f(v) if v["n"] or label.startswith("false wake") else "-" for _, v in groups])
    return out + _table(rows) + [""]


def render_glance(m: dict) -> list[str]:
    """Top-of-report summary: the handful of numbers to look at first, plus warnings."""
    bd = m.get("breakdowns") or {}
    groups = [(g, v) for g, v in bd.items() if v["n"] or v["n_no_wake"]]
    out = ["## At a glance", ""]
    if groups:
        cnt = lambda v, key, k, n: f"{_fmt(v.get(key), True)} ({v.get(k)}/{v.get(n)})" if v.get(n) else "-"
        lines = [
            ("intent accuracy (19)", lambda v: _fmt(v.get("accuracy"), True)),
            ("command accuracy (93)", lambda v: _fmt(v.get("command_accuracy"), True)),
            ("false accept (out of scope fired)", lambda v: cnt(v, "false_accept_rate", "false_accepts", "n_out_of_scope")),
            ("false reject (command ignored)", lambda v: _fmt(v.get("false_reject_rate"), True)),
            ("false wake (no wake word, fired)", lambda v: cnt(v, "false_wake_rate", "false_wakes", "n_no_wake")),
            ("slot exact", lambda v: _fmt(v.get("slot_exact_rate"), True)),
            ("latency p95", lambda v: f"{_fmt(v.get('latency_p95'), nd=2)} s" if v.get("latency_p95") is not None else "-"),
        ]
        rows = [[""] + [g for g, _ in groups]]
        for label, f in lines:
            rows.append([label] + [f(v) for _, v in groups])
        out += _table(rows) + [""]

    pi = m["pi"]
    g = lambda k, stat: (pi.get(k) or {}).get(stat)
    parts = []
    if g("rtf", "n"):
        parts.append(f"real-time factor {_fmt(g('rtf', 'mean'))} (p95 {_fmt(g('rtf', 'p95'))})")
    if g("infer_ms", "n"):
        parts.append(f"inference {_fmt(g('infer_ms', 'mean'), nd=0)} ms")
    if g("temp_c", "n"):
        parts.append(f"CPU temp max {_fmt(g('temp_c', 'max'), nd=1)} C")
    if g("cpu_pct_process", "n"):
        parts.append(f"runtime CPU {_fmt(g('cpu_pct_process', 'mean'), nd=0)}% mean")
    if g("rss_mb_process", "n"):
        parts.append(f"runtime RAM {_fmt(g('rss_mb_process', 'max'), nd=0)} MB peak")
    if m.get("model") and m["model"].get("flops_si"):
        parts.append(f"{m['model']['flops_si']} per inference")
    if parts:
        out += ["**Pi:** " + ", ".join(parts), ""]

    warn = []
    p = m["pipeline"]
    if p["unknown_intent_names"]:
        warn.append("unknown intent names from the Pi (scored wrong): " + ", ".join(p["unknown_intent_names"]))
    if p.get("commands_missing_timing"):
        warn.append(f"{p['commands_missing_timing']} command line(s) without infer_ms/audio_ms")
    if pi.get("throttled_flags_seen"):
        warn.append("the Pi throttled (flags " + ", ".join(pi["throttled_flags_seen"]) + "): check power/cooling")
    if p["trials"] and p["response_rate"] < 0.9:
        warn.append(f"the Pi answered only {_fmt(p['response_rate'], True)} of commands: check volume, "
                    "distance, wake word and the log path")
    ic = m.get("id_order_check")
    if ic and ic.get("better"):
        b = ic["better"]
        warn.append(f"your Pi printed class numbers, read in '{ic['chosen']}' order: only "
                    f"{ic['matches'][ic['chosen']]}/{ic['n']} matched. In '{b}' order "
                    f"{ic['matches'][b]}/{ic['n']} match, so your model probably numbers its classes that "
                    f"way. Re-score: python benchmark.py --rescore <this run folder> --id-order {b}")
    if p.get("extra_fires"):
        warn.append(f"{p['extra_fires']} extra fire(s): more than one command for one utterance")
    if warn:
        out += ["**Check:**", ""] + [f"- {w}" for w in warn] + [""]
    return out


def render_markdown(m: dict) -> str:
    out = [f"# VCM benchmark - {m['meta'].get('student') or 'student'} - {m['meta'].get('started')}", ""]
    meta = m["meta"]
    out += [f"Wake word: **{meta.get('wake_word')}** - trials: {m['pipeline']['trials']} with the wake word + "
            f"{m['false_wake']['n']} without - shuffle seed: {meta.get('seed')} - "
            f"connection: {meta.get('mode')} - holdout: {meta.get('holdout')}", ""]
    if meta.get("pi_log_file"):
        out += [f"Pi log file: {meta['pi_log_file']}", ""]
    mc = meta.get("mic_check")
    if mc:
        out += [f"Mic check (Pi input {meta.get('pi_mic')}): signal-to-noise {mc['snr_db']} dB, laptop speech "
                f"{mc['speech_dbfs']} dBFS, room noise {mc['noise_dbfs']} dBFS - {mc['verdict']}", ""]

    out += render_glance(m)
    out += ["# Detailed metrics", "", "## Classification", ""]
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
    F = m["false_wake"]
    if F["n"]:
        fw = f"{_fmt(F['false_wake_rate'], True)} {_ci(F['false_wake_ci95'])} ({F['false_wakes']}/{F['n']})"
        rows.append(["false wake rate (command without wake word fired)", fw, fw])
    out += _table(rows) + [""]

    p = m["pipeline"]
    out += [f"Responses: {_fmt(p['response_rate'], True)} of trials fired a command; "
            f"no response: {p['no_response']}; extra fires: {p['extra_fires']}; "
            f"wake detect rate: {_fmt(p['wake_detect_rate'], True)}", ""]
    if p.get("commands_missing_timing"):
        out += [f"**Timing missing:** {p['commands_missing_timing']} command line(s) had no infer_ms/audio_ms "
                "(required); inference time and real-time factor use only the lines that had them.", ""]
    if F["n"] and F["fired"]:
        out += ["**False wakes (no wake word, Pi fired):** " + "; ".join(F["fired"]), ""]
    if p["unknown_intent_names"]:
        out += [f"**Unknown intent names from the Pi (scored wrong; add aliases):** "
                f"{', '.join(p['unknown_intent_names'])}", ""]
    E = m.get("exact_wording")
    if E:
        out += [f"**93-class output:** {E['lines_with_phrase']} command line(s) named one of the 93 phrases. "
                f"Exact wording (the chosen phrase must be the spoken one): accuracy {_fmt(E['accuracy'], True)} "
                f"{_ci(E['accuracy_ci95'])}, balanced {_fmt(E['balanced_accuracy'], True)}, "
                f"F1 {_fmt(E['macro_f1'], True)}, F2 {_fmt(E['macro_f2'], True)}. The 93-command column above "
                "uses the same rule as for every student (intent + slot right), so it stays comparable.", ""]
    out += render_breakdowns(m.get("breakdowns") or {})

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
    out += [f"- **{s.get('model') or '?'}**, {s.get('cores')} cores "
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
            "rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake "
            "rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); "
            "any command the Pi fires for them is a false wake. These trials are not part of the "
            "19/93 scores.", ""]
    return "\n".join(out)


TRIAL_COLUMNS = ["order", "kind", "false_wake", "clip_idx", "accent_group", "transcript", "true_intent", "true_variation", "true_slot",
                 "pred_intent", "pred_slot", "pred_variation", "pred_variation_id", "pred_raw", "correct_intent", "correct_command",
                 "slot_exact", "slot_abs_error", "slot_phonetic_dist", "n_command_events", "wake_logged",
                 "latency_s", "infer_ms", "audio_ms", "speaker_id", "is_synthetic", "wake_take", "audio_file"]


def write_outputs(run_dir: Path, m: dict, trials: list[dict], samples: list[dict]) -> Path:
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    md = render_markdown(m)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    with open(run_dir / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(trials)
    if samples:
        keys = ["t", "temp_c", "cpu_pct", "freq_mhz", "load1", "mem_used_mb", "mem_avail_mb", "throttled"]
        with open(run_dir / "pi_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys + ["proc_cpu_pct", "proc_rss_mb", "proc_cpu_time_s"])
            for s in samples:
                p = s.get("proc") or {}
                w.writerow([s.get(k) for k in keys] + [p.get("cpu_pct"), p.get("rss_mb"), p.get("cpu_time_s")])
    return run_dir / "report.md"
