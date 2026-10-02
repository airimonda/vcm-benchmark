"""Scoring: classification metrics at intent (19 + reject) and command (93 + reject) level,
slot-value distances, and Pi resource / latency summaries."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .schema import NONE, OOS, REJECT, SLOTTED, variations_for


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return 0.0 if p + r == 0 else (1 + b2) * p * r / (b2 * p + r)


def classification_report(y_true: list[str], y_pred: list[str], reject: str = REJECT) -> dict:
    """Metrics for one label level.

    * `reject` is the "nothing in scope" class (out-of-scope truth; out-of-scope
      or no-response prediction). It counts as a normal class for accuracy,
      balanced accuracy and the macro scores.
    * Macro precision / recall / F1 / F2 average over the classes present in
      the truth, so a class the test set never contains can't help or hurt.
    * false_accept_rate: out-of-scope trials where the Pi fired a command.
    * false_reject_rate: in-scope trials where the Pi rejected or stayed silent.
    * misfire_rate: in-scope trials where the Pi fired the wrong command.
    """
    n = len(y_true)
    classes = sorted(set(y_true))
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    per = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per[c] = {"support": tp[c] + fn[c], "precision": prec, "recall": rec,
                  "f1": fbeta(prec, rec, 1), "f2": fbeta(prec, rec, 2)}
    correct = sum(tp.values())
    oos = [(t, p) for t, p in zip(y_true, y_pred) if t == reject]
    ins = [(t, p) for t, p in zip(y_true, y_pred) if t != reject]
    fa = sum(p != reject for _, p in oos)
    fr = sum(p == reject for _, p in ins)
    mis = sum(p != reject and p != t for t, p in ins)
    mean = lambda k: float(np.mean([v[k] for v in per.values()])) if per else float("nan")
    return {
        "n": n,
        "accuracy": correct / n if n else float("nan"),
        "accuracy_ci95": wilson(correct, n),
        "balanced_accuracy": mean("recall"),
        "macro_precision": mean("precision"),
        "macro_recall": mean("recall"),
        "macro_f1": mean("f1"),
        "macro_f2": mean("f2"),
        "false_accept_rate": fa / len(oos) if oos else float("nan"),
        "false_accept_ci95": wilson(fa, len(oos)),
        "false_accepts": fa, "n_out_of_scope": len(oos),
        "false_reject_rate": fr / len(ins) if ins else float("nan"),
        "misfire_rate": mis / len(ins) if ins else float("nan"),
        "per_class": per,
        "confusions": Counter((t, p) for t, p in zip(y_true, y_pred) if t != p).most_common(15),
    }


def intent_label(intent: str) -> str:
    return REJECT if intent in (OOS, NONE, "") else intent


def command_label(intent: str, slot: str, truth_variation: str | None, slot_ok: bool,
                  pred_variation: str = "") -> str:
    """93-way label of a prediction.

    The Pi predicts (intent, slot), not the wording, so a prediction counts as
    the true variation when intent and slot both match. A wrong prediction is
    mapped to the first variation of its own (intent, slot) — e.g. TEMPERATURE
    22 -> "Temperature 22 degrees" — so it shows up as a false positive there.
    """
    if intent in (OOS, NONE, ""):
        return REJECT
    if truth_variation is not None and slot_ok:
        return truth_variation
    if intent.startswith("OTHER:"):
        return intent
    if pred_variation:                     # 93-class model: the phrase it actually chose
        return pred_variation
    if intent in SLOTTED:
        from .slots import SLOT_VALUES, slot_distance
        for value in SLOT_VALUES[intent]:
            if slot and slot_distance(intent, value, slot)["exact"]:
                return variations_for(intent, value)[0].phrase
        return f"{intent}:{slot or '?'}"
    return variations_for(intent)[0].phrase


def summarize(values, pct=(50, 95, 99)) -> dict:
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                 dtype=float)
    if len(v) == 0:
        return {"n": 0}
    out = {"n": int(len(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    for p in pct:
        out[f"p{p}"] = float(np.percentile(v, p))
    return out


def slot_report(rows: list[dict]) -> dict:
    """rows: trials whose truth is a slotted intent and whose predicted intent is right.
    Each row carries the slot_distance() dict under "slot"."""
    by = defaultdict(list)
    for r in rows:
        by[r["intent"]].append(r["slot"])
    out = {}
    for intent, ds in sorted(by.items()):
        out[intent] = {
            "n": len(ds),
            "exact_rate": float(np.mean([d["exact"] for d in ds])),
            "mean_abs_error": _mean([d.get("abs_error") for d in ds]),
            "unit": next((d.get("unit") for d in ds if d.get("unit")), None),
            "mean_rel_error": _mean([d.get("rel_error") for d in ds]),
            "mean_phonetic_dist": _mean([d.get("phonetic_dist") for d in ds]),
            "mean_char_dist": _mean([d.get("char_dist") for d in ds]),
        }
    all_ds = [r["slot"] for r in rows]
    out["ALL"] = {
        "n": len(all_ds),
        "exact_rate": _mean([d["exact"] for d in all_ds]),
        "mean_rel_error": _mean([d.get("rel_error") for d in all_ds]),
        "mean_phonetic_dist": _mean([d.get("phonetic_dist") for d in all_ds]),
        "mean_char_dist": _mean([d.get("char_dist") for d in all_ds]),
    }
    return out


def _mean(xs) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return float(np.mean(v)) if v else None


def pi_report(samples: list[dict], trials: list[dict], t_start: float, t_end: float) -> dict:
    """Resource use during the test, plus latency / real-time factor from the trials."""
    s = [m for m in samples if t_start <= m.get("t", 0) <= t_end] or samples
    g = lambda k: [m.get(k) for m in s]
    proc = [m.get("proc") or {} for m in s]
    throttled = sorted({m.get("throttled") for m in s if m.get("throttled") not in (None, "0x0")})
    cpu_times = [p.get("cpu_time_s") for p in proc if p.get("cpu_time_s") is not None]
    speech_s = sum(t["cmd_end"] - t["cmd_start"] for t in trials if t.get("cmd_end"))
    wall = t_end - t_start
    lat = [t.get("latency_s") for t in trials]
    infer = [t.get("infer_ms") for t in trials]
    rtf = [t["infer_ms"] / t["audio_ms"] for t in trials
           if t.get("infer_ms") is not None and t.get("audio_ms")]
    proc_cpu_s = (max(cpu_times) - min(cpu_times)) if len(cpu_times) > 1 else None
    return {
        "samples": len(s),
        "temp_c": summarize(g("temp_c")),
        "cpu_pct_system": summarize(g("cpu_pct")),
        "cpu_pct_process": summarize([p.get("cpu_pct") for p in proc]),
        "rss_mb_process": summarize([p.get("rss_mb") for p in proc]),
        "mem_used_mb_system": summarize(g("mem_used_mb")),
        "freq_mhz": summarize(g("freq_mhz")),
        "load1": summarize(g("load1")),
        "throttled_flags_seen": throttled,
        "response_latency_s": summarize(lat),
        "infer_ms": summarize(infer),
        "rtf": summarize(rtf),
        "process_cpu_seconds": proc_cpu_s,
        "cpu_seconds_per_speech_second": proc_cpu_s / speech_s if proc_cpu_s and speech_s else None,
        "process_cpu_share_of_wall": proc_cpu_s / wall if proc_cpu_s and wall > 0 else None,
        "test_wall_time_s": wall,
        "speech_seconds_played": speech_s,
    }
